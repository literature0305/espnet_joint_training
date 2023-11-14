"""V2 backend for `asr_recog.py` using py:class:`espnet.nets.beam_search.BeamSearch`."""

import json
import logging

import torch
from packaging.version import parse as V

from espnet.asr.asr_utils import add_results_to_json, get_model_conf, torch_load
from espnet.asr.pytorch_backend.asr import load_trained_model
from espnet.nets.asr_interface import ASRInterface
from espnet.nets.batch_beam_search import BatchBeamSearch
from espnet.nets.beam_search_cali import BeamSearch
from espnet.nets.lm_interface import dynamic_import_lm
from espnet.nets.scorer_interface import BatchScorerInterface
from espnet.nets.scorers.length_bonus import LengthBonus
from espnet.utils.deterministic_utils import set_deterministic_pytorch
from espnet.utils.io_utils import LoadInputsAndTargets
import Levenshtein
import numpy as np
from npy_append_array import NpyAppendArray


def recog_v2(args):
    """Decode with custom models that implements ScorerInterface.

    Notes:
        The previous backend espnet.asr.pytorch_backend.asr.recog
        only supports E2E and RNNLM

    Args:
        args (namespace): The program arguments.
        See py:func:`espnet.bin.asr_recog.get_parser` for details

    """
    logging.warning("experimental API for custom LMs is selected by --api v2")
    if args.batchsize > 1:
        raise NotImplementedError("multi-utt batch decoding is not implemented")
    if args.streaming_mode is not None:
        raise NotImplementedError("streaming mode is not implemented")
    if args.word_rnnlm:
        raise NotImplementedError("word LM is not implemented")

    set_deterministic_pytorch(args)
    model, train_args = load_trained_model(args.model)
    assert isinstance(model, ASRInterface)

    if args.quantize_config is not None:
        q_config = set([getattr(torch.nn, q) for q in args.quantize_config])
    else:
        q_config = {torch.nn.Linear}

    if args.quantize_asr_model:
        logging.info("Use quantized asr model for decoding")

        # See https://github.com/espnet/espnet/pull/3616 for more information.
        if (
            V(torch.__version__) < V("1.4.0")
            and "lstm" in train_args.etype
            and torch.nn.LSTM in q_config
        ):
            raise ValueError(
                "Quantized LSTM in ESPnet is only supported with torch 1.4+."
            )

        if args.quantize_dtype == "float16" and V(torch.__version__) < V("1.5.0"):
            raise ValueError(
                "float16 dtype for dynamic quantization is not supported with torch "
                "version < 1.5.0. Switching to qint8 dtype instead."
            )

        dtype = getattr(torch, args.quantize_dtype)

        model = torch.quantization.quantize_dynamic(model, q_config, dtype=dtype)

    model.eval()
    load_inputs_and_targets = LoadInputsAndTargets(
        mode="asr",
        load_output=True,
        sort_in_input_length=False,
        preprocess_conf=train_args.preprocess_conf
        if args.preprocess_conf is None
        else args.preprocess_conf,
        preprocess_args={"train": False},
    )

    if args.rnnlm:
        lm_args = get_model_conf(args.rnnlm, args.rnnlm_conf)
        # NOTE: for a compatibility with less than 0.5.0 version models
        lm_model_module = getattr(lm_args, "model_module", "default")
        lm_class = dynamic_import_lm(lm_model_module, lm_args.backend)
        lm = lm_class(len(train_args.char_list), lm_args)
        torch_load(args.rnnlm, lm)
        if args.quantize_lm_model:
            logging.info("Use quantized lm model")
            dtype = getattr(torch, args.quantize_dtype)
            lm = torch.quantization.quantize_dynamic(lm, q_config, dtype=dtype)
        lm.eval()
    else:
        lm = None

    if args.ngram_model:
        from espnet.nets.scorers.ngram import NgramFullScorer, NgramPartScorer

        if args.ngram_scorer == "full":
            ngram = NgramFullScorer(args.ngram_model, train_args.char_list)
        else:
            ngram = NgramPartScorer(args.ngram_model, train_args.char_list)
    else:
        ngram = None

    scorers = model.scorers()
    scorers["lm"] = lm
    scorers["ngram"] = ngram
    scorers["length_bonus"] = LengthBonus(len(train_args.char_list))
    weights = dict(
        decoder=args.aed_weight,
        ctc=args.ctc_weight,
        lm=args.lm_weight,
        ngram=args.ngram_weight,
        length_bonus=args.penalty,
    )
    beam_search = BeamSearch(
        beam_size=args.beam_size,
        vocab_size=len(train_args.char_list),
        weights=weights,
        scorers=scorers,
        sos=model.sos,
        eos=model.eos,
        token_list=train_args.char_list,
        pre_beam_score_key=None if args.ctc_weight == 1.0 else "full",
    )
    # TODO(karita): make all scorers batchfied
    if args.batchsize == 1:
        non_batch = [
            k
            for k, v in beam_search.full_scorers.items()
            if not isinstance(v, BatchScorerInterface)
        ]
        if len(non_batch) == 0:
            beam_search.__class__ = BatchBeamSearch
            logging.info("BatchBeamSearch implementation is selected.")
        else:
            logging.warning(
                f"As non-batch scorers {non_batch} are found, "
                f"fall back to non-batch implementation."
            )

    if args.ngpu > 1:
        raise NotImplementedError("only single GPU decoding is supported")
    if args.ngpu == 1:
        device = "cuda"
    else:
        device = "cpu"
    dtype = getattr(torch, args.dtype)
    logging.info(f"Decoding device={device}, dtype={dtype}")
    model.to(device=device, dtype=dtype).eval()
    beam_search.to(device=device, dtype=dtype).eval()

    # read json data
    with open(args.recog_json, "rb") as f:
        js = json.load(f)["utts"]
    new_js = {}
    with open(args.name_ece_log_file, "a") as f_lev:
        with torch.no_grad():
            confid_accum = None
            confid_accum_ctc = None
            targets_accum = None
            for idx, name in enumerate(js.keys(), 1):
                logging.info("(%d/%d) decoding " + name, idx, len(js.keys()))
                batch = [(name, js[name])]
                feat = load_inputs_and_targets(batch)[0][0]
                target = load_inputs_and_targets(batch)[1][0]
                enc = model.encode(torch.as_tensor(feat).to(device=device, dtype=dtype))
                nbest_hyps = beam_search(
                    x=enc, maxlenratio=args.maxlenratio, minlenratio=args.minlenratio
                )
                if len(nbest_hyps) > 0:
                    confidence = nbest_hyps[0].confidence[:len(nbest_hyps[0].yseq)]
                    confidence_ctc = nbest_hyps[0].confidence_ctc[:len(nbest_hyps[0].yseq)]
                else:
                    raise ValueError('ERROR:Non nbest_hyps')
                    confidence = None
                    confidence_ctc = None
                nbest_hyps = [
                    h.asdict() for h in nbest_hyps[: min(len(nbest_hyps), args.nbest)]
                ]
                new_js[name] = add_results_to_json(
                    js[name], nbest_hyps, train_args.char_list
                )

                if confidence is not None:
                    conf_argmax=torch.argmax(confidence,dim=-1).tolist()
                    target_with_eos = target.tolist()
                    target_with_eos.append(conf_argmax[-2])
                    target_with_eos.append(conf_argmax[-1])
                    edit_log=Levenshtein.editops(target_with_eos,conf_argmax)
                    target_after=list(target_with_eos)
                    idx_conf = list(range(0, len(conf_argmax)))
                    idx_conf_after=list(idx_conf)

                    # write Levenshtein log
                    edit_log_seq=[]
                    for i in range(len(target_after)):
                        edit_log_seq.append('C')    

                    for ele in edit_log:
                        err=ele[0]
                        position_a = ele[1]
                        position_b = ele[2]
                        if err == 'insert':
                            target_after.insert(position_a, target_with_eos[position_a])
                            edit_log_seq.insert(position_a, 'I')
                        elif err == 'delete':
                            idx_conf_after.insert(position_b, idx_conf[position_b])
                            edit_log_seq[position_a]='D'

                    confidence_after = confidence[idx_conf_after]
                    confidence_after_ctc = confidence_ctc[idx_conf_after]
                    confidence_after_argmax = torch.argmax(confidence_after, dim=-1).tolist()

                    edit_log_seq_final = []
                    for i in range(len(target_after)):
                        if target_after[i] != confidence_after_argmax[i] and edit_log_seq[i]=='C':
                            edit_log_seq_final.append('S')
                        else:
                            edit_log_seq_final.append(edit_log_seq[i])

                    f_lev.write(str(edit_log_seq_final).replace(',', '').replace('[','').replace(']','').replace("'", '') + '\n')

                    if confid_accum is None:
                        confid_accum = confidence_after
                        confid_accum_ctc = confidence_after_ctc
                    else:
                        confid_accum = torch.cat((confid_accum, confidence_after), 0)
                        confid_accum_ctc = torch.cat((confid_accum_ctc, confidence_after_ctc), 0)

                    if targets_accum is None:
                        targets_accum = target_after
                    else:
                        targets_accum.extend(target_after)

                    assert len(target_after) == len(confidence_after)
                    assert len(targets_accum) == len(confid_accum)
                    assert len(targets_accum) == len(confid_accum_ctc)

            file_name_am1='ece_am_target.npy'
            file_name_am2='ece_am.npy'
            file_name_am3='ece_ctc.npy'
            with NpyAppendArray(file_name_am1) as npaa:
                npaa.append(np.array(targets_accum))
            with NpyAppendArray(file_name_am2) as npaa:
                npaa.append(confid_accum.numpy())
            with NpyAppendArray(file_name_am3) as npaa:
                npaa.append(confid_accum_ctc.numpy())



    with open(args.result_label, "wb") as f:
        f.write(
            json.dumps(
                {"utts": new_js}, indent=4, ensure_ascii=False, sort_keys=True
            ).encode("utf_8")
        )