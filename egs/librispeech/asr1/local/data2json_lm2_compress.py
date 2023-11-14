import sys
import json

if len(sys.argv) != 4:
    raise ValueError('need input file: python split_tts_data_in_json.py aa.json')

input_text_dir=sys.argv[1]
input_idx_dir=sys.argv[2]
input_json_dir=sys.argv[3]
output_json_dir_paired=input_idx_dir.replace('.int','.json')

with open(input_idx_dir, 'r') as f:
    lines_idx = f.readlines()
with open(input_text_dir, 'r') as f:
    lines_txt = f.readlines()
with open(input_json_dir, "r", encoding='utf8') as json_file:
    json_data = json.load(json_file)

json_towrite = {}
json_towrite['utts'] = {}

for utt in json_data['utts']:
    input_example = json_data['utts'][utt]['input']
    num_class = json_data['utts'][utt]['output'][0]['shape'][1]
    dim_feature = json_data['utts'][utt]['input'][0]['shape'][1]
    break

for idx, line in enumerate(zip(lines_txt, lines_idx)):
    line_txt, line_idx = line
    json_towrite['utts'][idx] = {}
    json_towrite['utts'][idx]['input']=[]
    json_towrite['utts'][idx]['output']=[]
    json_towrite['utts'][idx]['utt2spk']=str(idx)

    output_dict = {}
    output_dict['name']='target1'
    output_dict['shape']=[len(line_idx.split()), num_class]
    output_dict['tokenid']=line_idx.replace('\n','')

    input_dict = {}
    input_dict['shape']=[0, dim_feature]

    json_towrite['utts'][idx]['output'].append(output_dict)
    json_towrite['utts'][idx]['input'].append(input_dict)

# write
with open(output_json_dir_paired, 'w', encoding='utf8') as out_file:
    json.dump(json_towrite, out_file, indent=4, ensure_ascii=False)