#!/usr/bin/env bash

# Set the number of times to run the script and the arguments to pass
am_w=(0.3 0.5 0.7)
ctc_w=0.5
lm_w=0.9

N=${#am_w[@]}

# Define the function that will run the script recursively
function run_recursive {
    # Get the number of arguments
    num_args=$1
    shift
    # If there are no more arguments, exit the function
    if [[ $num_args -eq 0 ]]; then
        return
    fi
    # Run the script with the next argument
    cat conf/tuning/decode_ctcatt_LM_cali_backup | sed s/'AED_WEIGHT'/"$1"/g > conf/tuning/decode_ctcatt_LM_cali.yaml
    ./2_decode_with_LM_calibration.sh conf/tuning/train_conformer_ctcatt_wo_lsm_lm_mtl.yaml errlog001-3_decode_$1 &> errlog001-4_decode_AM${1}_LM${lm_w}_CTC${ctc_w}
    ./2_decode_wo_LM_calibration.sh conf/tuning/train_conformer_ctcatt_wo_lsm_lm_mtl.yaml errlog001-3_decode_wo-LM_$1 &> errlog001-4_decode_AM${1}_LM0.0_CTC${ctc_w}
    # Recursively call this function with the remaining arguments
    run_recursive "$((num_args - 1))" "${@:2}"
}

# Call the function to run the script recursively with all arguments
run_recursive $N "${am_w[@]}"
