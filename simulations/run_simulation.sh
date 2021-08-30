#!/bin/bash

curr_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)/"

POSITIONAL=()
while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    --class-dcbm)
    class_dcbm="$2"
    shift # past argument
    shift # past value
    ;;
    --case-msd)
    case_msd="$2"
    shift
    shift
    ;;
    --time-horizon)
    time_horizon="$2"
    shift
    shift
    ;;
    --num-subjects)
    num_subjects="$2"
    shift
    shift
    ;;
    --r-subject)
    r_subject="$2"
    shift
    shift
    ;;
    --r-time)
    r_time="$2"
    shift
    shift
    ;;
    --n-jobs)
    n_jobs="$2"
    shift
    shift
    ;;
    *)    # unknown option
    POSITIONAL+=("$1") # save it in an array for later
    shift # past argument
    ;;
esac
done

echo "DCBM-Class: ${class_dcbm}"
echo "Case: ${case_msd}"
echo "Time horizon: ${time_horizon}"
echo "r-time: ${r_time}"
echo "Number of subjects:${num_subjects}"
echo "r-subject: ${r_subject}"

alpha_values=(0.01 0.05 0.1 0.15 0.2 0.25 0.3)
beta_values=(0.01 0.05 0.1 0.15 0.2 0.25 0.3)

num_cd=0
num_cv=100

for ((i = 0 ; i < num_cv ; i++)); do
  for alpha in ${alpha_values[@]}; do

    python ${curr_dir}simulation.py \
      --class-dcbm=${class_dcbm} --case-msd=${case_msd} \
      --time-horizon=${time_horizon} --r-time=${r_time} \
      --num-subjects=${num_subjects} --r-subject=${r_subject} \
      cv-pisces --alpha=${alpha}

    for beta in ${beta_values[@]}; do
      python ${curr_dir}simulation.py \
        --class-dcbm=${class_dcbm} --case-msd=${case_msd} \
        --time-horizon=${time_horizon} --r-time=${r_time} \
        --num-subjects=${num_subjects} --r-subject=${r_subject} \
        cv-muspces --alpha=${alpha} --beta=${beta}
    done
  done
done

for ((i = 0 ; i < num_cd ; i++)); do
    python ${curr_dir}simulation.py \
      --class-dcbm=${class_dcbm} --case-msd=${case_msd} \
      --time-horizon=${time_horizon} --r-time=${r_time} \
      --num-subjects=${num_subjects} --r-subject=${r_subject} \
      community-detection --id-number=${i}
done
