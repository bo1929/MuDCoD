#!/bin/bash

curr_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)/"
mkdir -p "${curr_dir}log/"
mkdir -p "${curr_dir}../results"

classes_dcbm=(medium100)
time_horizon=(6)
num_subject=(8)
r_time=(0.0 0.2 0.5)
r_subject=(0.0 0.2 0.5)
cases_msd=(1 3)

qos="mid"
time="11:59:00"

for class_dcbm in ${classes_dcbm[@]}; do
  for th in ${time_horizon[@]}; do
    for ns in ${num_subject[@]}; do
      for rt in ${r_time[@]}; do
        for rs in ${r_subject[@]}; do
          for case_msd in ${cases_msd[@]}; do
            rs_dec=$(echo "${rs}"| cut -d'.' -f 2)
            rt_dec=$(echo "${rt}"| cut -d'.' -f 2)
            name="${class_dcbm}_case${case_msd}_th${th}_rt${rt_dec}_ns${ns}_rs${rs_dec}"
            echo "#!/bin/bash
# -= Resources =-
#
#SBATCH --job-name=${name}
#SBATCH --account=mdbf
#SBATCH --ntasks-per-node=4
#SBATCH --qos=${qos}_mdbf
#SBATCH --partition=${qos}_mdbf
#SBATCH --time=${time}
#SBATCH --output=${curr_dir}log/${name}.out
#SBATCH --mem=12G

# Set stack size to unlimited

ulimit -s unlimited
ulimit -l unlimited
ulimit -a
${curr_dir}run_simulation.sh \
  --class-dcbm ${class_dcbm} --case-msd ${case_msd} \
  --time-horizon ${th} --r-time ${rt} \
  --num-subjects ${ns} --r-subject ${rs}
"           > ${curr_dir}"tmp_slurm_submission.sh"
            sbatch ${curr_dir}"tmp_slurm_submission.sh"
          done
        done
      done
    done
  done
done
