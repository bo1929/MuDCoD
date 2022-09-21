#!/bin/bash
# -= Resources =-
#
#SBATCH --job-name=medium500_case1_th2_rt2_ns8_rs2
#SBATCH --account=mdbf
#SBATCH --ntasks-per-node=1
#SBATCH --qos=mid_mdbf
#SBATCH --partition=mid_mdbf
#SBATCH --time=23:59:00
#SBATCH --output=/cta/users/aosman/mudcod/simulations/log/medium500_case1_th2_rt2_ns8_rs2.out
#SBATCH --mem=12G

# Set stack size to unlimited

ulimit -s unlimited
ulimit -l unlimited
ulimit -a
/cta/users/aosman/mudcod/simulations/run_simulation.sh   --class-dcbm medium500 --case-msd 1   --time-horizon 2 --r-time 0.2   --num-subjects 8 --r-subject 0.2

