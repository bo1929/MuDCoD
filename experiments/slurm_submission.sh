#!/bin/bash
# -= Resources =-
#
#SBATCH --job-name=DA-muspces
#SBATCH --account=mdbf
#SBATCH --ntasks-per-node=2
#SBATCH --qos=mid_mdbf
#SBATCH --partition=mid_mdbf
#SBATCH --time=23:59:00
#SBATCH --output=/cta/users/aosman/mudcod/experiments/DA-muspces.output
#SBATCH --mem=32G

# Set stack size to unlimited
python perform_community_detection.py  --percentile 95 --cell-type DA --verbose True --method-name muspces
