#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --export=ALL
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=08:00:00
#SBATCH --job-name=scohface_phys_eff

module load cuda/10.2
module load anaconda3/2022.05
source activate rppg-toolbox
python train.py --mode classification > log_classification_1007.txt
