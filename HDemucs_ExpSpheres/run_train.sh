#!/bin/bash
# sbatch -J run_train -o ./logs/log_%j.out -e ./logs/log_%j.err ./run_train.sh

# conda activate /mnt/share/pcabanas/CONDAENVS/demucstrain/

srun python -u lightning_main.py
# srun bash -c 'CUDA_VISIBLE_DEVICES=1 python -u lightning_main.py'