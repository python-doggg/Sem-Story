#!/usr/bin/bash
#SBATCH -o test2.%j.out
#SBATCH --partition=ai_training
#SBATCH --qos=medium
#SBATCH -J test2
#SBATCH --ntasks-per-node=8
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --nodelist=dx-ai-node15

eval "$(conda shell.bash hook)"
conda activate storygen
#CUDA_VISIBLE_DEVICES=0 accelerate launch train_StorySalon_stage2.py
#CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --multi_gpu train_stage2_change1.py #SBATCH --nodelist=dx-ai-node14
CUDA_VISIBLE_DEVICES=0 python test_all.py