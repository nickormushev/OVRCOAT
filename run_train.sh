#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH --output=logs/train-%J.out
#SBATCH --error=logs/train-%J.err
#SBATCH --job-name="fcclip_train"
#SBATCH --constraint=h100

module load CUDA
nvidia-smi

cd fcclip/modeling/pixel_decoder/ops
sh make.sh
cd ../../../../

python3 ./train_net.py  --num-gpus 1 --config-file configs/coco/panoptic-segmentation/fcclip/fcclip_convnext_large_eval_ade20k.yaml \
    SOLVER.IMS_PER_BATCH 4 SOLVER.BASE_LR 0.000025

 