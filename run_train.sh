#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu
#SBATCH --time=20:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH --output=logs/train-%J.out
#SBATCH --error=logs/train-%J.err
#SBATCH --job-name="00003_fcclip_train"
#SBATCH --constraint=h100

#module load CUDA
#nvidia-smi

#cd fcclip/modeling/pixel_decoder/ops
#sh make.sh
#cd ../../../../

python3 ./train_net.py  --num-gpus 1 --config-file configs/coco/panoptic-segmentation/fcclip/my_fcclip_convnext_large_eval_ade20k_r50.yaml \
    SOLVER.IMS_PER_BATCH 12 SOLVER.BASE_LR 0.0003 OUTPUT_DIR LR_00003 MODEL.FC_CLIP.DIST_WEIGHT 1
