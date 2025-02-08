#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=5:00:00
#SBATCH --mem=20G
#SBATCH --output=logs/face_anon-%J.out
#SBATCH --error=logs/face_anon-%J.err
#SBATCH --job-name="face_anon"

#module load CUDA
#nvidia-smi

#cd fcclip/modeling/pixel_decoder/ops
#sh make.sh
#cd ../../../../

python3 ./fcclip/evaluation/panoptic_evaluation.py \
                --gt_json_file ./datasets/ADEChallengeData2016/ade20k_panoptic_val.json \
                --pred_json_file ./tests/preds-clip-oracle-score-is-one-correct-class-validation/annotations.json \
                --gt_folder ./datasets/ADEChallengeData2016/ade20k_panoptic_val \
                --pred_folder ./tests/preds-clip-oracle-score-is-one-correct-class-validation
