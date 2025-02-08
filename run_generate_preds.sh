#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu
#SBATCH --time=5:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH --output=logs/gen-%J.out
#SBATCH --error=logs/gen-%J.err
#SBATCH --job-name="fcclip_gen"

module load CUDA
nvidia-smi

cd fcclip/modeling/pixel_decoder/ops
sh make.sh
cd ../../../../

python3 ./tests/generate_preds.py --input-dir ./datasets/ADEChallengeData2016/images/validation/ --output-dir ./tests/preds-clip-oracle-score-is-one-correct-class-validation/
