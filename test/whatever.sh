#!/bin/sh
#SBATCH -o gpu-job-%j.output
#SBATCH -p K20q
#SBATCH --gres=gpu:1
#SBATCH -n 1

module load cuda90/toolkit
module load cuda90/blas/9.0.176

export PATH=/home/wwu009/miniconda3/envs/pix2pix/bin:$PATH
cd /home/wwu009/Project/X-Net/test

set -ex
python whatever.py