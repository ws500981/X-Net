#!/bin/sh
#SBATCH -o gpu-job-%j.output
#SBATCH -p PV1003q
#SBATCH --gres=gpu:1
#SBATCH -n 1

module load cuda90/toolkit
module load cuda90/blas/9.0.176

export PATH=/home/wwu009/miniconda3/envs/pix2pix/bin:$PATH
cd /home/wwu009/Project/X-Net

set -ex
python trytoaddcallback_tosavevalresult.py --exp_nm save_loadedandoutput_tryabit
