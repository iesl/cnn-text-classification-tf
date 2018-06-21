#!/bin/bash
#
#SBATCH --mem=50000
#SBATCH --job-name=cnn-tf
#SBATCH --partition=m40-long
#SBATCH --output=model-%A.out
#SBATCH --error=model-%A.err
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ruppaal@cs.umass.edu

# Log what we're running and where.
echo $SLURM_JOBID - `hostname` >> ~/slurm-jobs.txt

export PATH=~/anaconda/bin:"$PATH"
module purge
module load python/2.7.12
module load tensorflow/1.0.1
#module load cuda80/blas/8.0.44
#module load cuda80/fft/8.0.44
#module load cuda80/nsight/8.0.44
#module load cuda80/profiler/8.0.44
#module load cuda80/toolkit/8.0.44
module load cudnn/7.0-cuda_8.0

cd /home/ruppaal/Work/IESL/cnn-text-classification-tf/

./train.py --embedding_dim 300 --word2vec "data/GoogleNews-vectors-negative300.bin"
