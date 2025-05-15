#! /usr/bin/bash

#SBATCH -J "First Job"
#SBATCH --output=/local_ssd/practical_wise24/lung_cancer/adlm-lung-cancer/logs/slurm/%j.out
#SBATCH --error=/local_ssd/practical_wise24/lung_cancer/adlm-lung-cancer/logs/slurm/%j.err
#SBATCH --mem=40G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --partition=course

# load python module
module load python/anaconda3

conda activate /local_ssd/practical_wise24/lung_cancer/adlm-lung-cancer/venv

which python

python src/train.py
