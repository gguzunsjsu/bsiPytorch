#!/bin/bash
#SBATCH --mail-user=akankshapankaj.joshi@sjsu.edu
#SBATCH --mail-user=/dev/null
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --job-name=bertTrain
#SBATCH --output=bertTrain_%j.out
#SBATCH --error=bertTrain_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=48:00:00
##SBATCH --mem-per-cpu=2000
##SBATCH --gres=gpu:p100:1
#SBATCH --partition=gpu

# on coe-hpc1 cluster load
module load python3/3.8.8
#
# on coe-hpc2 cluster load:
# module load python-3.10.8-gcc-11.2.0-c5b5yhp slurm

export http_proxy=http://172.16.1.2:3128; export https_proxy=http://172.16.1.2:3128

cd /home/016690830/RA
source /home/016690830/venv/bin/activate

#nvidia-smi

# python3 train.py
python3 train_extract_vector.py