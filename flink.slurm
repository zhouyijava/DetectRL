#!/bin/bash
#SBATCH --nodes=1  
#SBATCH --ntasks-per-node=1  
#SBATCH --cpus-per-task=42
#SBATCH --mem-per-cpu=3850
#SBATCH --time=00:30:00

#SBATCH --account=su004-gpulowpri
#SBATCH --partition=gpulowpri
#SBATCH --gres=gpu:ampere_a100:1

#SBATCH --output=logs/zysl-%j.out
#SBATCH --error=logs/zysl-%j.err

#Load modules
module purge 
module load GCCcore/13.3.0 Python/3.12.3
module load cuDNN/8.7.0.84-CUDA-11.8.0

#Activate environment 
source ../venv/bin/activate

echo '$1 is being processed by three GPUs'
srun -n 1 -G 1 -c 40 --cpus-per-gpu=40 python stream_parallel_run.py --data_path $1
#srun -n 1 -G 3 -c 126 --cpus-per-gpu=42 python stream_parallel_run.py --data_path $1
echo '$1 is being processed sucessfully by three GPUs'

