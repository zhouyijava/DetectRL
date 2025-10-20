#!/bin/bash  

#SBATCH --account=su004  

#SBATCH --nodes=1  

#SBATCH --ntasks-per-node=1  

#SBATCH --cpus-per-task=128

#SBATCH --mem-per-cpu=3850

#SBATCH --time=48:00:00  

#SBATCH --account=su004 

#SBATCH --partition=gpu  

#SBATCH --gres=gpu:ampere_a100:1 

#SBATCH --output=/home/y/yj171/DetectRL_zhouyi/slurm_logs/slurm-%j.out
#SBATCH --error=/home/y/yj171/DetectRL_zhouyi/slurm_logs/slurm-%j.err

#Load modules  



#Activate environment 
source /home/y/yj171/miniconda3/bin/activate detectrl

cd /home/y/yj171/DetectRL_zhouyi

bash /home/y/yj171/DetectRL_zhouyi/detectrl.sh