#!/bin/bash  
#SBATCH --account=su004  
#SBATCH --nodes=1  
#SBATCH --ntasks-per-node=1  
#SBATCH --cpus-per-task=42
#SBATCH --mem-per-cpu=3850
#SBATCH --time=24:00:00  

#SBATCH --account=su004-gpulowpri
#SBATCH --partition=gpulowpri
#SBATCH --gres=gpu:ampere_a100:1 

#SBATCH --output=./logs/slurm-%j.out
#SBATCH --error=./logs/slurm-%j.err

#Load modules  
module purge 
module load GCCcore/13.3.0 Python/3.12.3
module load cuDNN/8.7.0.84-CUDA-11.8.0 

#Activate environment 
source ../venv/bin/activate
pip install --upgrade pip

case $1 in
	0)
 		echo 'running, Task' $(($1+1)) 'and' $(($1+2))
		# Task1 and Task2 evaluation
		sh domains_evaluation.sh
		# sh llms_evaluation.sh
		# sh attacks_evaluation.sh
		;;
	1)
        echo 'running, Task' $(($1+2))
		# Task3 evaluation
		sh varying_length_evaluation.sh
		;;
	2)
        echo 'running, Task' $(($1+2))
		# Task4 evaluation
		sh human_writing_evaluation.sh
		;;
	*)
		echo 'there was an problems!!!\n', $0
		;;
esac
