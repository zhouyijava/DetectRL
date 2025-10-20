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

# pip install --upgrade pip  
# pip install datasets==1.18.4 pyarrow==15.0.0 'fsspec<2023.12.0' nltk kagglehub torch rank_bm25 selenium openai google
# pip install --upgrade google-api-python-client 
# pip install -q -U google-generativeai 
# pip install anthropic textattack
# pip install --upgrade transformers
#sh data_generation.sh

cd /home/y/yj171/DetectRL_zhouyi

case $1 in
	0)
 		echo 'running, Task' $(($1+1)) 'and' $(($1+2))
		# Task1 and Task2 evaluation
		sh domains_evaluation.sh
		sh llms_evaluation.sh
		sh attacks_evaluation.sh
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
