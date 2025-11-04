#!/bin/bash
#SBATCH --nodes=1  
#SBATCH --ntasks-per-node=1  
#SBATCH --cpus-per-task=42
#SBATCH --mem-per-cpu=3850
#SBATCH --time=24:00:00

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

echo '$0 is being processed by three GPUs'

python Detectors/Fast_DetectGPT_evaluation_new.py --test_data_path /home/y/yz741/DRL/DetectRL/repository/azure-sdk-for-python/2024-04/text_content_small_size.json > triviality_python2404t.log 2>&1
python Detectors/likelihood_evaluation_new.py --test_data_path /home/y/yz741/DRL/DetectRL/repository/azure-sdk-for-python/2024-04/text_content_small_size.json > triviality_python2404t.log 2>&1
python Detectors/Fast_DetectGPT_evaluation_new.py --test_data_path /home/y/yz741/DRL/DetectRL/repository/azure-sdk-for-python/2024-07/comments_content_small_size.json > triviality_python2407t.log 2>&1
python Detectors/DetectGPT_evaluation_new_new.py --test_data_path /home/y/yz741/DRL/DetectRL/repository/azure-sdk-for-python/2024-07/comments_content_small_size.json > triviality_python2407t.log 2>&1

python Detectors/DetectGPT_evaluation_new_new.py --test_data_path /home/y/yz741/DRL/DetectRL/repository/azure-sdk-for-python/2025-04/comments_content_small_size.json > triviality_python2504c.log 2>&1
python Detectors/Fast_DetectGPT_evaluation_new.py --test_data_path /home/y/yz741/DRL/DetectRL/repository/azure-sdk-for-python/2025-04/comments_content_small_size.json > triviality_python2504c.log 2>&1
python Detectors/likelihood_evaluation_new.py --test_data_path /home/y/yz741/DRL/DetectRL/repository/azure-sdk-for-python/2025-04/comments_content_small_size.json > triviality_python2504c.log 2>&1
python Detectors/rank_evaluation_new.py --test_data_path /home/y/yz741/DRL/DetectRL/repository/azure-sdk-for-python/2025-04/comments_content_small_size.json > triviality_python2504c.log 2>&1


echo "yz notice that all jobs finished!"