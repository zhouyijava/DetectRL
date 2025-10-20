Detectors_path=Detectors/

Task1_path=Benchmark/Tasks/Task1
Task2_path=Benchmark/Tasks/Task2

# cd Benchmark\Benchmark

# Task1
python $Detectors_path/likelihood_evaluation.py --test_data_path $Task1_path/direct_prompt_test.json, $Task1_path/prompt_attacks_test.json,$Task1_path/paraphrase_attacks_test.json,$Task1_path/perturbation_attacks_test.json,$Task1_path/data_mixing_test.json
echo 'There is the result of likelihood_evaluation.py above'
python $Detectors_path/rank_evaluation.py --test_data_path $Task1_path/direct_prompt_test.json, $Task1_path/prompt_attacks_test.json,$Task1_path/paraphrase_attacks_test.json,$Task1_path/perturbation_attacks_test.json,$Task1_path/data_mixing_test.json
echo 'There is the result of rank_evaluation.py above'
python $Detectors_path/logRank_evaluation.py --test_data_path $Task1_path/direct_prompt_test.json, $Task1_path/prompt_attacks_test.json,$Task1_path/paraphrase_attacks_test.json,$Task1_path/perturbation_attacks_test.json,$Task1_path/data_mixing_test.json
echo 'There is the result of logRank_evaluation.py above'
python $Detectors_path/entropy_evaluation.py --test_data_path $Task1_path/direct_prompt_test.json, $Task1_path/prompt_attacks_test.json,$Task1_path/paraphrase_attacks_test.json,$Task1_path/perturbation_attacks_test.json,$Task1_path/data_mixing_test.json
echo 'There is the result of entropy_evaluation.py above'
python $Detectors_path/LRR_evaluation.py --test_data_path $Task1_path/direct_prompt_test.json, $Task1_path/prompt_attacks_test.json,$Task1_path/paraphrase_attacks_test.json,$Task1_path/perturbation_attacks_test.json,$Task1_path/data_mixing_test.json
echo 'There is the result of LRR_evaluation.py above'
python $Detectors_path/NPR_evaluation.py --test_data_path $Task1_path/direct_prompt_test.json, $Task1_path/prompt_attacks_test.json,$Task1_path/paraphrase_attacks_test.json,$Task1_path/perturbation_attacks_test.json,$Task1_path/data_mixing_test.json
echo 'There is the result of NPR_evaluation.py above'
python $Detectors_path/DetectGPT_evaluation.py --test_data_path $Task1_path/direct_prompt_test.json, $Task1_path/prompt_attacks_test.json,$Task1_path/paraphrase_attacks_test.json,$Task1_path/perturbation_attacks_test.json,$Task1_path/data_mixing_test.json
echo 'There is the result of DetectGPT_evaluation.py above'
python $Detectors_path/Fast_DetectGPT_evaluation.py --test_data_path $Task1_path/direct_prompt_test.json $Task1_path/prompt_attacks_test.json,$Task1_path/paraphrase_attacks_test.json,$Task1_path/perturbation_attacks_test.json,$Task1_path/data_mixing_test.json
echo 'There is the result of Fast_DetectGPT_evaluation.py above'
python $Detectors_path/binoculars_evaluation.py --test_data_path $Task1_path/direct_prompt_test.json $Task1_path/prompt_attacks_test.json,$Task1_path/paraphrase_attacks_test.json,$Task1_path/perturbation_attacks_test.json,$Task1_path/data_mixing_test.json
echo 'There is the result of binoculars_evaluation.py above'

python $Detectors_path/train_roberta.py --train_data_path $Task1_path/direct_prompt_train.json, --test_data_path $Task1_path/direct_prompt_test.json,
python $Detectors_path/train_roberta.py --train_data_path $Task1_path/prompt_attacks_train.json, --test_data_path $Task1_path/prompt_attacks_test.json,
python $Detectors_path/train_roberta.py --train_data_path $Task1_path/paraphrase_attacks_train.json --test_data_path $Task1_path/paraphrase_attacks_test.json,
python $Detectors_path/train_roberta.py --train_data_path $Task1_path/perturbation_attacks_train.json --test_data_path $Task1_path/perturbation_attacks_test.json,
python $Detectors_path/train_roberta.py --train_data_path $Task1_path/data_mixing_train.json, --test_data_path $Task1_path/data_mixing_test.json,


# Task2
python $Detectors_path/zero_shot_transfer_evaluation.py --method likelihood --test_data_path $Task2_path/direct_prompt_test.json, --transfer_data_path $Task2_path/prompt_attacks_test.json,$Task2_path/paraphrase_attacks_test.json,$Task2_path/perturbation_attacks_test.json,$Task2_path/data_mixing_test.json
echo 'The result of likelihood'
python $Detectors_path/zero_shot_transfer_evaluation.py --method rank --test_data_path $Task2_path/direct_prompt_test.json, --transfer_data_path $Task2_path/prompt_attacks_test.json,$Task2_path/paraphrase_attacks_test.json,$Task2_path/perturbation_attacks_test.json,$Task2_path/data_mixing_test.json
echo 'The result of rank'
python $Detectors_path/zero_shot_transfer_evaluation.py --method logRank --test_data_path $Task2_path/direct_prompt_test.json, --transfer_data_path $Task2_path/prompt_attacks_test.json,$Task2_path/paraphrase_attacks_test.json,$Task2_path/perturbation_attacks_test.json,$Task2_path/data_mixing_test.json
echo 'The result of logRank'
python $Detectors_path/zero_shot_transfer_evaluation.py --method entropy --test_data_path $Task2_path/direct_prompt_test.json, --transfer_data_path $Task2_path/prompt_attacks_test.json,$Task2_path/paraphrase_attacks_test.json,$Task2_path/perturbation_attacks_test.json,$Task2_path/data_mixing_test.json
echo 'The result of entropy'
python $Detectors_path/zero_shot_transfer_evaluation.py --method LRR --test_data_path $Task2_path/direct_prompt_test.json, --transfer_data_path $Task2_path/prompt_attacks_test.json,$Task2_path/paraphrase_attacks_test.json,$Task2_path/perturbation_attacks_test.json,$Task2_path/data_mixing_test.json
echo 'The result of LRR'
python $Detectors_path/zero_shot_transfer_evaluation.py --method NPR --test_data_path $Task2_path/direct_prompt_test.json, --transfer_data_path $Task2_path/prompt_attacks_test.json,$Task2_path/paraphrase_attacks_test.json,$Task2_path/perturbation_attacks_test.json,$Task2_path/data_mixing_test.json
echo 'The result of NPR'
python $Detectors_path/zero_shot_transfer_evaluation.py --method DetectGPT --test_data_path $Task2_path/direct_prompt_test.json, --transfer_data_path $Task2_path/prompt_attacks_test.json,$Task2_path/paraphrase_attacks_test.json,$Task2_path/perturbation_attacks_test.json,$Task2_path/data_mixing_test.json
echo 'The result of DetectGPT'
python $Detectors_path/zero_shot_transfer_evaluation.py --method Fast_DetectGPT --test_data_path $Task2_path/direct_prompt_test.json, --transfer_data_path $Task2_path/prompt_attacks_test.json,$Task2_path/paraphrase_attacks_test.json,$Task2_path/perturbation_attacks_test.json,$Task2_path/data_mixing_test.json
echo 'The result of Fast_DetectGPT'

python $Detectors_path/zero_shot_transfer_evaluation.py --method likelihood --test_data_path $Task2_path/prompt_attacks_test.json, --transfer_data_path $Task2_path/direct_prompt_test.json,$Task2_path/paraphrase_attacks_test.json,$Task2_path/perturbation_attacks_test.json,$Task2_path/data_mixing_test.json
echo 'The result of likelihood'
python $Detectors_path/zero_shot_transfer_evaluation.py --method rank --test_data_path $Task2_path/prompt_attacks_test.json, --transfer_data_path $Task2_path/direct_prompt_test.json,$Task2_path/paraphrase_attacks_test.json,$Task2_path/perturbation_attacks_test.json,$Task2_path/data_mixing_test.json
echo 'The result of rank'
python $Detectors_path/zero_shot_transfer_evaluation.py --method logRank --test_data_path $Task2_path/prompt_attacks_test.json, --transfer_data_path $Task2_path/direct_prompt_test.json,$Task2_path/paraphrase_attacks_test.json,$Task2_path/perturbation_attacks_test.json,$Task2_path/data_mixing_test.json
echo 'The result of logRank'
python $Detectors_path/zero_shot_transfer_evaluation.py --method entropy --test_data_path $Task2_path/prompt_attacks_test.json, --transfer_data_path $Task2_path/direct_prompt_test.json,$Task2_path/paraphrase_attacks_test.json,$Task2_path/perturbation_attacks_test.json,$Task2_path/data_mixing_test.json
echo 'The result of entropy'
python $Detectors_path/zero_shot_transfer_evaluation.py --method LRR --test_data_path $Task2_path/prompt_attacks_test.json, --transfer_data_path $Task2_path/direct_prompt_test.json,$Task2_path/paraphrase_attacks_test.json,$Task2_path/perturbation_attacks_test.json,$Task2_path/data_mixing_test.json
echo 'The result of LRR'
python $Detectors_path/zero_shot_transfer_evaluation.py --method NPR --test_data_path $Task2_path/prompt_attacks_test.json, --transfer_data_path $Task2_path/direct_prompt_test.json,$Task2_path/paraphrase_attacks_test.json,$Task2_path/perturbation_attacks_test.json,$Task2_path/data_mixing_test.json
echo 'The result of NPR'
python $Detectors_path/zero_shot_transfer_evaluation.py --method DetectGPT --test_data_path $Task2_path/prompt_attacks_test.json, --transfer_data_path $Task2_path/direct_prompt_test.json,$Task2_path/paraphrase_attacks_test.json,$Task2_path/perturbation_attacks_test.json,$Task2_path/data_mixing_test.json
echo 'The result of DetectGPT'
python $Detectors_path/zero_shot_transfer_evaluation.py --method Fast_DetectGPT --test_data_path $Task2_path/prompt_attacks_test.json, --transfer_data_path $Task2_path/direct_prompt_test.json,$Task2_path/paraphrase_attacks_test.json,$Task2_path/perturbation_attacks_test.json,$Task2_path/data_mixing_test.json
echo 'The result of Fast_DetectGPT'

python $Detectors_path/zero_shot_transfer_evaluation.py --method likelihood --test_data_path $Task2_path/paraphrase_attacks_test.json, --transfer_data_path $Task2_path/direct_prompt_test.json,$Task2_path/prompt_attacks_test.json,$Task2_path/perturbation_attacks_test.json,$Task2_path/data_mixing_test.json
echo 'The result of likelihood'
python $Detectors_path/zero_shot_transfer_evaluation.py --method rank --test_data_path $Task2_path/paraphrase_attacks_test.json, --transfer_data_path $Task2_path/direct_prompt_test.json,$Task2_path/prompt_attacks_test.json,$Task2_path/perturbation_attacks_test.json,$Task2_path/data_mixing_test.json
echo 'The result of rank'
python $Detectors_path/zero_shot_transfer_evaluation.py --method logRank --test_data_path $Task2_path/paraphrase_attacks_test.json, --transfer_data_path $Task2_path/direct_prompt_test.json,$Task2_path/prompt_attacks_test.json,$Task2_path/perturbation_attacks_test.json,$Task2_path/data_mixing_test.json
echo 'The result of logRank'
python $Detectors_path/zero_shot_transfer_evaluation.py --method entropy --test_data_path $Task2_path/paraphrase_attacks_test.json, --transfer_data_path $Task2_path/direct_prompt_test.json,$Task2_path/prompt_attacks_test.json,$Task2_path/perturbation_attacks_test.json,$Task2_path/data_mixing_test.json
echo 'The result of entropy'
python $Detectors_path/zero_shot_transfer_evaluation.py --method LRR --test_data_path $Task2_path/paraphrase_attacks_test.json, --transfer_data_path $Task2_path/direct_prompt_test.json,$Task2_path/prompt_attacks_test.json,$Task2_path/perturbation_attacks_test.json,$Task2_path/data_mixing_test.json
echo 'The result of LRR'
python $Detectors_path/zero_shot_transfer_evaluation.py --method NPR --test_data_path $Task2_path/paraphrase_attacks_test.json, --transfer_data_path $Task2_path/direct_prompt_test.json,$Task2_path/prompt_attacks_test.json,$Task2_path/perturbation_attacks_test.json,$Task2_path/data_mixing_test.json
echo 'The result of NPR'
python $Detectors_path/zero_shot_transfer_evaluation.py --method DetectGPT --test_data_path $Task2_path/paraphrase_attacks_test.json, --transfer_data_path $Task2_path/direct_prompt_test.json,$Task2_path/prompt_attacks_test.json,$Task2_path/perturbation_attacks_test.json,$Task2_path/data_mixing_test.json
echo 'The result of DetectGPT'
python $Detectors_path/zero_shot_transfer_evaluation.py --method Fast_DetectGPT --test_data_path $Task2_path/paraphrase_attacks_test.json, --transfer_data_path $Task2_path/direct_prompt_test.json,$Task2_path/prompt_attacks_test.json,$Task2_path/perturbation_attacks_test.json,$Task2_path/data_mixing_test.json
echo 'The result of Fast_DetectGPT'


python $Detectors_path/zero_shot_transfer_evaluation.py --method likelihood --test_data_path $Task2_path/perturbation_attacks_test.json, --transfer_data_path $Task2_path/direct_prompt_test.json,$Task2_path/prompt_attacks_test.json,$Task2_path/paraphrase_attacks_test.json,$Task2_path/data_mixing_test.json
echo 'The result of likelihood'
python $Detectors_path/zero_shot_transfer_evaluation.py --method rank --test_data_path $Task2_path/perturbation_attacks_test.json, --transfer_data_path $Task2_path/direct_prompt_test.json,$Task2_path/prompt_attacks_test.json,$Task2_path/paraphrase_attacks_test.json,$Task2_path/data_mixing_test.json
echo 'The result of rank'
python $Detectors_path/zero_shot_transfer_evaluation.py --method logRank --test_data_path $Task2_path/perturbation_attacks_test.json, --transfer_data_path $Task2_path/direct_prompt_test.json,$Task2_path/prompt_attacks_test.json,$Task2_path/paraphrase_attacks_test.json,$Task2_path/data_mixing_test.json
echo 'The result of logRank'
python $Detectors_path/zero_shot_transfer_evaluation.py --method entropy --test_data_path $Task2_path/perturbation_attacks_test.json, --transfer_data_path $Task2_path/direct_prompt_test.json,$Task2_path/prompt_attacks_test.json,$Task2_path/paraphrase_attacks_test.json,$Task2_path/data_mixing_test.json
echo 'The result of entropy'
python $Detectors_path/zero_shot_transfer_evaluation.py --method LRR --test_data_path $Task2_path/perturbation_attacks_test.json, --transfer_data_path $Task2_path/direct_prompt_test.json,$Task2_path/prompt_attacks_test.json,$Task2_path/paraphrase_attacks_test.json,$Task2_path/data_mixing_test.json
echo 'The result of LRR'
python $Detectors_path/zero_shot_transfer_evaluation.py --method NPR --test_data_path $Task2_path/perturbation_attacks_test.json, --transfer_data_path $Task2_path/direct_prompt_test.json,$Task2_path/prompt_attacks_test.json,$Task2_path/paraphrase_attacks_test.json,$Task2_path/data_mixing_test.json
echo 'The result of NPR'
python $Detectors_path/zero_shot_transfer_evaluation.py --method DetectGPT --test_data_path $Task2_path/perturbation_attacks_test.json, --transfer_data_path $Task2_path/direct_prompt_test.json,$Task2_path/prompt_attacks_test.json,$Task2_path/paraphrase_attacks_test.json,$Task2_path/data_mixing_test.json
echo 'The result of DetectGPT'
python $Detectors_path/zero_shot_transfer_evaluation.py --method Fast_DetectGPT --test_data_path $Task2_path/perturbation_attacks_test.json, --transfer_data_path $Task2_path/direct_prompt_test.json,$Task2_path/prompt_attacks_test.json,$Task2_path/paraphrase_attacks_test.json,$Task2_path/data_mixing_test.json
echo 'The result of Fast_DetectGPT'


python $Detectors_path/zero_shot_transfer_evaluation.py --method likelihood --test_data_path $Task2_path/data_mixing_test.json, --transfer_data_path $Task2_path/direct_prompt_test.json,$Task2_path/prompt_attacks_test.json,$Task2_path/paraphrase_attacks_test.json,$Task2_path/perturbation_attacks_test.json
echo 'The result of likelihood'
python $Detectors_path/zero_shot_transfer_evaluation.py --method rank --test_data_path $Task2_path/data_mixing_test.json, --transfer_data_path $Task2_path/direct_prompt_test.json,$Task2_path/prompt_attacks_test.json,$Task2_path/paraphrase_attacks_test.json,$Task2_path/perturbation_attacks_test.json
echo 'The result of rank'
python $Detectors_path/zero_shot_transfer_evaluation.py --method logRank --test_data_path $Task2_path/data_mixing_test.json, --transfer_data_path $Task2_path/direct_prompt_test.json,$Task2_path/prompt_attacks_test.json,$Task2_path/paraphrase_attacks_test.json,$Task2_path/perturbation_attacks_test.json
echo 'The result of logRank'
python $Detectors_path/zero_shot_transfer_evaluation.py --method entropy --test_data_path $Task2_path/data_mixing_test.json, --transfer_data_path $Task2_path/direct_prompt_test.json,$Task2_path/prompt_attacks_test.json,$Task2_path/paraphrase_attacks_test.json,$Task2_path/perturbation_attacks_test.json
echo 'The result of entropy'
python $Detectors_path/zero_shot_transfer_evaluation.py --method LRR --test_data_path $Task2_path/data_mixing_test.json, --transfer_data_path $Task2_path/direct_prompt_test.json,$Task2_path/prompt_attacks_test.json,$Task2_path/paraphrase_attacks_test.json,$Task2_path/perturbation_attacks_test.json
echo 'The result of LRR'
python $Detectors_path/zero_shot_transfer_evaluation.py --method NPR --test_data_path $Task2_path/data_mixing_test.json, --transfer_data_path $Task2_path/direct_prompt_test.json,$Task2_path/prompt_attacks_test.json,$Task2_path/paraphrase_attacks_test.json,$Task2_path/perturbation_attacks_test.json
echo 'The result of NPR'
python $Detectors_path/zero_shot_transfer_evaluation.py --method DetectGPT --test_data_path $Task2_path/data_mixing_test.json, --transfer_data_path $Task2_path/direct_prompt_test.json,$Task2_path/prompt_attacks_test.json,$Task2_path/paraphrase_attacks_test.json,$Task2_path/perturbation_attacks_test.json
echo 'The result of DetectGPT'
python $Detectors_path/zero_shot_transfer_evaluation.py --method Fast_DetectGPT --test_data_path $Task2_path/data_mixing_test.json, --transfer_data_path $Task2_path/direct_prompt_test.json,$Task2_path/prompt_attacks_test.json,$Task2_path/paraphrase_attacks_test.json,$Task2_path/perturbation_attacks_test.json
echo 'The result of Fast_DetectGPT'


python $Detectors_path/train_roberta.py --train_data_path $Task2_path/direct_prompt_train.json, --test_data_path $Task2_path/prompt_attacks_test.json,$Task2_path/paraphrase_attacks_test.json,$Task2_path/perturbation_attacks_test.json,$Task2_path/data_mixing_test.json,
python $Detectors_path/train_roberta.py --train_data_path $Task2_path/prompt_attacks_train.json, --test_data_path $Task2_path/direct_prompt_test.json,$Task2_path/paraphrase_attacks_test.json,$Task2_path/perturbation_attacks_test.json,$Task2_path/ata_mixing_test.json,
python $Detectors_path/train_roberta.py --train_data_path $Task2_path/paraphrase_attacks_train.json --test_data_path $Task2_path/direct_prompt_test.json,$Task2_path/prompt_attacks_test.json,$Task2_path/perturbation_attacks_test.json,$Task2_path/data_mixing_test.json,
python $Detectors_path/train_roberta.py --train_data_path $Task2_path/perturbation_attacks_train.json --test_data_path $Task2_path/direct_prompt_test.json,$Task2_path/prompt_attacks_test.json,$Task2_path/paraphrase_attacks_test.json,$Task2_path/data_mixing_test.json,
python $Detectors_path/train_roberta.py --train_data_path $Task2_path/data_mixing_train.json, --test_data_path $Task2_path/direct_prompt_test.json,$Task2_path/prompt_attacks_test.json,$Task2_path/paraphrase_attacks_test.json,$Task2_path/perturbation_attacks_test.json,