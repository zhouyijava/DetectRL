import logging
import numpy as np
import tqdm
import argparse
import json
from metrics import get_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def experiment(args):

    final_result = {}
    for method in [args.methods]:
        logging.info(f"Testing method {method}")

        score_key = {
            "likelihood": "text_ll",
            "entropy": "entropy",
            "rank": "text_rank",
            "logRank": "text_logrank",
            "LRR": "text_LRR",
            "DetectGPT": "detectgpt_score",
            "Fast_DetectGPT": "text_crit",
            "bino": "bino_score",
        }

        result_path = args.test_data_path.split(".json")[0] + f"_{method}_result.json"
        result_path = result_path.replace('Task2', 'Task1')
        logging.info(f"Loading result from {result_path}")
        with open(result_path, "r") as f:
            result_data = json.load(f)
            if method in ["NPR", "DetectGPT"]:
                optimal_threshold = result_data["100_perturbation"]["optimal_threshold"]
            else:
                optimal_threshold = result_data["optimal_threshold"]

        filenames = args.transfer_data_path.split(",")
        method_result = {}
        for filename in filenames:
            logging.info(f"Test in {filename}")
            data_path = filename.split(".json")[0] + f"_{method}_data.json"
            # print('+++++++++++++++++/n',data_path)
            # if 'Benchmark' not in data_path:
            #     data_path = 'Benchmark/Tasks/Task1/'+data_path
            # else:
            #     data_path = data_path.replace('Task2', 'Task1')
            # print('+++++++++++++++++/n',data_path)
            logging.info(f"Test in {data_path}")
            test_data = json.load(open(data_path, "r"))

            predictions = {'human': [], 'llm': []}
            for item in tqdm.tqdm(test_data):
                if method == "NPR":
                    label = item["label"]

                    if label == "human":
                        try:
                            predictions['human'].append(item[f'perturbed_text_logrank_100'] / item["text_logrank"])
                        except:
                            pass
                            print("error")
                    elif label == "llm":
                        try:
                            predictions['llm'].append(item[f'perturbed_text_logrank_100'] / item["text_logrank"])
                        except:
                            pass
                            print("error")
                    else:
                        raise ValueError(f"Unknown label {label}")
                else:
                    label = item["label"]
                    # result
                    if label == "human":
                        predictions['human'].append(item[score_key[method]])
                    elif label == "llm":
                        predictions['llm'].append(item[score_key[method]])
                    else:
                        raise ValueError(f"Unknown label {label}")

            predictions['human'] = [i for i in predictions['human'] if np.isfinite(i)]
            predictions['llm'] = [i for i in predictions['llm'] if np.isfinite(i)]

            optimal_threshold, conf_matrix, precision, recall, f1, accuracy = get_metrics(predictions['human'], predictions['llm'], optimal_threshold)

            result = {
                "optimal_threshold": optimal_threshold,
                "conf_matrix": conf_matrix,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "accuracy": accuracy
            }

            parts = filename.split('/')
            filename = parts[-1]  # 获取文件名 'cross_domains_arxiv_train.json'
            file_base = filename.split('_test')[0]  # 从文件名中分割出基础部分 'cross_domains_arxiv'

            method_result[file_base] = result
            print('zero-shot-transfer')
            print(filenames)
            logging.info(f"{result}")

        final_result[method] = method_result

    with open(args.test_data_path.split(".json")[0] + f"_transfer_result.json", "w") as f:
        json.dump(final_result, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_data_path', type=str, required=True,
                        help="Path to the test data. could be several files with ','. "
                             "Note that the data should have been perturbed.")
    parser.add_argument('--transfer_data_path', type=str, required=True,
                        help="Path to the test data. could be several files with ','. "
                             "Note that the data should have been perturbed.")
    parser.add_argument('--methods', default=["likelihood", "entropy", "rank", "logRank", "LRR", "NPR", "DetectGPT", "Fast_DetectGPT", 'bino'], required=False)
    parser.add_argument('--DEVICE', default="cuda", type=str, required=False)
    parser.add_argument('--seed', default=2023, type=int, required=False)
    args = parser.parse_args()

    experiment(args)
