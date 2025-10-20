import functools
import logging
import math
import random
import numpy as np
import torch
import tqdm
import argparse
import json
from DetectGPT import perturb_texts
from metrics import get_roc_metrics
from rank import get_rank, get_ranks
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def experiment(args):

    logging.info(f"Loading base model of type {args.base_model}...")
    base_tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    base_model = AutoModelForCausalLM.from_pretrained(args.base_model)

    mask_tokenizer = AutoTokenizer.from_pretrained(args.mask_model)
    mask_model = AutoModelForSeq2SeqLM.from_pretrained(args.mask_model)

    model_config = {
        "base_tokenizer": base_tokenizer,
        "base_model": base_model,
        "mask_tokenizer": mask_tokenizer,
        "mask_model": mask_model,
    }

    filenames = args.test_data_path.split(",")
    for filename in filenames:
        logging.info(f"Test in {filename}")
        test_data = json.load(open(filename, "r"))

        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        perturb_fn = functools.partial(perturb_texts, args=args, model_config=model_config)

        mask_model.eval()
        mask_model.cuda()
        for item in tqdm.tqdm(test_data):
            try:
                perturbed_text = perturb_fn([item["text"] for _ in range(max(args.n_perturbation_list))])
                assert len(perturbed_text) == max(
                    args.n_perturbation_list), f"Expected {max(args.n_perturbation_list)} perturbed samples, got {len(perturbed_text)}"
                item["perturbed_text"] = perturbed_text
            except:
                logging.error(f"Failed to perturb text {item['text']}")
                item["perturbed_text"] = [None for _ in range(max(args.n_perturbation_list))]
        mask_model.to("cpu")

        base_model.eval()
        base_model.cuda()
        for item in tqdm.tqdm(test_data):
            text = item["text"]
            try:
                item["text_logrank"] = get_rank(text, args, base_tokenizer, base_model)
                perturbed_text_rank = get_ranks(item["perturbed_text"], args, base_tokenizer, base_model, log=True)

                item["perturbed_text_logrank"] = perturbed_text_rank
                for n_perturbation in args.n_perturbation_list:
                    item[f"perturbed_text_logrank_{n_perturbation}"] = np.mean(
                        [i for i in perturbed_text_rank[:n_perturbation] if i != None])
            except:
                logging.error(f"Failed to rank text {text}")
                item["text_logrank"] = None
                item["perturbed_text_logrank"] = [None for _ in range(max(args.n_perturbation_list))]
                for n_perturbation in args.n_perturbation_list:
                    item[f"perturbed_text_logrank_{n_perturbation}"] = None
        base_model.to("cpu")

        results = {}
        for n_perturbation in args.n_perturbation_list:
            predictions = {'human': [], 'llm': []}
            for item in test_data:
                if item["text_logrank"] is None:
                    pass
                else:
                    label = item["label"]

                    if label == "human":
                        predictions['human'].append(
                            item[f'perturbed_text_logrank_{n_perturbation}'] / item["text_logrank"])
                    elif label == "llm":
                        predictions['llm'].append(
                            item[f'perturbed_text_logrank_{n_perturbation}'] / item["text_logrank"])
                    else:
                        raise ValueError(f"Unknown label {label}")

            predictions['human'] = [i for i in predictions['human'] if not math.isnan(i)]
            predictions['llm'] = [i for i in predictions['llm'] if not math.isnan(i)]

            predictions['human'] = [i for i in predictions['human'] if np.isfinite(i)]
            predictions['llm'] = [i for i in predictions['llm'] if np.isfinite(i)]

            roc_auc, optimal_threshold, conf_matrix, precision, recall, f1, accuracy, tpr_at_fpr_0_01 = get_roc_metrics(
                predictions['human'], predictions['llm'])

            result = {
                "roc_auc": roc_auc,
                "optimal_threshold": optimal_threshold,
                "conf_matrix": conf_matrix,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "accuracy": accuracy
            }

            results[f"{n_perturbation}_perturbation"] = result
        print('npr')
        print(filenames)
        print(f"{results}")
        with open(filename.split(".json")[0] + "_NPR_data.json", "w") as f:
            json.dump(test_data, f, indent=4)

        with open(filename.split(".json")[0] + "_NPR_result.json", "w") as f:
            json.dump(results, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_data_path', type=str, required=True,
                        help="Path to the test data. could be several files with ','. "
                             "Note that the data should have been perturbed.")
    parser.add_argument('--base_model', default="EleutherAI/gpt-neo-2.7B", type=str)
    parser.add_argument('--mask_model', default="t5-small", type=str)
    parser.add_argument('--output_file', default="output_file", type=str)
    parser.add_argument('--n_perturbation_list', default=[1, 10, 20, 50, 100], type=list)
    parser.add_argument('--span_length', type=int, default=2)
    parser.add_argument('--mask_top_p', type=float, default=1.0)
    parser.add_argument('--buffer_size', type=int, default=1)
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--pct_words_masked', type=float,
                        default=0.3)  # pct masked is actually pct_words_masked * (span_length / (span_length + 2 * buffer_size))
    parser.add_argument('--DEVICE', default="cuda", type=str, required=False)
    parser.add_argument('--seed', default=2023, type=int, required=False)
    args = parser.parse_args()

    experiment(args)
