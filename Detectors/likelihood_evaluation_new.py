import logging
import random
import numpy as np
import torch
import tqdm
import argparse
import json
from transformers import AutoTokenizer, AutoModelForCausalLM

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

def get_ll(text, args, tokenizer, model):
    with torch.no_grad():
        tokenized = tokenizer(text, return_tensors="pt").to(args.DEVICE)
        labels = tokenized['input_ids']
        if labels.nelement() == 0:
            logging.error(f"Empty input: {text}")
            return None
        else:
            return -model(**tokenized, labels=labels).loss.item()

def truncate_text_to_sentences(text, min_word_count=100):
    word_count = 0
    end_of_last_sentence = 0

    words = text.split()

    for i, word in enumerate(words):
        if word[-1] in '.!?':
            if word_count >= min_word_count:
                return ' '.join(words[:i + 1])
            end_of_last_sentence = i
        word_count += 1

    if end_of_last_sentence > 0:
        return ' '.join(words[:end_of_last_sentence + 1])
    else:
        return ' '.join(words)

def experiment(args):
    # load model
    logging.info(f"Loading base model of type {args.base_model}...")
    base_tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    base_model = AutoModelForCausalLM.from_pretrained(args.base_model)
    base_model.eval()
    base_model.cuda()

    filenames = args.test_data_path.split(",")
    for filename in filenames:
        logging.info(f"Processing data in {filename}")
        test_data = json.load(open(filename, "r"))

        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        results = []
        for item in tqdm.tqdm(test_data):
            text = item.get("text")
            if not text:  # 如果 text 为空或缺失
                text = item["comments"]
            global_id = item.get("global_id", None)
            # Truncate text if needed
            truncated_text = truncate_text_to_sentences(text, min_word_count=100)
            # Compute log-likelihood
            log_likelihood = get_ll(truncated_text, args, base_tokenizer, base_model)
            
            if log_likelihood is not None and np.isfinite(log_likelihood):
                results.append({
                    "global_id": global_id, 
                    "text": text,
                    "truncated_text": truncated_text,
                    "log_likelihood": log_likelihood
                })
            else:
                logging.warning(f"Skipping invalid log-likelihood for text: {text[:50]}...")

        # Save results
        output_filename = filename.split(".json")[0] + "_likelihood_results.json"
        with open(output_filename, "w") as f:
            json.dump(results, f, indent=4)
        logging.info(f"Results saved to {output_filename}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_data_path', type=str, required=True,
                        help="Path to the test data. Could be several files separated by ','.")
    parser.add_argument('--base_model', default="EleutherAI/gpt-neo-2.7B", type=str, required=False)
    parser.add_argument('--DEVICE', default="cuda", type=str, required=False)
    parser.add_argument('--seed', default=2023, type=int, required=False)
    args = parser.parse_args()

    experiment(args)