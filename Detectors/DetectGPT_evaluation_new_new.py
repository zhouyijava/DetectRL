import functools
import logging
import math
import random
import numpy as np
import torch
import tqdm
import argparse
import json
import ast
import re
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM


from DetectGPT import perturb_texts
from loss import get_ll, get_lls

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

def preprocess_text(text):
    """Remove Markdown syntax, URLs, and special characters."""
    text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)  # Remove Markdown comments
    text = re.sub(r'#+\s*', '', text)  # Remove headers
    text = re.sub(r'\[.*?\]\(.*?\)', '', text)  # Remove links
    text = re.sub(r'[^\w\s.,!?]', '', text)  # Remove special characters
    return text.strip()

def truncate_text(text, tokenizer, max_length=512):
    """Truncate text to fit within max_length tokens."""
    encoded = tokenizer(text, truncation=True, max_length=max_length, return_tensors="pt")
    return tokenizer.decode(encoded.input_ids[0], skip_special_tokens=True)

def experiment(args):
    # Load models
    logging.info(f"Loading base model of type {args.base_model}...")
    base_tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    base_model = AutoModelForCausalLM.from_pretrained(args.base_model)

    logging.info(f"Loading mask model of type {args.mask_model}...")
    mask_tokenizer = AutoTokenizer.from_pretrained(args.mask_model)
    mask_model = AutoModelForSeq2SeqLM.from_pretrained(args.mask_model)

    model_config = {
        "base_tokenizer": base_tokenizer,
        "base_model": base_model,
        "mask_tokenizer": mask_tokenizer,
        "mask_model": mask_model,
    }

    # Process test data
    logging.info(f"Processing {args.test_data_path}")
    test_data = json.load(open(args.test_data_path, "r"))

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Preprocess and truncate texts
    for item in test_data:
        item["global_id"] = item.get("global_id", None)

        text = item.get("text")
        if not text:  # 如果 text 为空或缺失
            text = item.get("comments")

        item["original_text"] = text
        item["text"] = preprocess_text(text)  # Remove Markdown
        item["text"] = truncate_text(item["text"], mask_tokenizer, max_length=512)  # For t5-small
        item["text_for_ll"] = truncate_text(item["text"], base_tokenizer, max_length=2048)  # For gpt-neo

    perturb_fn = functools.partial(perturb_texts, args=args, model_config=model_config)

    # Perturb texts
    mask_model.eval()
    mask_model.cuda()
    for item in tqdm.tqdm(test_data):
        max_attempts = 5
        for attempt in range(max_attempts):
            try:
                perturbed_text = perturb_fn([item["text"] for _ in range(max(args.n_perturbation_list))])
                if len(perturbed_text) == max(args.n_perturbation_list):
                    item["perturbed_text"] = perturbed_text
                    break
                else:
                    logging.warning(f"Attempt {attempt + 1}: Got {len(perturbed_text)} perturbed samples, expected {max(args.n_perturbation_list)}")
            except Exception as e:
                logging.warning(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt == max_attempts - 1:
                logging.error(f"Failed to perturb text after {max_attempts} attempts: {item['text'][:50]}...")
                item["perturbed_text"] = []
                item["skipped"] = True

    mask_model.to("cpu")

    # Compute log probabilities and scores
    base_model.eval()
    base_model.cuda()
    for item in tqdm.tqdm(test_data):
        if item.get("skipped", False):
            item["text_ll"] = 0
            item["perturbed_text_ll"] = []
            for n_perturbation in args.n_perturbation_list:
                item[f"perturbed_text_ll_{n_perturbation}"] = 0
                item[f"perturbed_text_ll_std_{n_perturbation}"] = 1
                item[f"detectgpt_score_{n_perturbation}"] = 0
            continue

        try:
            item["text_ll"] = get_ll(item["text_for_ll"], args, base_tokenizer, base_model)
        except Exception as e:
            logging.error(f"Failed to compute text_ll for text: {item['text'][:50]}... Error: {str(e)}")
            item["text_ll"] = 0
            item["skipped"] = True
            item["perturbed_text_ll"] = []
            for n_perturbation in args.n_perturbation_list:
                item[f"perturbed_text_ll_{n_perturbation}"] = 0
                item[f"perturbed_text_ll_std_{n_perturbation}"] = 1
                item[f"detectgpt_score_{n_perturbation}"] = 0
            continue

        try:
            perturbed_text_ll = get_lls(item["perturbed_text"], args, base_tokenizer, base_model) if item["perturbed_text"] else []
            item["perturbed_text_ll"] = perturbed_text_ll
        except Exception as e:
            logging.error(f"Failed to compute perturbed_text_ll for text: {item['text'][:50]}... Error: {str(e)}")
            item["perturbed_text_ll"] = []
            item["skipped"] = True

        for n_perturbation in args.n_perturbation_list:
            valid_lls = [i for i in item["perturbed_text_ll"][:n_perturbation] if i != 0] if item["perturbed_text_ll"] else []
            item[f"perturbed_text_ll_{n_perturbation}"] = np.mean(valid_lls) if valid_lls else 0
            item[f"perturbed_text_ll_std_{n_perturbation}"] = np.std(valid_lls) if len(valid_lls) > 1 else 1
            item[f"detectgpt_score_{n_perturbation}"] = (
                item["text_ll"] - item[f"perturbed_text_ll_{n_perturbation}"]
            ) / item[f"perturbed_text_ll_std_{n_perturbation}"] if item[f"perturbed_text_ll_std_{n_perturbation}"] != 0 else 0

    base_model.to("cpu")

    # Save results
    output_file = args.test_data_path.split(".json")[0] + "_DetectGPT_detected.json"
    with open(output_file, "w") as f:
        json.dump(test_data, f, indent=4)
    logging.info(f"Results saved to {output_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_data_path', type=str, required=True,
                        help="Path to the test data.")
    parser.add_argument('--base_model', default="EleutherAI/gpt-neo-2.7B", type=str)
    parser.add_argument('--mask_model', default="t5-small", type=str)
    parser.add_argument('--output_file', default="output_file", type=str)
    parser.add_argument('--n_perturbation_list', default="[10]", type=str,
                        help="List of perturbation counts, e.g., '[1, 10, 20]'")
    parser.add_argument('--span_length', type=int, default=1)
    parser.add_argument('--buffer_size', type=int, default=1)
    parser.add_argument('--mask_top_p', type=float, default=1.0)
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--pct_words_masked', type=float, default=0.05)  # Reduced for stability
    parser.add_argument('--DEVICE', default="cuda", type=str, required=False)
    parser.add_argument('--seed', default=2023, type=int, required=False)
    args = parser.parse_args()

    # Convert n_perturbation_list from string to list
    args.n_perturbation_list = ast.literal_eval(args.n_perturbation_list)
    experiment(args)