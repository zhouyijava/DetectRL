import logging
import random
import numpy as np
import torch
import tqdm
import argparse
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import re

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

def get_rank(text, args, tokenizer, model, log=False):
    with torch.no_grad():
        if text == "":
            return None
        else:
            tokenized = tokenizer(text, return_tensors="pt").to(args.DEVICE)
            logits = model(**tokenized).logits[:, :-1]
            labels = tokenized.input_ids[:, 1:]

            matches = (logits.argsort(-1, descending=True) == labels.unsqueeze(-1)).nonzero()

            assert matches.shape[1] == 3, f"Expected 3 dimensions in matches tensor, got {matches.shape}"

            ranks, timesteps = matches[:, -1], matches[:, -2]

            assert (timesteps == torch.arange(len(timesteps)).to(
                timesteps.device)).all(), "Expected one match per timestep"

            ranks = ranks.float() + 1
            if log:
                ranks = torch.log(ranks)

            return ranks.float().mean().item()

def get_ll(text, args, tokenizer, model):
    with torch.no_grad():
        tokenized = tokenizer(text, return_tensors="pt").to(args.DEVICE)
        labels = tokenized['input_ids']
        if labels.nelement() == 0:
            logging.error(f"Empty input: {text}")
            return 0
        else:
            return -model(**tokenized, labels=labels).loss.item()

def preprocess_text(text):
    """
    Enhanced preprocessing: Remove Markdown, code elements, shell commands, and normalize.
    """
    if not text or not isinstance(text, str):
        return ""
    # Existing Markdown removal
    text = re.sub(r'#+\s*', '', text)
    text = re.sub(r'\[([^\]]*)\]\([^\)]*\)', r'\1', text)
    text = re.sub(r'\*\*([^\*]*)\*\*', r'\1', text)
    text = re.sub(r'\*([^\*]*)\*', r'\1', text)
    text = re.sub(r'__([^_]*)__', r'\1', text)
    text = re.sub(r'_([^_]*)_', r'\1', text)
    text = re.sub(r'```[^`]*```', '', text)
    text = re.sub(r'`[^`]*`', '', text)
    # New: Remove code-like elements (e.g., pip install, dependencies)
    text = re.sub(r'pip install\s+torch.*', '', text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r'dependency\s+.*?(?=\n\n|$)', '', text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r'this repo provides the code for\s+.*?(?=\n\n|$)', '', text, flags=re.IGNORECASE | re.DOTALL)
    # Normalize newlines and pipes (common in code)
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\|+', ' ', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # UTF-8 handling
    try:
        text = text.encode("utf-8").decode("utf-8")
    except UnicodeDecodeError:
        text = "".join(c for c in text if ord(c) < 128)
    return text

def experiment(args):
    # Load model
    logging.info(f"Loading base model of type {args.base_model}...")
    try:
        base_tokenizer = AutoTokenizer.from_pretrained(args.base_model)
        base_model = AutoModelForCausalLM.from_pretrained(args.base_model)
        base_model.eval()
        base_model.to(args.DEVICE)
    except Exception as e:
        logging.error(f"Failed to load model {args.base_model}: {str(e)}")
        return

    filenames = args.test_data_path.split(",")
    for filename in filenames:
        logging.info(f"Test in {filename}")
        try:
            test_data = json.load(open(filename, "r"))
        except Exception as e:
            logging.error(f"Failed to load {filename}: {str(e)}")
            continue

        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        # Preprocess texts and store original
        for item in test_data:
            item["global_id"] = item.get("global_id", None) 

            text = item.get("text")
            if not text or not isinstance(text, str) or not text.strip():
                text = item.get("comments")
            item["original_text"] = text  # Store original text
            item["text"] = preprocess_text(text)  # Preprocess text

        results = []
        for item in tqdm.tqdm(test_data):
            text = item.get("text", "")
            if not text or not isinstance(text, str) or len(text.strip()) == 0:
                logging.warning(f"Skipping invalid text at index {test_data.index(item)}: {item['original_text'][:50]}...")
                item["text_ll"] = 0
                item["text_logrank"] = None
                item["text_LRR"] = float("inf")
                continue

            try:
                item["text_ll"] = get_ll(text, args, base_tokenizer, base_model)
                item["text_logrank"] = get_rank(text, args, base_tokenizer, base_model, log=True)
                if item["text_logrank"] is not None and item["text_logrank"] != 0:
                    item["text_LRR"] = -item["text_ll"] / item["text_logrank"]
                else:
                    item["text_LRR"] = float("inf")
            except Exception as e:
                logging.error(f"Error processing text at index {test_data.index(item)}: {text[:50]}... Error: {str(e)}")
                item["text_ll"] = 0
                item["text_logrank"] = None
                item["text_LRR"] = float("inf")

            if np.isfinite(item["text_LRR"]):
                results.append({
                    "global_id": item["global_id"], 
                    "original_text": item["original_text"],
                    "text": text,
                    "text_ll": item["text_ll"],
                    "text_logrank": item["text_logrank"],
                    "text_LRR": item["text_LRR"]
                })

        

        # Save LRR results
        output_result_file = filename.split(".json")[0] + "_LRR_result.json"
        with open(output_result_file, "w") as f:
            json.dump(results, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_data_path', type=str, required=True,
                        help="Path to the test data. could be several files with ','. "
                             "Note that the data should have been perturbed.")
    parser.add_argument('--base_model', default="EleutherAI/gpt-neo-2.7B", type=str, required=False)
    parser.add_argument('--DEVICE', default="cuda", type=str, required=False)
    parser.add_argument('--seed', default=2023, type=int, required=False)
    args = parser.parse_args()

    experiment(args)
