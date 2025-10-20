import logging
import random
import torch
import tqdm
import argparse
import json
import numpy as np
from rank import get_rank
from transformers import AutoTokenizer, AutoModelForCausalLM

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def experiment(args):
    # load model
    logging.info(f"Loading base model of type {args.base_model}...")
    base_tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    base_model = AutoModelForCausalLM.from_pretrained(args.base_model)
    base_model.eval()
    base_model.cuda()
    max_length = 2048

    filenames = args.test_data_path.split(",")
    for filename in filenames:
        logging.info(f"Processing {filename}")
        test_data = json.load(open(filename, "r"))

        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        # for item in tqdm.tqdm(test_data):
        for i, item in enumerate(tqdm.tqdm(test_data)):
            text = item.get("text")

            if not text:  # 如果 text 为空或缺失
                text = item["comments"]
            
            try:
                # 编码文本以检查 token 长度（不截断）
                inputs = base_tokenizer(text, return_tensors="pt")
                token_length = inputs["input_ids"].size(1)
                # logging.info(f"Processing item {i}, token length: {token_length}")

                # 如果 token 长度超过 max_length，进行截断
                if token_length > max_length:
                    # logging.warning(f"Item {i} token length {token_length} exceeds max_length {max_length}, truncating...")
                    inputs = base_tokenizer(text, max_length=max_length, truncation=True, return_tensors="pt")
                    text = base_tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
            
            except Exception as e:
                logging.error(f"Error tokenizing item {i}: {text[0]}... | Error: {str(e)}")
                item["text_logrank"] = 0.0
                continue

            # Calculate the negative rank value to match original code
            item["text_rank"] = -get_rank(text, args, base_tokenizer, base_model, log=False)

            # Handle non-finite values
            if not np.isfinite(item["text_rank"]):
                item["text_rank"] = None

        # Save the updated data with text_rank added to each item
        print(f"Processed ranks for {filename}")
        with open(filename.split(".json")[0] + "_rank_data.json", "w") as f:
            json.dump(test_data, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_data_path', type=str, required=True,
                        help="Path to the test data. could be several files with ','. "
                             "Note: Original code assumed perturbed data, but rank calculation works on non-perturbed text as well.")
    parser.add_argument('--base_model', default="EleutherAI/gpt-neo-2.7B", type=str, required=False)
    parser.add_argument('--DEVICE', default="cuda", type=str, required=False)
    parser.add_argument('--seed', default=2023, type=int, required=False)
    args = parser.parse_args()

    experiment(args)