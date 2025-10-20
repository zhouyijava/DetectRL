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
from rank import get_rank, get_ranks
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
import re
import gc

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler("npr_debug.log"),
        logging.StreamHandler()
    ]
)


def clean_text(text):
    """清洗文本，处理特殊字符"""
    if not text:
        return ""
    # 移除可能引起问题的特殊字符
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # 移除非ASCII字符
    return text.strip()

def clear_gpu_memory():
    """清理GPU内存"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    logging.info("GPU memory cleared")

def process_chunk(chunk, args, model_config, chunk_idx, total_chunks, filename):
    """处理一个数据块"""
    base_tokenizer = model_config["base_tokenizer"]
    base_model = model_config["base_model"]
    mask_tokenizer = model_config["mask_tokenizer"]
    mask_model = model_config["mask_model"]
    
    perturb_fn = functools.partial(perturb_texts, args=args, model_config=model_config)

    # Perturb texts for this chunk
    mask_model.eval()
    mask_model.cuda()
    for item in tqdm.tqdm(chunk, desc=f"Perturbing chunk {chunk_idx+1}/{total_chunks}"):
        text = item.get("text")
        if not text:  # if text is None or empty string
            text = item.get("comments")  # safely get comments
        
        if text:
            text = clean_text(text)
        
        try:
            # Check token length and truncate if necessary
            inputs = mask_tokenizer(text, return_tensors="pt", truncation=False)
            token_length = len(inputs.input_ids[0])
            logging.info(f"Token length: {token_length}")
            text_to_perturb = text
            if token_length > 512:
                logging.info(f"Text exceeds 512 tokens, truncating to 512 tokens...")
                inputs = mask_tokenizer(text, max_length=512, truncation=True, return_tensors="pt")
                text_to_perturb = mask_tokenizer.decode(inputs.input_ids[0], skip_special_tokens=True)

            perturbed_text = perturb_fn([text_to_perturb for _ in range(max(args.n_perturbation_list))])
            logging.info(f"Perturbed texts (first 2): {perturbed_text[:2]}")
            assert len(perturbed_text) == max(
                args.n_perturbation_list), f"Expected {max(args.n_perturbation_list)} perturbed samples, got {len(perturbed_text)}"
            item["perturbed_text"] = perturbed_text
        except Exception as e:
            logging.error(f"Failed to perturb text {text}: {str(e)}")
            item["perturbed_text"] = [None for _ in range(max(args.n_perturbation_list))]
    
    mask_model.to("cpu")
    clear_gpu_memory()

    # Compute ranks and NPR values for this chunk
    base_model.eval()
    base_model.cuda()
    for item in tqdm.tqdm(chunk, desc=f"Computing ranks for chunk {chunk_idx+1}/{total_chunks}"):
        text = item.get("text")
        if not text:  # if text is None or empty string
            text = item.get("comments")  # safely get comments
        try:
            item["text_logrank"] = get_rank(text, args, base_tokenizer, base_model)
            logging.info(f"Text logrank: {item['text_logrank']}")
            perturbed_text_rank = get_ranks(item["perturbed_text"], args, base_tokenizer, base_model, log=True)
            item["perturbed_text_logrank"] = perturbed_text_rank
            logging.info(f"Perturbed text logranks: {perturbed_text_rank[:2]}")

            # Calculate NPR for each perturbation level
            for n_perturbation in args.n_perturbation_list:
                valid_ranks = [rank for rank in perturbed_text_rank[:n_perturbation] if rank is not None]
                if valid_ranks and item["text_logrank"] is not None and item["text_logrank"] != 0:
                    mean_perturbed_logrank = np.mean(valid_ranks)
                    item[f"npr_{n_perturbation}"] = mean_perturbed_logrank / item["text_logrank"]
                    logging.info(f"NPR_{n_perturbation}: {item[f'npr_{n_perturbation}']}")
                else:
                    item[f"npr_{n_perturbation}"] = None
                    logging.warning(f"Could not compute NPR_{n_perturbation} for text {text}")
        except Exception as e:
            logging.error(f"Failed to rank text {text}: {str(e)}")
            item["text_logrank"] = None
            item["perturbed_text_logrank"] = [None for _ in range(max(args.n_perturbation_list))]
            for n_perturbation in args.n_perturbation_list:
                item[f"npr_{n_perturbation}"] = None
    
    base_model.to("cpu")
    clear_gpu_memory()
    
    return chunk

def experiment(args):
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

    filenames = args.test_data_path.split(",")
    for filename in filenames:
        logging.info(f"Processing data from {filename}")
        test_data = json.load(open(filename, "r"))
        
        # 清理文本
        for item in test_data:
            text = item.get("text")
            if not text:
                text = item.get("comments")
            if text:
                text = clean_text(text)

        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        # 分块处理
        chunk_size = 500a
        total_items = len(test_data)
        total_chunks = (total_items + chunk_size - 1) // chunk_size  # 向上取整
        
        all_results = []
        
        for chunk_idx in range(total_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min((chunk_idx + 1) * chunk_size, total_items)
            chunk = test_data[start_idx:end_idx]
            
            logging.info(f"Processing chunk {chunk_idx+1}/{total_chunks} (items {start_idx}-{end_idx-1})")
            
            processed_chunk = process_chunk(chunk, args, model_config, chunk_idx, total_chunks, filename)
            all_results.extend(processed_chunk)
            
            # 每处理完一个块就保存一次结果
            output_filename = filename.split(".json")[0] + "_NPR_results.json"
            with open(output_filename, "w") as f:
                json.dump(all_results, f, indent=4)
            logging.info(f"Progress saved to {output_filename} after chunk {chunk_idx+1}/{total_chunks}")
        
        logging.info(f"Completed processing {filename}. Final results saved to {output_filename}")

    # 最终清理
    clear_gpu_memory()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_data_path', type=str, required=True,
                        help="Path to the test data. Could be several files separated by ','.")
    parser.add_argument('--base_model', default="EleutherAI/gpt-neo-2.7B", type=str)
    parser.add_argument('--mask_model', default="t5-small", type=str)
    parser.add_argument('--n_perturbation_list', default=[1, 10, 20, 50, 100], type=list)
    parser.add_argument('--span_length', type=int, default=2)
    parser.add_argument('--mask_top_p', type=float, default=1.0)
    parser.add_argument('--buffer_size', type=int, default=1)
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--pct_words_masked', type=float, default=0.3)
    parser.add_argument('--DEVICE', default="cuda", type=str, required=False)
    parser.add_argument('--seed', default=2023, type=int, required=False)
    args = parser.parse_args()

    experiment(args)