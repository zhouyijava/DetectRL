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
from rank import get_rank_safe as get_rank, get_ranks
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
    # # region---------- Start: safer compute ranks with diagnostics ----------
    # def tensor_info(name, tensor):
    #     try:
    #         if tensor is None:
    #             logging.info(f"[TENSOR INFO] {name}: None")
    #             return
    #         if isinstance(tensor, torch.Tensor):
    #             logging.info(f"[TENSOR INFO] {name}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}, device={tensor.device}")
    #             if tensor.numel() > 0 and tensor.is_floating_point():
    #                 logging.info(f"    min={float(torch.nanmin(tensor))}, max={float(torch.nanmax(tensor))}, has_nan={bool(torch.isnan(tensor).any())}")
    #             elif tensor.numel() > 0:
    #                 logging.info(f"    min={int(torch.min(tensor).cpu().item())}, max={int(torch.max(tensor).cpu().item())}")
    #     except Exception as ex:
    #         logging.warning(f"[TENSOR INFO] failed to inspect {name}: {ex}")

    # def diagnose_text_failure(text, idx):
    #     """
    #     当 GPU 报错时，尽量把可用信息写入日志并在 CPU 上复现前向推理/推断，以获取更明确的错误堆栈。
    #     """
    #     logging.error(f"Diagnosing failure for item index {idx}, text snippet: {repr(text[:200])}")
    #     try:
    #         # Tokenize with base tokenizer
    #         tokenized = base_tokenizer(text, return_tensors="pt", truncation=False)
    #         input_ids = tokenized.get("input_ids")
    #         logging.info("Base tokenizer vocab_size = %s", getattr(base_model.config, "vocab_size", "N/A"))
    #         if input_ids is not None:
    #             tensor_info("input_ids", input_ids)
    #             # 最大 token id
    #             try:
    #                 max_id = int(torch.max(input_ids).item())
    #                 logging.info(f"    max token id = {max_id}")
    #             except Exception:
    #                 logging.info("    could not get max token id")
    #         # Try running model on CPU (safer) to reproduce and catch CPU exception
    #         logging.info("Attempting a CPU forward pass to reproduce error...")
    #         try:
    #             cpu_model = base_model.to("cpu")
    #             cpu_model.eval()
    #             with torch.no_grad():
    #                 # Move tokenized tensors to cpu
    #                 tokenized_cpu = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in tokenized.items()}
    #                 _ = cpu_model(**tokenized_cpu)
    #             logging.info("CPU forward pass succeeded (no exception). GPU-only issue or non-deterministic.")
    #         except Exception as cpu_ex:
    #             logging.error(f"CPU forward pass raised exception: {repr(cpu_ex)}")
    #     except Exception as outer_ex:
    #         logging.error(f"Diagnose failed: {repr(outer_ex)}")

    # # 将 base_model 放到 cuda 并逐个样本处理，包裹诊断
    # base_model.eval()
    # base_device = "cuda" if torch.cuda.is_available() else "cpu"
    # try:
    #     base_model.to(base_device)
    # except Exception as e:
    #     logging.error(f"Failed to move base_model to {base_device}: {e}")
    #     # 如果直接移动模型就失败，尝试清理并继续
    #     clear_gpu_memory()
    #     base_model.to("cpu")

    # for idx_in_chunk, item in enumerate(tqdm.tqdm(chunk, desc=f"Computing ranks for chunk {chunk_idx+1}/{total_chunks}")):
    #     global_idx = chunk_idx * len(chunk) + idx_in_chunk
    #     text = item.get("text") or item.get("comments")
    #     if text:
    #         text = clean_text(text)
    #     try:
    #         # Debugging: check tokenization for this text before calling get_rank
    #         try:
    #             sample_tokens = base_tokenizer(text, return_tensors="pt", truncation=False)
    #             input_ids = sample_tokens.get("input_ids")
    #             if input_ids is not None:
    #                 # log info but avoid huge prints
    #                 logging.debug(f"[Token check] idx={global_idx}, token_len={input_ids.shape[1]}")
    #                 vocab_size = getattr(base_model.config, "vocab_size", None)
    #                 if vocab_size is not None:
    #                     try:
    #                         max_tok = int(torch.max(input_ids).item())
    #                         if max_tok >= vocab_size:
    #                             logging.error(f"[IndexOutOfRange] text idx={global_idx} has token id {max_tok} >= vocab_size {vocab_size}")
    #                             # Mark and skip, also diagnose
    #                             diagnose_text_failure(text, global_idx)
    #                             item["text_logrank"] = None
    #                             item["perturbed_text_logrank"] = [None for _ in range(max(args.n_perturbation_list))]
    #                             for n_perturbation in args.n_perturbation_list:
    #                                 item[f"npr_{n_perturbation}"] = None
    #                             continue
    #                     except Exception as exx:
    #                         logging.debug(f"Could not compute max token id for idx={global_idx}: {exx}")
    #         except Exception as token_ex:
    #             logging.warning(f"Tokenization check failed for idx={global_idx}: {token_ex}")

    #         # 正常调用 get_rank / get_ranks（假设函数内部做了自己的 .to(device) 调用）
    #         item["text_logrank"] = get_rank(text, args, base_tokenizer, base_model)
    #         logging.info(f"Text logrank idx={global_idx}: {item['text_logrank']}")
    #         perturbed_text_rank = get_ranks(item.get("perturbed_text", []), args, base_tokenizer, base_model, log=True)
    #         item["perturbed_text_logrank"] = perturbed_text_rank
    #         logging.info(f"Perturbed text logranks idx={global_idx}: {perturbed_text_rank[:2]}")

    #         # Calculate NPR for each perturbation level
    #         for n_perturbation in args.n_perturbation_list:
    #             valid_ranks = [rank for rank in perturbed_text_rank[:n_perturbation] if rank is not None]
    #             if valid_ranks and item["text_logrank"] is not None and item["text_logrank"] != 0:
    #                 mean_perturbed_logrank = np.mean(valid_ranks)
    #                 item[f"npr_{n_perturbation}"] = mean_perturbed_logrank / item["text_logrank"]
    #                 logging.info(f"NPR_{n_perturbation} idx={global_idx}: {item[f'npr_{n_perturbation}']}")
    #             else:
    #                 item[f"npr_{n_perturbation}"] = None
    #                 logging.warning(f"Could not compute NPR_{n_perturbation} for text idx={global_idx}")
    #     except Exception as e:
    #         # 出现CUDA/其它异常时：记录、诊断并继续下一个样本
    #         logging.error(f"Exception when processing idx={global_idx}: {repr(e)}")
    #         try:
    #             # 同步 CUDA，确保错误不是异步积累
    #             if torch.cuda.is_available():
    #                 torch.cuda.synchronize()
    #         except Exception:
    #             pass
    #         diagnose_text_failure(text, global_idx)
    #         # 标记为失败，保持结构
    #         item["text_logrank"] = None
    #         item["perturbed_text_logrank"] = [None for _ in range(max(args.n_perturbation_list))]
    #         for n_perturbation in args.n_perturbation_list:
    #             item[f"npr_{n_perturbation}"] = None
    #         # 清理CUDA缓存后继续
    #         clear_gpu_memory()
    # # endregion---------- End: safer compute ranks with diagnostics ----------

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
        chunk_size = 500
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