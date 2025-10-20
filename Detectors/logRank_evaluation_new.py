import logging
import torch
import tqdm
import argparse
import json
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
        data = json.load(open(filename, "r"))

        for i, item in enumerate(tqdm.tqdm(data)):
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

            # inputs = base_tokenizer(text, max_length=max_length, truncation=True, return_tensors="pt")
            # truncated_text = base_tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)

            item["text_logrank"] = -get_rank(text, args, base_tokenizer, base_model, log=True)

        print('logrank computation complete')
        print(filenames)

        with open(filename.split(".json")[0] + "_logRank_data.json", "w") as f:
            json.dump(data, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_data_path', type=str, required=True,
                        help="Path to the data file. Could be several files separated by ','.")
    parser.add_argument('--base_model', default="EleutherAI/gpt-neo-2.7B", type=str, required=False)
    parser.add_argument('--DEVICE', default="cuda", type=str, required=False)
    args = parser.parse_args()

    experiment(args)