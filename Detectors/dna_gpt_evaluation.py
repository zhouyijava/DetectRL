import logging
import random
import re

import numpy as np
import torch
import tqdm
import argparse
import json

from transformers import AutoTokenizer, AutoModelForCausalLM
from metrics import get_roc_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def get_log_probs(texts, args, base_tokenizer, base_model):
    batch_size = args.batch_size
    batch_lprobs = []
    for batch in range(len(texts) // batch_size):
        tokenized = base_tokenizer(texts[batch * batch_size:(batch + 1) * batch_size], return_tensors="pt",
                                   padding=True).to(args.DEVICE)
        labels = tokenized.input_ids[:, 1:]
        with torch.no_grad():
            logits_score = base_model(**tokenized).logits[:, :-1]
            lprobs = get_likelihood(logits_score, labels, base_tokenizer.pad_token_id)
            batch_lprobs.append(lprobs)
    return torch.cat(batch_lprobs, dim=0)


def get_dna_gpt(text, args, base_tokenizer, base_model, regen_tokenizer, regen_model):
    lprob = get_log_prob(text, args, base_tokenizer, base_model)
    regens = get_regen_samples(text, args, regen_tokenizer, regen_model)
    lprob_regens = get_log_probs(regens, args, base_tokenizer, base_model)
    wscore = lprob[0] - lprob_regens.mean()
    return wscore.item()

def get_e_dna_gpt(text, args, regen_texts, base_tokenizer, base_model, regen_tokenizer, regen_model):
    lprob = get_log_prob(text, args, base_tokenizer, base_model)
    lprob_regens = get_log_probs(regen_texts, args, base_tokenizer, base_model)
    wscore = lprob[0] - lprob_regens.mean()
    return wscore.item()


def clean_text(text):
    text = re.sub(r'\n', r'', text)
    return text


def truncate_text_to_sentences(text, min_word_count=100):
    word_count = 0
    end_of_last_sentence = 0

    # Split the text into words and iterate over them
    words = text.split()

    for i, word in enumerate(words):
        # Check for sentence-ending punctuation
        if word[-1] in '.!?':
            # Check if we have reached the minimum word count
            if word_count >= min_word_count:
                # We return the text up to the end of the current sentence
                return ' '.join(words[:i + 1])
            end_of_last_sentence = i
        word_count += 1

    # If we have not reached the minimum word count by the end of the text,
    # return the text up to the end of the last complete sentence if any,
    # otherwise return the whole text.
    if end_of_last_sentence > 0:
        return ' '.join(words[:end_of_last_sentence + 1])
    else:
        return ' '.join(words)


def _sample_from_model(texts, args, base_tokenizer, base_model, truncate_ratio=0.5):
    # encode each text as a list of token ids
    texts = [t.split(' ') for t in texts]
    word_count = len(texts[0])
    texts = [' '.join(t[: int(len(t) * truncate_ratio)]) for t in texts]
    all_encoded = base_tokenizer(texts, return_tensors="pt").to(args.DEVICE)

    base_model.eval()
    decoded = ['' for _ in range(len(texts))]

    # sample from the model until we get a sample with at least min_words words for each example
    # this is an inefficient way to do this (since we regenerate for all inputs if just one is too short), but it works
    tries = 0
    m = 0
    while m < 100:
        if tries != 0:
            print()
            print(f"min words: {m}, needed {word_count}, regenerating (try {tries})")

        sampling_kwargs = {'temperature': args.temperature}
        if args.do_top_p:
            sampling_kwargs['top_p'] = args.top_p
        elif args.do_top_k:
            sampling_kwargs['top_k'] = args.top_k
        min_length = 150
        outputs = base_model.generate(**all_encoded, min_length=800, max_length=1024, do_sample=True,
                                      **sampling_kwargs, pad_token_id=base_tokenizer.eos_token_id,
                                      eos_token_id=base_tokenizer.eos_token_id)
        decoded = base_tokenizer.batch_decode(outputs, skip_special_tokens=True)
        m = min(len(x.split()) for x in decoded)
        tries += 1

    for i in range(len(decoded)):
        decoded[i] = clean_text(decoded[i])
        decoded[i] = truncate_text_to_sentences(decoded[i], word_count)

    regen_text_lengths = [len(x.split()) for x in decoded]
    print(f"Sample text length: {regen_text_lengths}, word count: {word_count}")

    return decoded


def generate_samples(texts, args, base_tokenizer, base_model, batch_size):
    assert len(texts) % batch_size == 0
    sampled_texts = []
    for batch in range(len(texts) // batch_size):
        print('Generating samples for batch', batch, 'of', len(texts) // batch_size)
        original_text = texts[batch * batch_size:(batch + 1) * batch_size]

        sampled_text = _sample_from_model(original_text, args, base_tokenizer, base_model, truncate_ratio=args.truncate_ratio)

        sampled_texts.extend(sampled_text)

    return sampled_texts


def get_regen_samples(text, args, base_tokenizer, base_model):
    data = [text] * args.regen_number
    data = generate_samples(data, args, base_tokenizer, base_model, batch_size=args.batch_size)
    return data


def get_likelihood(logits, labels, pad_index):
    labels = labels.unsqueeze(-1) if labels.ndim == logits.ndim - 1 else labels
    lprobs = torch.log_softmax(logits, dim=-1)
    log_likelihood = lprobs.gather(dim=-1, index=labels)
    mask = labels != pad_index
    log_likelihood = (log_likelihood * mask).sum(dim=1) / mask.sum(dim=1)
    return log_likelihood.squeeze(-1)


def get_log_prob(text, args, base_tokenizer, base_model):
    base_tokenizer.pad_token = base_tokenizer.eos_token
    tokenized = base_tokenizer(text, return_tensors="pt", padding=True).to(args.DEVICE)
    labels = tokenized.input_ids[:, 1:]
    with torch.no_grad():
        logits_score = base_model(**tokenized).logits[:, :-1]
        return get_likelihood(logits_score, labels, base_tokenizer.pad_token_id)


def experiment(args):
    # load model
    logging.info(f"Loading base model of type {args.base_model}...")
    base_tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    base_model = AutoModelForCausalLM.from_pretrained(args.base_model)
    base_model.eval()
    base_model.cuda()

    regen_tokenizer = AutoTokenizer.from_pretrained(args.regen_model)
    regen_model = AutoModelForCausalLM.from_pretrained(args.regen_model)
    regen_model.eval()
    regen_model.cuda()

    filenames = args.test_data_path.split(",")
    for filename in filenames:
        logging.info(f"Test in {filename}")
        # test_data = json.load(open(filename, "r"))
        test_data = json.load(open(filename.split(".json")[0] + "_dna_gpt_data.json", "r"))

        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        predictions = {'human': [], 'llm': []}
        for item in tqdm.tqdm(test_data):
            text = item["text"]
            label = item["label"]

            # item["dna_gpt_crit"] = get_dna_gpt(text, args, base_tokenizer, base_model, regen_tokenizer, regen_model)

            item["dna_gpt_crit"] = get_e_dna_gpt(text, args, item["regen_text"], base_tokenizer, base_model, regen_tokenizer, regen_model)

            # result
            if label == "human":
                predictions['human'].append(item["dna_gpt_crit"])
            elif label == "llm":
                predictions['llm'].append(item["dna_gpt_crit"])
            else:
                raise ValueError(f"Unknown label {label}")

        predictions['human'] = [i for i in predictions['human'] if np.isfinite(i)]
        predictions['llm'] = [i for i in predictions['llm'] if np.isfinite(i)]

        roc_auc, optimal_threshold, conf_matrix, precision, recall, f1, accuracy, tpr_at_fpr_0_01 = get_roc_metrics(predictions['human'],
                                                                                                   predictions['llm'])

        result = {
            "roc_auc": roc_auc,
            "optimal_threshold": optimal_threshold,
            "conf_matrix": conf_matrix,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy
        }
        print('dba_gpt')
        print(filenames)
        logging.info(f"{result}")
        with open(filename.split(".json")[0] + "_dna_gpt_data.json", "w") as f:
            json.dump(test_data, f, indent=4)

        with open(filename.split(".json")[0] + "_dna_gpt_result.json", "w") as f:
            json.dump(result, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_data_path', type=str, required=True,
                        help="Path to the test data. could be several files with ','. "
                             "Note that the data should have been perturbed.")
    parser.add_argument('--base_model', default="EleutherAI/gpt-neo-2.7B", type=str, required=False)
    parser.add_argument('--regen_model', default="NousResearch/Meta-Llama-3-8B-Instruct", type=str, required=False)
    parser.add_argument('--DEVICE', default="cuda", type=str, required=False)
    parser.add_argument('--batch_size', default=2, type=int, required=False)
    parser.add_argument('--truncate_ratio', default=0.5, type=float, required=False)
    parser.add_argument('--regen_number', default=10, type=int, required=False)
    parser.add_argument('--top_k', type=int, default=40)
    parser.add_argument('--do_top_p', default=True)
    parser.add_argument('--do_top_k', default=True)
    parser.add_argument('--top_p', type=float, default=0.96)
    parser.add_argument('--temperature', type=float, default=1)
    parser.add_argument('--seed', default=2023, type=int, required=False)
    args = parser.parse_args()

    experiment(args)
