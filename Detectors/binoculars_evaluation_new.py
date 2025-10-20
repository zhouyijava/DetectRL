import logging
import random
import torch
import tqdm
import argparse
import json
from binoculars_detector import Binoculars
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

def experiment(args):
    # Initialize Binoculars (experiments in paper use the "accuracy" mode threshold wherever applicable)
    bino = Binoculars(mode="accuracy", max_token_observed=args.tokens_seen)

    filenames = args.test_data_path.split(",")
    for filename in filenames:
        logging.info(f"Processing {filename}")
        test_data = json.load(open(filename, "r"))

        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        for item in tqdm.tqdm(test_data):
            text = item.get('text')

            if not text:  # 如果 text 为空或缺失
                text = item["comments"]

            score = bino.compute_score(text)
            item["bino_score"] = -score if np.isfinite(score) else None  # 恢复取负号
            item["prediction"] = bino.predict(text) if np.isfinite(score) else "Invalid score"

        # Save the updated data with scores and predictions
        with open(filename.split(".json")[0] + "_bino_data.json", "w") as f:
            json.dump(test_data, f, indent=4)

        # Save the threshold used for reference
        result = {"threshold": bino.threshold}
        with open(filename.split(".json")[0] + "_bino_result.json", "w") as f:
            json.dump(result, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_data_path', type=str, required=True,
                        help="Path to the test data. could be several files with ','. "
                             "Note that the data does not need to be perturbed.")
    parser.add_argument("--tokens_seen", type=int, default=512, help="Number of tokens seen by the model")
    parser.add_argument('--DEVICE', default="cuda", type=str, required=False)
    parser.add_argument('--seed', default=2023, type=int, required=False)
    args = parser.parse_args()

    experiment(args)