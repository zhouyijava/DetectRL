#!/usr/bin/env python3
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
import itertools
import threading
import time
import argparse

def find_json_files(root_dir):
    """逐个 yield json 文件路径"""
    for dirpath, _, filenames in os.walk(root_dir):
        for name in filenames:
            if name.lower().endswith("content_small_size.json"):
            # if name.lower().endswith("comments_content_small_size.json"):
            # if name.lower().endswith("text_content_small_size.json"):
                yield os.path.join(dirpath, name)

def get_next_gpu():
    """线程安全地轮流分配 GPU"""
    with gpu_lock:
        return next(gpu_cycle)


def run_detector(script_path, json_path):
    """运行单个 detector 脚本"""
    gpu_id = get_next_gpu()
    cmd = ["python", script_path, "--test_data_path", json_path]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    json_name = os.path.basename(json_path)
    detector_name = os.path.basename(script_path).replace(".py", "")
    log_path = os.path.join(LOG_DIR, f"{detector_name}_{json_name}.log")

    with open(log_path, "w") as logf:
        try:
            subprocess.run(cmd, check=True, stdout=logf, stderr=subprocess.STDOUT, env=env)
            return (script_path, gpu_id, True)
        except subprocess.CalledProcessError:
            return (script_path, gpu_id, False)

def process_one_json(json_path):
    """处理单个 JSON 文件：依次运行多个 detector（并行）"""
    print(f"\n🧩 Processing JSON: {json_path}")

    with ThreadPoolExecutor(max_workers=WORKERS_PER_JSON) as executor:
        futures = {executor.submit(run_detector, s, json_path): s for s in DETECTOR_SCRIPTS}

        for i, future in enumerate(as_completed(futures), start=1):
            script = futures[future]
            try:
                script_path, gpu_id, ok = future.result()
                status = "✅ OK" if ok else "❌ FAIL"
                print(f"   [{i}/{len(DETECTOR_SCRIPTS)}] GPU{gpu_id}: {status} {os.path.basename(script_path)}")
            except Exception as e:
                print(f"   ⚠️  Exception on {script}: {e}")

    # 清理 GPU 内存（尤其是 subprocess 已运行完毕）
    import torch
    torch.cuda.empty_cache()
    time.sleep(10)  # 给 GPU 一点时间释放资源

def process_one_detector(detector, all_jsons):
    """处理单个 JSON 文件：依次运行多个 detector（并行）"""
    print(f"\n🧩 Detector processing : {detector}")

    with ThreadPoolExecutor(max_workers=WORKERS_PER_JSON) as executor:
        futures = {executor.submit(run_detector, detector, json_path): json_path for json_path in all_jsons}

        for i, future in enumerate(as_completed(futures), start=1):
            script = futures[future]
            try:
                script_path, gpu_id, ok = future.result()
                status = "✅ OK" if ok else "❌ FAIL"
                print(f"   [{i}/{len(DETECTOR_SCRIPTS)}] GPU{gpu_id}: {status} {os.path.basename(script_path)}")
            except Exception as e:
                print(f"   ⚠️  Exception on {script}: {e}")

    # 清理 GPU 内存（尤其是 subprocess 已运行完毕）
    import torch
    torch.cuda.empty_cache()
    time.sleep(10)  # 给 GPU 一点时间释放资源

def main():
    all_jsons = list(find_json_files(ROOT_DIR))
    print(f"✅ Found {len(all_jsons)} JSON files under {ROOT_DIR}")
    print(f"🚀 Will process one file at a time, using {WORKERS_PER_JSON} detectors concurrently.\n")

    for idx, json_path in enumerate(all_jsons, start=1):
        print(f"\n📄 [{idx}/{len(all_jsons)}] {json_path}")
        process_one_json(json_path)
    # for idx, detector in enumerate(DETECTOR_SCRIPTS, start=1):
    #     print(f"\n📄 [{idx}/{len(DETECTOR_SCRIPTS)}] {detector}")
    #     process_one_detector(detector, all_jsons)
    print("\n🎯 All done. Logs saved in", LOG_DIR)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help="")
    args = parser.parse_args()
    # ---------------- CONFIG ----------------
    ROOT_DIR = f"/home/y/yz741/DRL/DetectRL/repository/{args.data_path}"
    DETECTOR_SCRIPTS = [
        # "./Detectors/binoculars_evaluation_new.py",
        # "./Detectors/DetectGPT_evaluation_new_new.py",
        # "./Detectors/entropy_evaluation_new.py",
        "./Detectors/Fast_DetectGPT_evaluation_new.py",
        "./Detectors/likelihood_evaluation_new.py",
        "./Detectors/logRank_evaluation_new.py",
        # "./Detectors/LRR_evaluation_new.py",
        # "./Detectors/NPR_evaluation_new.py",
        "./Detectors/rank_evaluation_new.py",
    ]

    GPUS = [0]          # 3 GPUs
    WORKERS_PER_JSON = 1       # 每个 JSON 同时运行多少 detector（推荐 ≤ num_gpus）
    LOG_DIR = "./logs"
    os.makedirs(LOG_DIR, exist_ok=True)
    # ----------------------------------------
    gpu_lock = threading.Lock()
    gpu_cycle = itertools.cycle(GPUS)

    main()
