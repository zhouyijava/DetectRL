import json
import os
import pandas as pd
import argparse

def main():
    parser = argparse.ArgumentParser(description="根据阈值判断文本是否为AI生成")
    parser.add_argument("--folder", type=str, required=True, help="数据文件所在的文件夹路径")
    parser.add_argument("--prefix", type=str, required=True, help="文件前缀，例如 text_content_small_size_")
    args = parser.parse_args()

    folder = args.folder
    prefix = args.prefix

    # === 各文件的特征名与阈值定义 ===
    criteria = {
        "bino_data": ("bino_score", -0.9208333492279053),
        "DetectGPT_detected": ("detectgpt_score_10", 0.1),
        "entropy_data": ("entropy", 2.961249828338623),
        "Fast_DetectGPT_results": ("text_crit", 0.4369613095238095),
        "likelihood_results": ("log_likelihood", -2.239182710647583),
        "logRank_data": ("text_logrank", -1.119217038154602),
        "LRR_result": ("text_LRR", 2.069418450381845),
        "NPR_results": ("npr_100", 1.0310512781143188),
        "rank_data": ("text_rank", -52.50980758666992)
    }

    results = []

    for key, (feature, threshold) in criteria.items():
        file_path = os.path.join(folder, f"{prefix}{key}.json")
        if not os.path.exists(file_path):
            print(f"⚠️ 未找到文件: {file_path}，跳过。")
            continue

        # 读取 JSON 文件
        with open(file_path, "r") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                print(f"⚠️ 文件 {file_path} 格式错误，跳过。")
                continue

        # 如果是字典则转为列表
        if isinstance(data, dict):
            data = list(data.values())

        df = pd.DataFrame(data)

        if feature not in df.columns:
            print(f"⚠️ 特征 {feature} 不在 {file_path} 中，跳过。")
            continue

        # 判断 AI / Human
        label_col = f"{key}_label"
        df[label_col] = df[feature].apply(lambda x: "AI" if x > threshold else "human")

        # 只保留关键列
        df = df[[feature, label_col]]
        results.append(df.rename(columns={feature: feature, label_col: f"{key}_label"}))

    # === 合并输出 ===
    if results:
        combined = pd.concat(results, axis=1)

        # 精简列名：去掉重复的 key 名
        new_columns = []
        for col in combined.columns:
            if col.endswith("_label"):
                # 例：bino_data_label → bino_label
                name = col.replace("_data_label", "_label").replace("_result_label", "_label").replace("_results_label", "_label")
                parts = name.split("_")
                # 只保留特征的前缀（bino, detectgpt, entropy...）
                if len(parts) > 2:
                    name = parts[0] + "_label"
                new_columns.append(name)
            else:
                new_columns.append(col)
        combined.columns = new_columns

        output_file = os.path.join(folder, f"{prefix}ai_human_judgement_summary.csv")
        combined.to_csv(output_file, index=False)
        print(f"✅ 已生成结果文件：{output_file}")
    else:
        print("❌ 未生成结果，可能文件或字段未找到。")

if __name__ == "__main__":
    main()
