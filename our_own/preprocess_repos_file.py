#!/usr/bin/env python3
"""
classify_repo_files.py

功能：
- 扫描 repo 并输出：
  1) text_files.jsonl
  2) code_files.jsonl
  3) other_files.jsonl
  4) text_content_files.jsonl
  5) code_content_only.jsonl
  6) comments_content_only.jsonl
  7) code_content_small_trunk.jsonl
  8) comments_content_small_trunk.jsonl
  9) text_content_small_size.json   <-- Markdown JSON 数组
 10) other_content_files.jsonl
 11) comments_content_small_size.json  <-- 注释 JSON 数组
"""

from pathlib import Path
import json
import re
import ast
import os
import argparse


DEFAULT_TEXT_EXTS = {".md"}
DEFAULT_CODE_EXTS = {".go", ".java", ".js", ".ts", ".php", ".py", ".rb"}


# ---------------- 参数解析 ----------------
def parse_args():
    parser = argparse.ArgumentParser(description="Classify repo files and extract content.")
    parser.add_argument("--repo_path", required=True, help="要扫描的仓库路径")
    parser.add_argument("--output_dir", required=True, help="输出目录路径")
    parser.add_argument("--code-ext", default=None, help="自定义代码扩展（逗号分隔）")
    parser.add_argument("--text-ext", default=None, help="自定义文本扩展（逗号分隔）")
    parser.add_argument("--max_words", type=int, default=512, help="文本分块最大词数（默认512）")
    return parser.parse_args()


def normalize_ext_list(ext_str):
    if not ext_str:
        return None
    exts = set()
    for p in ext_str.split(","):
        p = p.strip()
        if p and not p.startswith("."):
            p = "." + p
        if p:
            exts.add(p.lower())
    return exts


# ---------------- 分类文件 ----------------
def classify_files(repo_path: Path, text_exts, code_exts):
    text_files, code_files, other_files = [], [], []
    for p in repo_path.rglob("*"):
        if p.is_file():
            ext = p.suffix.lower()
            if ext in text_exts:
                text_files.append(str(p))
            elif ext in code_exts:
                code_files.append(str(p))
            else:
                other_files.append(str(p))
    return text_files, code_files, other_files


def save_as_jsonl(file_list, output_path, ftype):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for file in file_list:
            f.write(json.dumps({"type": ftype, "path": file}, ensure_ascii=False) + "\n")
    print(f"✅ Saved {len(file_list)} {ftype} files → {output_path}")
    return output_path


def load_file_list(jsonl_path):
    paths = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            paths.append(data["path"])
    return paths


# ---------------- 保存文本/other 内容 ----------------
def save_text_content(file_paths, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f_out:
        for path in file_paths:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
                f_out.write(json.dumps({"path": path, "content": content}, ensure_ascii=False) + "\n")
            except Exception as e:
                print(f"⚠️ Cannot read {path}: {e}")
    print(f"✅ Saved text contents → {output_path}")


def save_other_content(file_paths, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f_out:
        for path in file_paths:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
                f_out.write(json.dumps({"path": path, "content": content}, ensure_ascii=False) + "\n")
            except Exception as e:
                print(f"⚠️ Skipped {path}: {e}")
    print(f"✅ Saved other files → {output_path}")


# ---------------- 拆分代码/注释 ----------------
def split_code_and_comments(content):
    code_lines, comment_lines = [], []
    content = re.sub(r"/\*.*?\*/", "", content, flags=re.DOTALL)
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("#") or stripped.startswith("//"):
            comment_lines.append(line)
        else:
            code_lines.append(line)
    return "\n".join(code_lines).strip(), "\n".join(comment_lines).strip()


def save_code_comment_split(code_paths, code_output, comment_output):
    with open(code_output, "w", encoding="utf-8") as code_f, open(comment_output, "w", encoding="utf-8") as com_f:
        for path in code_paths:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
                code, comments = split_code_and_comments(content)
                if code:
                    code_f.write(json.dumps({"path": path, "code": code}, ensure_ascii=False) + "\n")
                if comments:
                    com_f.write(json.dumps({"path": path, "comments": comments}, ensure_ascii=False) + "\n")
            except Exception as e:
                print(f"⚠️ Failed {path}: {e}")
    print(f"✅ Saved code/comment splits")


# ---------------- 代码块拆分 ----------------
def detect_lang(path):
    ext = Path(path).suffix.lower()
    return {
        ".py": "python", ".java": "java", ".js": "js",
        ".ts": "ts", ".go": "go", ".php": "php", ".rb": "ruby"
    }.get(ext, "unknown")


def split_python_code(code):
    try:
        tree = ast.parse(code)
        lines = code.splitlines()
        blocks = []
        for node in tree.body:
            s, e = getattr(node, "lineno", None), getattr(node, "end_lineno", None)
            if s and e:
                blocks.append("\n".join(lines[s - 1:e]))
        return blocks or [code]
    except Exception:
        return [code]


def split_java_code(code):
    lines = code.splitlines()
    blocks, stack, start = [], [], 0
    for i, line in enumerate(lines):
        if "{" in line:
            stack.append("{")
        if "}" in line and stack:
            stack.pop()
        if not stack and "{" in line:
            blocks.append("\n".join(lines[start:i + 1]).strip())
            start = i + 1
    if start < len(lines):
        rest = "\n".join(lines[start:]).strip()
        if rest:
            blocks.append(rest)
    return blocks or [code]


def split_code_blocks(input_jsonl, output_jsonl):
    with open(input_jsonl, "r", encoding="utf-8") as f_in, open(output_jsonl, "w", encoding="utf-8") as f_out:
        for idx, line in enumerate(f_in, start=1):
            data = json.loads(line)
            path, code = data["path"], data["code"]
            lang = detect_lang(path)
            if lang == "python":
                blocks = split_python_code(code)
            elif lang == "java":
                blocks = split_java_code(code)
            else:
                blocks = [l for l in code.splitlines() if l.strip()]
            for j, block in enumerate(blocks, start=1):
                f_out.write(json.dumps({
                    "global_id": f"{idx}.{j}",
                    "path": path,
                    "lang": lang,
                    "code": block
                }, ensure_ascii=False) + "\n")
    print(f"✅ Saved code_content_small_trunk.jsonl")


def split_comment_blocks(input_jsonl, output_jsonl):
    with open(input_jsonl, "r", encoding="utf-8") as f_in, open(output_jsonl, "w", encoding="utf-8") as f_out:
        for idx, line in enumerate(f_in, start=1):
            data = json.loads(line)
            path, comments = data.get("path"), data.get("comments", "")
            if not comments:
                continue
            lines = [l for l in comments.splitlines() if l.strip()]
            for j, block in enumerate(lines, start=1):
                f_out.write(json.dumps({
                    "global_id": f"{idx}.{j}",
                    "path": path,
                    "comments": block
                }, ensure_ascii=False) + "\n")
    print(f"✅ Saved comments_content_small_trunk.jsonl")


# ---------------- 句子分块（Markdown / 注释） ----------------
def split_text_into_sentences(text):
    sents = re.split(r'([。！？!?\.])', text)
    if not sents:
        return []
    paired = ["".join(x) for x in zip(sents[0::2], sents[1::2])]
    if len(sents) % 2 != 0:
        paired.append(sents[-1])
    return [s.strip() for s in paired if s.strip()]


def chunk_by_sentences(text, max_words=512):
    sents = split_text_into_sentences(text)
    chunks, cur, count = [], [], 0
    for s in sents:
        words = s.split()
        if count + len(words) > max_words and cur:
            chunks.append(" ".join(cur))
            cur, count = [], 0
        cur.append(s)
        count += len(words)
    if cur:
        chunks.append(" ".join(cur))
    return chunks


def save_markdown_chunks_as_json(input_jsonl, output_json, max_words=512):
    data_list = []
    with open(input_jsonl, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            d = json.loads(line)
            path, content = d["path"], d["content"]
            if not path.lower().endswith(".md"):
                continue
            chunks = chunk_by_sentences(content, max_words)
            for j, chunk in enumerate(chunks, start=1):
                data_list.append({
                    "global_id": f"{i}.{j}",
                    "path": path,
                    "text": chunk
                })
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(data_list, f, ensure_ascii=False, indent=4)
    print(f"✅ Saved text_content_small_size.json (JSON array)")


def save_comment_chunks_as_json(input_jsonl, output_json, max_words=512):
    data_list = []
    with open(input_jsonl, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            d = json.loads(line)
            path, comments = d["path"], d.get("comments", "")
            if not comments:
                continue
            chunks = chunk_by_sentences(comments, max_words)
            for j, chunk in enumerate(chunks, start=1):
                data_list.append({
                    "global_id": f"{i}.{j}",
                    "path": path,
                    "comments": chunk
                })
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(data_list, f, ensure_ascii=False, indent=4)
    print(f"✅ Saved comments_content_small_size.json (JSON array)")


# ---------------- 主流程 ----------------
def main():
    args = parse_args()
    repo = Path(args.repo_path).resolve()
    out = Path(args.output_dir).resolve()
    text_ext = normalize_ext_list(args.text_ext) or DEFAULT_TEXT_EXTS
    code_ext = normalize_ext_list(args.code_ext) or DEFAULT_CODE_EXTS

    print(f"📁 Repo: {repo}")
    print(f"📂 Output: {out}")

    # 分类
    text_files, code_files, other_files = classify_files(repo, text_ext, code_ext)
    text_jsonl = save_as_jsonl(text_files, out / "text_files.jsonl", "text")
    code_jsonl = save_as_jsonl(code_files, out / "code_files.jsonl", "code")
    other_jsonl = save_as_jsonl(other_files, out / "other_files.jsonl", "other")

    # 内容提取
    save_text_content(load_file_list(text_jsonl), out / "text_content_files.jsonl")
    save_code_comment_split(load_file_list(code_jsonl), out / "code_content_only.jsonl", out / "comments_content_only.jsonl")
    save_other_content(load_file_list(other_jsonl), out / "other_content_files.jsonl")

    # 拆分块
    split_code_blocks(out / "code_content_only.jsonl", out / "code_content_small_trunk.jsonl")
    split_comment_blocks(out / "comments_content_only.jsonl", out / "comments_content_small_trunk.jsonl")

    # Markdown 和 Comment JSON 数组
    save_markdown_chunks_as_json(out / "text_content_files.jsonl", out / "text_content_small_size.json", args.max_words)
    save_comment_chunks_as_json(out / "comments_content_only.jsonl", out / "comments_content_small_size.json", args.max_words)

    print("\n🎯 All 12 files generated successfully!")


if __name__ == "__main__":
    main()
