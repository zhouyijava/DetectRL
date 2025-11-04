import torch
# rank_safe.py  （把改动应用到你的 rank.py）
import torch

def safe_tokenize_for_model(tokenizer, model, text, truncation=True):
    """
    对文本进行安全的 tokenization：
    - 使用 model.config.max_position_embeddings 或 tokenizer.model_max_length 做截断上限
    - 返回 tokenized dict，包含 tensors（尚未移动到 device）
    """
    # 首先确定最大长度（优先用模型定义的 max position）
    max_len = None
    try:
        max_len = int(getattr(model.config, "max_position_embeddings"))
    except Exception:
        pass
    if max_len is None:
        try:
            max_len = int(getattr(tokenizer, "model_max_length"))
        except Exception:
            max_len = 1024  # 保守默认

    # Ensure reasonable fallback (avoid absurd large values)
    if max_len <= 0 or max_len > 65536:
        max_len = 1024

    # Tokenize with truncation to max_len
    tokenized = tokenizer(
        text,
        return_tensors="pt",
        truncation=truncation,
        max_length=max_len,
        padding=False
    )
    return tokenized, max_len

def move_tokenized_to_device(tokenized, device):
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in tokenized.items()}

def get_rank_safe(text, args, tokenizer, model, device=None):
    """
    一个安全的 get_rank 实现：
    - 截断到模型的 position limit
    - 把 tensors 移到 model device
    - 捕获异常并在 CPU 上尝试复现以获得更清晰的错误信息
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model_device = next(model.parameters()).device if any(True for _ in model.parameters()) else device

    tokenized, max_len = safe_tokenize_for_model(tokenizer, model, text, truncation=True)

    # quick sanity logging (可改为 logging.debug)
    try:
        input_ids = tokenized.get("input_ids")
        if input_ids is not None:
            # token length
            tok_len = input_ids.shape[1]
            # max token id
            try:
                max_tok_id = int(torch.max(input_ids).item())
            except Exception:
                max_tok_id = None
            # model vocab size
            vocab_size = getattr(model.config, "vocab_size", None)
            # log to stdout or logger
            print(f"[safe_get_rank] tok_len={tok_len} max_tok_id={max_tok_id} vocab_size={vocab_size} max_pos={max_len}")
            if tok_len > max_len:
                print(f"[safe_get_rank] WARNING: token length {tok_len} > max_pos {max_len} (should be truncated)")
    except Exception:
        pass

    # move inputs to model's device
    try:
        tokenized = move_tokenized_to_device(tokenized, model_device)
        model.to(model_device)
    except Exception as e:
        print(f"[safe_get_rank] Failed to move to device {model_device}: {e}")

    # Forward pass (with try / except to catch CUDA assertions early and reproduce on CPU)
    try:
        model.eval()
        with torch.no_grad():
            out = model(**tokenized)
            # original code probably took logits and then did something like logits[:, :-1]
            logits = out.logits
            # your ranking logic goes here; for example:
            # return some rank computed from logits
            return logits  # 改成你实际返回值 (示例)
    except Exception as e:
        print(f"[safe_get_rank] Exception during model forward on device {model_device}: {repr(e)}")
        # 尝试在 CPU 上复现以获取更清晰的错误信息
        try:
            print("[safe_get_rank] Attempting CPU reproduction...")
            cpu_model = model.to("cpu")
            cpu_tokenized = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in tokenized.items()}
            cpu_model.eval()
            with torch.no_grad():
                _ = cpu_model(**cpu_tokenized)
            print("[safe_get_rank] CPU forward succeeded (weird) — might be GPU-only or intermittent issue.")
        except Exception as cpu_e:
            print(f"[safe_get_rank] CPU reproduction exception: {repr(cpu_e)}")
        # 抛出原始异常给上层处理
        raise


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


def get_ranks(texts, args, tokenizer, model, log=True):
    return [get_rank_safe(text, args, tokenizer, model, log=log) for text in texts]
