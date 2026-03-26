"""
AI 推理 Pipeline Demo
清晰展示 Prefill 和 Decode 两个阶段的区别

依赖安装：
    pip install torch transformers
"""

import time
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# ─────────────────────────────────────────────
# 1. 加载模型
# ─────────────────────────────────────────────

print("加载模型...")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(DEVICE)
print(f"运行设备: {DEVICE}\n")


# ─────────────────────────────────────────────
# 2. 采样函数（top-k + top-p）
# ─────────────────────────────────────────────

def sample(logits, top_k=50, top_p=0.9, temperature=1.0):
    """从 logits 采样下一个 token"""
    logits = logits / temperature

    # Top-k：只保留概率最高的 k 个
    if top_k > 0:
        top_values, _ = torch.topk(logits, top_k)
        logits[logits < top_values[:, -1:]] = -1e10

    # Top-p：累积概率超过 p 的部分截掉
    if top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_logits[cumulative_probs > top_p] = -1e10
        logits = torch.zeros_like(logits).scatter_(1, sorted_idx, sorted_logits)

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


# ─────────────────────────────────────────────
# 3. Prefill：一次性处理完整 prompt
# ─────────────────────────────────────────────

def prefill(model, input_ids):
    """
    Prefill 阶段：
    - 输入：完整 prompt 的所有 token（shape: [1, N]）
    - 计算：所有 token 两两之间的 attention（N×N 矩阵）
    - 输出：第一个新 token + KV Cache
    - 瓶颈：Compute bound（GPU 算力）
    """
    t0 = time.perf_counter()

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            past_key_values=None,   # 第一次，没有历史
            use_cache=True,         # 让模型把 K,V 存起来
        )

    latency_ms = (time.perf_counter() - t0) * 1000

    # 取最后一个位置的 logits，采样出第一个新 token
    next_token = sample(outputs.logits[:, -1, :])
    kv_cache = outputs.past_key_values

    # 打印 KV Cache 的结构
    cache_list = list(kv_cache)
    num_layers = len(cache_list)
    k_shape = cache_list[0][0].shape # [batch, heads, seq_len, head_dim]
    print(f"  KV Cache: {num_layers} 层，每层 K/V shape = {list(k_shape)}")
    print(f"  seq_len = {k_shape[2]}（等于 prompt 长度）")

    return next_token, kv_cache, latency_ms


# ─────────────────────────────────────────────
# 4. Decode：每次生成 1 个 token
# ─────────────────────────────────────────────

def decode_one_step(model, token, kv_cache):
    """
    Decode 阶段（单步）：
    - 输入：上一步生成的 1 个 token（shape: [1, 1]）
    - 计算：这 1 个 token 和历史所有 token 的 attention（从 KV Cache 读）
    - 输出：下一个 token + 更新后的 KV Cache
    - 瓶颈：Bandwidth bound（VRAM 带宽）
    """
    t0 = time.perf_counter()

    with torch.no_grad():
        outputs = model(
            input_ids=token,            # 只喂 1 个 token
            past_key_values=kv_cache,   # 历史 K,V 直接复用，不重算
            use_cache=True,
        )

    latency_ms = (time.perf_counter() - t0) * 1000
    next_token = sample(outputs.logits[:, -1, :])
    return next_token, outputs.past_key_values, latency_ms


# ─────────────────────────────────────────────
# 5. 完整推理流程
# ─────────────────────────────────────────────

def generate(prompt, max_new_tokens=20):
    print("=" * 55)
    print(f"Prompt: {prompt!r}")
    print("=" * 55)

    # Tokenize prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
    prompt_len = input_ids.shape[1]
    print(f"Prompt 长度: {prompt_len} tokens\n")

    # ── Prefill ──────────────────────────────
    print("[ PREFILL ]")
    print(f"  输入: {prompt_len} 个 token 并行处理")

    first_token, kv_cache, prefill_ms = prefill(model, input_ids)
    first_word = tokenizer.decode(first_token[0])
    print(f"  生成第 1 个新 token: {first_word!r}")
    print(f"  Latency: {prefill_ms:.1f} ms\n")

    # ── Decode ───────────────────────────────
    print("[ DECODE ]")
    generated_tokens = [first_token]
    decode_latencies = []

    token = first_token
    for step in range(1, max_new_tokens):
        token, kv_cache, decode_ms = decode_one_step(model, token, kv_cache)
        generated_tokens.append(token)
        decode_latencies.append(decode_ms)

        seq_len = list(kv_cache)[0][0].shape[2]
        word = tokenizer.decode(token[0])
        print(f"  Step {step:2d} | token: {word!r:12s} | "
              f"KV seq_len: {seq_len:4d} | latency: {decode_ms:.1f} ms")

    # ── 汇总 ─────────────────────────────────
    all_new_ids = torch.cat(generated_tokens, dim=1)
    generated_text = tokenizer.decode(all_new_ids[0])

    avg_decode = sum(decode_latencies) / len(decode_latencies)
    print(f"\n{'─'*55}")
    print(f"Prefill latency:     {prefill_ms:.1f} ms  （处理 {prompt_len} tokens）")
    print(f"Decode avg latency:  {avg_decode:.1f} ms  （每 token）")
    print(f"Prefill/Decode 比值: {prefill_ms/avg_decode:.1f}x")
    print(f"\n生成结果: {prompt}{generated_text}")
    print("=" * 55)


# ─────────────────────────────────────────────
# 6. 运行
# ─────────────────────────────────────────────

if __name__ == "__main__":
    generate(
        prompt="The future of AI inference is",
        max_new_tokens=20,
    )