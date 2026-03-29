"""
AI 推理 Pipeline Demo
清晰展示 Prefill 和 Decode 两个阶段的区别，并对比算子分布

依赖安装：
    pip install torch transformers
"""

import time
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.profiler import profile, ProfilerActivity, record_function

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
# 3. Prefill：一次性处理完整 prompt，带 profiling
# ─────────────────────────────────────────────

def prefill(model, input_ids):
    """
    Prefill 阶段：
    - 输入: 完整 prompt（多个 token 并行处理）
    - 特点: Compute-bound（大矩阵乘法主导），GPU 利用率高
    - 输出: 第一个新 token + KV Cache
    """
    t0 = time.perf_counter()

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,          # 记录张量 shape，方便分析矩阵大小
        with_flops=True,             # 统计 FLOPs，对比两阶段计算量
    ) as prof:
        with record_function("prefill_forward"):  # 在 trace 中标记区段
            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    past_key_values=None,
                    use_cache=True,
                )

    latency_ms = (time.perf_counter() - t0) * 1000

    next_token = sample(outputs.logits[:, -1, :])
    kv_cache = outputs.past_key_values

    cache_list = list(kv_cache)
    num_layers = len(cache_list)
    k_shape = cache_list[0][0].shape
    print(f"  KV Cache: {num_layers} 层，每层 K/V shape = {list(k_shape)}")
    print(f"  seq_len = {k_shape[2]}（等于 prompt 长度）")

    return next_token, kv_cache, latency_ms, prof


# ─────────────────────────────────────────────
# 4. Decode：每次生成 1 个 token
# ─────────────────────────────────────────────

def decode_one_step(model, token, kv_cache, do_profile=False):
    """
    Decode 阶段（单步）：
    - 输入：上一步生成的 1 个 token（shape: [1, 1]）
    - 特点: Memory-bound（大量读 KV Cache，矩阵乘法退化为 矩阵×向量）
    - 输出：下一个 token + 更新后的 KV Cache

    Prefill vs Decode 核心区别：
      Prefill matmul: [seq_len, d_model] × [d_model, d_model]  → 大矩阵，计算密集
      Decode  matmul: [1,       d_model] × [d_model, d_model]  → 向量×矩阵，带宽受限
    """
    t0 = time.perf_counter()

    if do_profile:
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            with_flops=True,
        ) as prof:
            with record_function("decode_forward"):
                with torch.no_grad():
                    outputs = model(
                        input_ids=token,
                        past_key_values=kv_cache,
                        use_cache=True,
                    )
    else:
        prof = None
        with torch.no_grad():
            outputs = model(
                input_ids=token,
                past_key_values=kv_cache,
                use_cache=True,
            )

    latency_ms = (time.perf_counter() - t0) * 1000
    next_token = sample(outputs.logits[:, -1, :])
    return next_token, outputs.past_key_values, latency_ms, prof


# ─────────────────────────────────────────────
# 5. 对比分析：Prefill vs Decode 算子分布
# ─────────────────────────────────────────────

def compare_profiles(prefill_prof, decode_prof, top_n=12):
    """
    对比两个阶段的算子耗时分布。

    关键观察点：
    - aten::mm / aten::addmm：矩阵乘法。Prefill 中占比高（计算密集）；
                               Decode 中占比下降，因输入从 [seq, d] 退化为 [1, d]
    - aten::bmm：批量矩阵乘（Attention QK^T 和 AV）。同上规律。
    - aten::copy_ / aten::cat：KV Cache 的读写/拼接。Decode 中占比上升，
                                因为每步都要把新的 K/V append 到 cache 里。
    - FLOPs 差异：Prefill FLOPs >> Decode FLOPs（seq_len 倍数差距）
    """
    sort_by = "cuda_time_total" if DEVICE == "cuda" else "cpu_time_total"
    time_key = "cuda_time_total" if DEVICE == "cuda" else "cpu_time_total"

    def extract_stats(prof):
        """提取算子名 → (总时间us, 调用次数, FLOPs) 的映射"""
        stats = {}
        for evt in prof.key_averages():
            if not evt.key.startswith("aten::"):
                continue
            stats[evt.key] = {
                "time_us": getattr(evt, time_key, 0),
                "count":   evt.count,
                "flops":   evt.flops if hasattr(evt, "flops") else 0,
            }
        return stats

    p_stats = extract_stats(prefill_prof)
    d_stats = extract_stats(decode_prof)

    # 按 prefill 时间降序，取前 top_n 个算子
    top_ops = sorted(p_stats.keys(), key=lambda k: p_stats[k]["time_us"], reverse=True)[:top_n]

    # ── 原始 profiler 表格 ─────────────────────────────────────────────
    print("\n" + "═" * 70)
    print("  PROFILER 原始输出 — Prefill")
    print("═" * 70)
    print(prefill_prof.key_averages().table(sort_by=sort_by, row_limit=top_n))

    print("\n" + "═" * 70)
    print("  PROFILER 原始输出 — Decode（第 1 步，代表性样本）")
    print("═" * 70)
    print(decode_prof.key_averages().table(sort_by=sort_by, row_limit=top_n))

    # ── 对比表格 ──────────────────────────────────────────────────────
    print("\n" + "═" * 70)
    print("  算子耗时对比：Prefill vs Decode")
    print(f"  排序依据: {time_key}，单位: us")
    print("═" * 70)
    header = f"{'算子':<30} {'Prefill(us)':>12} {'Decode(us)':>12} {'倍数':>8} {'含义'}"
    print(header)
    print("─" * 70)

    # 解释字典：帮助理解每个算子在两个阶段的行为差异
    op_notes = {
        "aten::mm":     "全连接层矩阵乘（Q/K/V proj, FFN）",
        "aten::addmm":  "带偏置的矩阵乘（Linear层）",
        "aten::bmm":    "注意力 QK^T 和 AV 的批量矩阵乘",
        "aten::copy_":  "KV Cache 拼接时的内存拷贝",
        "aten::cat":    "KV Cache 沿 seq 维度拼接",
        "aten::gelu":   "FFN 激活函数",
        "aten::softmax":"Attention softmax",
        "aten::add":    "残差连接加法",
        "aten::layer_norm": "LayerNorm 归一化",
    }

    for op in top_ops:
        p_us = p_stats[op]["time_us"]
        d_us = d_stats.get(op, {}).get("time_us", 0)
        ratio = p_us / d_us if d_us > 0 else float("inf")
        note = op_notes.get(op, "")
        ratio_str = f"{ratio:6.1f}x" if ratio != float("inf") else "  N/A  "
        print(f"{op:<30} {p_us:>12.1f} {d_us:>12.1f} {ratio_str:>8}  {note}")

    # ── FLOPs 对比 ────────────────────────────────────────────────────
    total_prefill_flops = sum(v["flops"] for v in p_stats.values())
    total_decode_flops  = sum(v["flops"] for v in d_stats.values())
    if total_prefill_flops > 0 and total_decode_flops > 0:
        print(f"\n  FLOPs 对比：")
        print(f"    Prefill: {total_prefill_flops/1e9:.3f} GFLOPs")
        print(f"    Decode:  {total_decode_flops/1e9:.3f} GFLOPs")
        print(f"    比值:    {total_prefill_flops/total_decode_flops:.1f}x  "
              f"≈ prompt_len（符合预期：prefill 多处理了 seq_len 倍的数据）")

    # ── 结论 ─────────────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("  核心结论：")
    print("  ┌─ Prefill: Compute-bound")
    print("  │   矩阵乘（mm/addmm/bmm）占主导，GPU 利用率高")
    print("  │   输入 shape: [batch, seq_len, d_model]，seq_len 通常几百")
    print("  │")
    print("  └─ Decode:  Memory-bound")
    print("      矩阵乘退化为向量×矩阵（seq=1），FLOP/byte 比率大幅下降")
    print("      瓶颈转移到 KV Cache 的读取（带宽受限）")
    print("      这就是为什么 decode 延迟比 prefill 低，但 GPU 利用率也低")
    print("─" * 70)


# ─────────────────────────────────────────────
# 6. 完整推理流程
# ─────────────────────────────────────────────

def generate(prompt, max_new_tokens=20):
    print("=" * 55)
    print(f"Prompt: {prompt!r}")
    print("=" * 55)

    # Tokenize prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
    prompt_len = input_ids.shape[1]
    print(f"Prompt 长度: {prompt_len} tokens\n")

    # ── Prefill（带 profiling）────────────────
    print("[ PREFILL ]")
    print(f"  输入: {prompt_len} 个 token 并行处理")

    first_token, kv_cache, prefill_ms, prefill_prof = prefill(model, input_ids)
    first_word = tokenizer.decode(first_token[0])
    print(f"  生成第 1 个新 token: {first_word!r}")
    print(f"  Latency: {prefill_ms:.1f} ms\n")

    # ── Decode ───────────────────────────────
    print("[ DECODE ]")
    generated_tokens = [first_token]
    decode_latencies = []
    decode_prof = None  # 只对第一个 decode step 做 profiling（代表性样本）

    token = first_token
    for step in range(1, max_new_tokens):
        profile_this_step = (step == 1)  # 只 profile 第一步

        token, kv_cache, decode_ms, step_prof = decode_one_step(
            model, token, kv_cache, do_profile=profile_this_step
        )
        if step == 1:
            decode_prof = step_prof  # 保存第一步的 profiler

        generated_tokens.append(token)
        decode_latencies.append(decode_ms)

        seq_len = list(kv_cache)[0][0].shape[2]
        word = tokenizer.decode(token[0])
        profile_tag = " ← profiled" if profile_this_step else ""
        print(f"  Step {step:2d} | token: {word!r:12s} | "
              f"KV seq_len: {seq_len:4d} | latency: {decode_ms:.1f} ms{profile_tag}")

    # ── 汇总 latency ─────────────────────────
    all_new_ids = torch.cat(generated_tokens, dim=1)
    generated_text = tokenizer.decode(all_new_ids[0])

    avg_decode = sum(decode_latencies) / len(decode_latencies)
    print(f"\n{'─'*55}")
    print(f"Prefill latency:     {prefill_ms:.1f} ms  （处理 {prompt_len} tokens）")
    print(f"Decode avg latency:  {avg_decode:.1f} ms  （每 token）")
    print(f"Prefill/Decode 比值: {prefill_ms/avg_decode:.1f}x")
    print(f"\n生成结果: {prompt}{generated_text}")
    print("=" * 55)

    # ── Profiling 对比分析 ────────────────────
    compare_profiles(prefill_prof, decode_prof)

    # ── 导出 Chrome Trace（可用 chrome://tracing 或 Perfetto 查看）──
    prefill_prof.export_chrome_trace("prefill_trace.json")
    decode_prof.export_chrome_trace("decode_trace.json")
    print("\n  Chrome Trace 已导出：")
    print("    prefill_trace.json")
    print("    decode_trace.json")
    print("  打开方式：浏览器访问 chrome://tracing，加载 json 文件")


# ─────────────────────────────────────────────
# 7. 运行
# ─────────────────────────────────────────────

if __name__ == "__main__":
    generate(
        prompt="The future of AI inference is",
        max_new_tokens=20,
    )
