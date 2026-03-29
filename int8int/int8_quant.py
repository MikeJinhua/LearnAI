import torch
import time
from transformers import GPT2LMHeadModel, GPT2Tokenizer

print("加载模型...")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model_fp32 = GPT2LMHeadModel.from_pretrained("gpt2")
model_fp32.eval()

# ─────────────────────────────────────────────
# 1. FP32 推理 latency
# ─────────────────────────────────────────────
prompt = "The future of AI inference is"
input_ids = tokenizer.encode(prompt, return_tensors="pt")

# 预热
with torch.no_grad():
    model_fp32(input_ids)

# 计时
times = []
for _ in range(20):
    t0 = time.perf_counter()
    with torch.no_grad():
        model_fp32(input_ids)
    times.append((time.perf_counter() - t0) * 1000)

fp32_ms = sum(times) / len(times)
print(f"FP32 latency: {fp32_ms:.2f} ms")

# ─────────────────────────────────────────────
# 2. INT8 量化
# ─────────────────────────────────────────────
model_int8 = GPT2LMHeadModel.from_pretrained("gpt2")
model_int8.eval()

model_int8 = torch.quantization.quantize_dynamic(
    model_int8,
    {torch.nn.Linear},  # 只量化 Linear 层
    dtype=torch.qint8
)

print("INT8 量化完成")

# ─────────────────────────────────────────────
# 3. INT8 推理 latency
# ─────────────────────────────────────────────
# 预热
with torch.no_grad():
    model_int8(input_ids)

times = []
for _ in range(20):
    t0 = time.perf_counter()
    with torch.no_grad():
        model_int8(input_ids)
    times.append((time.perf_counter() - t0) * 1000)

int8_ms = sum(times) / len(times)
print(f"INT8 latency: {int8_ms:.2f} ms")

# ─────────────────────────────────────────────
# 4. 对比
# ─────────────────────────────────────────────
print(f"\n结果对比：")
print(f"FP32: {fp32_ms:.2f} ms")
print(f"INT8: {int8_ms:.2f} ms")
print(f"加速比: {fp32_ms/int8_ms:.2f}x")

# 模型大小对比
import os
torch.save(model_fp32.state_dict(), "fp32_model.pt")
torch.save(model_int8.state_dict(), "int8_model.pt")
fp32_size = os.path.getsize("fp32_model.pt") / 1024 / 1024
int8_size = os.path.getsize("int8_model.pt") / 1024 / 1024
print(f"\n模型大小：")
print(f"FP32: {fp32_size:.1f} MB")
print(f"INT8: {int8_size:.1f} MB")
print(f"压缩比: {fp32_size/int8_size:.2f}x")

# 清理临时文件
os.remove("fp32_model.pt")
os.remove("int8_model.pt")