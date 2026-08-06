# Tomorrow's TODO（Windows + RTX 3060）

## 1. 跑 NCU 报告

```cmd
cd f:\AI\flash_attention
nvcc -O2 -arch=sm_86 -lineinfo flash_attn.cu -o flash_attn.exe

ncu --set full ^
    --kernel-name flash_attn_kernel ^
    --kernel-name qk_dot_kernel ^
    --kernel-name softmax_kernel ^
    --kernel-name sv_dot_kernel ^
    -o flash_attn_report ^
    flash_attn.exe

ncu-ui flash_attn_report.ncu-rep
```

在 GUI 里截图并保存到 `flash_attention/ncu_report/`：
- [ ] Memory Throughput（Flash vs Naive 对比）
- [ ] L2 Cache Hit Rate
- [ ] Roofline（Memory Bound vs Compute Bound）
- [ ] Achieved Occupancy

截图存好后补进 `flash_attention/README.md` 的 NCU 那一节。

---

## 2. 填入实测 Benchmark 数字

跑各模块，把结果填进对应 README（现在都是 `—` 占位符）：

**matmul：**
```cmd
cd f:\AI\matmul
nvcc -O2 -arch=sm_86 matrixmul.cu -o matmul.exe
matmul.exe
```
- [ ] 填入 `matmul/README.md` — Naive / Shared Memory 各多少 ms，加速比

**softmax：**
```cmd
cd f:\AI\softmax
nvcc -O2 -arch=sm_86 softmax.cu -o softmax.exe
softmax.exe
```
- [ ] 填入 `softmax/README.md` — Naive / V2 / V3 各多少 ms

**layernorm：**
```cmd
cd f:\AI\layernorm
nvcc -O2 -arch=sm_86 layernorm.cu -o layernorm.exe
layernorm.exe
```
- [ ] 填入 `layernorm/README.md` — GPU LayerNorm 多少 ms

同时把数字也填进根目录 `README.md` 的模块一览表。

---

## 3. 提交 Push

```cmd
git add -A
git commit -m "docs: add benchmark numbers and ncu report"
git push origin main
```

- [ ] Push 成功
