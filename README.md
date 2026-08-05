# CUDA Kernel 从零实现

从零手写 LLM 推理里最核心的几个 CUDA kernel，每个模块都对比 Naive 和优化版本，量化分析加速来源。

**环境：** RTX 3060 12GB / CUDA 12.1 / Windows 11

---

## 模块一览

| 模块 | 优化手段 | 加速比 |
|------|----------|--------|
| [matmul](matmul/) | Shared Memory Tiling | —x |
| [softmax](softmax/) | 内存合并 + 树形归约 + Online Softmax | ~10x |
| [layernorm](layernorm/) | Shared Memory 树形归约 | — |
| [flash_attention](flash_attention/) | Tiling + Online Softmax | 2x vs Naive GPU |

> matmul / layernorm 加速比待补：在 Windows 机器运行后填入。

---

## 学习路线：优化思路是递进的

四个模块不是孤立的，每一个都在前一个的基础上叠加新技巧：

```
matmul
  └─ 核心问题：全局内存重复读取
     解法：Shared Memory Tiling，每块数据只从全局内存读一次

softmax
  └─ 核心问题：单线程串行，内存访问不合并
     解法：多线程并行 + Coalesced Access + 树形归约
     进阶：Online Softmax，max 和 sum 一遍合并

layernorm
  └─ 与 Softmax 结构相同
     核心：两轮树形归约（先算 mean，再算 var）

flash_attention
  └─ 把 matmul tiling + online softmax 合并到一个 kernel
     核心问题：标准 Attention 需要写出 O(N²) 的 attention 矩阵
     解法：tile 流式处理，中间结果留在 shared memory，永不写回全局内存
```

---

## 模块 1：Matrix Multiplication

**问题**：计算 C = A × B（N=1024），每个输出元素需要读 A 的一行和 B 的一列。
Naive 版本每行被重复读取 N 次，全局内存访问量 = 2N³ ≈ 2GB。

**优化**：Shared Memory Tiling。把 A/B 分成 16×16 的小块，由 block 内 256 个 thread 协作加载，每块只从全局内存读一次，被复用 16 次，全局内存读写量降至 2N²。

```
全局内存读取量：
  Naive:         2 × 1024³ ≈ 2147 MB
  Shared Memory: 2 × 1024² ≈    2 MB  （节省 ~1000x，实测 ~Xx）
```

→ [详细说明](matmul/README.md)

---

## 模块 2：Softmax

**问题**：Naive 每 block 只有 1 个 thread，GPU 内存带宽利用率仅 3%（32 个 float 的事务只用 1 个）。

**优化 V2**：256 个 thread 并行，warp 内连续访问内存（利用率 ≈ 100%），shared memory 树形归约找 max/sum（8 步 vs 1024 步串行）。

**优化 V3**：Online Softmax，一遍扫描同时维护 max 和 sum，省掉一轮全局内存读取。

```
内存带宽利用率：
  Naive → V2:  3% → 100%，实测 ~10x 加速

Online Softmax 核心公式：
  m_new = max(m, score)
  l_new = l × exp(m - m_new) + exp(score - m_new)
  扫描一遍即得正确的归一化分母，无需存历史 score
```

→ [详细说明](softmax/README.md)

---

## 模块 3：Layer Normalization

**公式**：`y = (x - μ) / sqrt(σ² + ε) × γ + β`

结构与 Softmax 完全相同，区别在数学：Softmax 归约 max/sum，LayerNorm 归约 mean/var。
同样用 shared memory 树形归约，两轮扫描（先算均值，再算方差），256 线程并行写回。

→ [详细说明](layernorm/README.md)

---

## 模块 4：Flash Attention

**问题**：标准 Attention 需要把 seq×seq 的 S 矩阵写到全局内存，seq=4096 时 64MB，被三个 kernel 反复搬运共 ≈ 256MB。

**优化**：把 QK^T、softmax、SV 三步合并进一个 kernel。Q/K/V 分 tile 流式处理，attention score 只在 shared memory 里存活，永不写回全局内存。用 Online Softmax 的 (m, l, o_acc) 状态在 tile 间累积，最后一次性写回 O。

```
全局内存读写量对比（seq=4096, head_dim=64）：
  Naive GPU：QK^T 写 64MB + softmax 读写 128MB + SV 读 64MB ≈ 256MB
  Flash Attn：Q/K/V 各读一次 + O 写一次                      ≈   8MB
  理论节省：32x，实测 2x（其他瓶颈限制）

seq 越长，S 矩阵是 O(N²)，Q/K/V/O 是 O(N·d)，收益越大：
  seq=1024 → 节省约 9x
  seq=4096 → 节省约 32x
  seq=8192 → 节省约 35x
```

→ [详细说明](flash_attention/README.md)

---

## NCU Profiling（待补）

```cmd
# 编译加 -lineinfo 支持 source correlation
nvcc -O2 -arch=sm_86 -lineinfo flash_attn.cu -o flash_attn.exe

# Profiling
ncu --set full ^
    --kernel-name flash_attn_kernel ^
    --kernel-name qk_dot_kernel ^
    -o flash_attn_report ^
    flash_attn.exe

ncu-ui flash_attn_report.ncu-rep
```

重点关注：Memory Throughput、L2 Hit Rate、Roofline（Memory Bound vs Compute Bound）。
