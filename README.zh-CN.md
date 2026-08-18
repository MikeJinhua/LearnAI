# CUDA Kernel 从零实现

从零手写 LLM 推理里最核心的几个 CUDA kernel，每个模块都对比 Naive 和优化版本，量化分析加速来源。

**环境：** RTX 3060 12GB / CUDA 13.2 / Windows 11

**说明：** 除特别注明外，所有性能数据均为 FP32，使用 `cudaEvent` 记录 kernel 执行时间；其中大多数 benchmark 取 100 runs 的平均值。未在当前仓库中保留的对照项会明确标注为“未测/未保留”，不做推测。

---

## 模块一览

| 模块 | 优化手段 | 实测结果 |
|------|----------|----------|
| [matmul](matmul/) | Shared Memory Tiling | 3.70 ms → 2.08 ms，1.78x（N=1024，FP32，single-run 代码基准） |
| [softmax](softmax/) | 内存合并 + 树形归约 + Online Softmax | Naive 0.415 ms，V2 0.036 ms，V3 0.037 ms（FP32，100 runs avg） |
| [layernorm](layernorm/) | Shared Memory 树形归约 | GPU LayerNorm 0.038 ms（FP32，100 runs avg；未保留原始 CPU baseline） |
| [flash_attention](flash_attention/) | Tiling + Online Softmax | CPU 873.34 ms / Naive 21.10 ms / Flash 10.70 ms，Flash vs Naive = 2.0x（FP32，seq=4096, head_dim=64，100 runs avg） |

> 这份 README 采用最稳妥的主数据：有代码和 benchmark 说明支撑的版本；不把不同测试条件混在同一条结论里。

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
  Shared Memory: 2 × 1024² ≈    2 MB
```

实际 benchmark（代码中保留的主版本）是：

```
Naive = 3.70 ms
Shared = 2.08 ms
Speedup = 1.78x
```

顺便算一笔：

```
2 × 1024³ / 2.08 ms ≈ 1.04 TFLOPS
RTX 3060 FP32 峰值 ≈ 12.7 TFLOPS
1.04 / 12.7 ≈ 8%
```

这说明在这个实现里，瓶颈并不在“算力到顶”，而在 shared memory 的访存压力、tile 粒度和 bank conflict。它和 Flash Attention 的瓶颈来源是同一类问题：吞吐被片上访存管线卡住，而不是纯粹的峰值算力上限。

**下一步优化方向**：register blocking（每个 thread 计算更大块输出、累加器留在寄存器）、float4 向量化、消除 bank conflict、再往上才考虑 Tensor Core / `ldmatrix` 路径。

→ [详细说明](matmul/README.md)

---

## 模块 2：Softmax

**问题**：Naive 每 block 只有 1 个 thread，warp 内连续访问不充分，内存带宽利用率极低。

**优化 V2**：256 个 thread 并行，warp 内连续访问内存，shared memory 树形归约找 max/sum（8 步 vs 1024 步串行）。

**优化 V3**：Online Softmax，一遍扫描同时维护 max 和 sum，省掉一轮全局内存读取。

```
实际结果：
  Naive: 0.415 ms
  V2:    0.036 ms
  V3:    0.037 ms
  加速比：约 11.5x（V2 vs Naive）

Online Softmax 核心公式：
  m_new = max(m, score)
  l_new = l × exp(m - m_new) + exp(score - m_new)
  扫描一遍即得正确的归一化分母，无需存历史 score
```

这里要讲清楚：在这份单独的 softmax benchmark 里，V3 比 V2 稍慢，原因是“省掉一轮全局读”被“逐元素 rescale + 分支 + 修正因子”抵消了。也就是说，**online softmax 的真正价值不在单次 softmax 小问题里，而在 Flash Attention 这种 tile-based 流式处理场景里，它是必须品**。在这里它不是一个 standalone 的“最优”版本，而是为后续 fused attention 提供了关键状态更新机制。

→ [详细说明](softmax/README.md)

---

## 模块 3：Layer Normalization

**公式**：`y = (x - μ) / sqrt(σ² + ε) × γ + β`

结构与 Softmax 完全相同，区别在数学：Softmax 归约 max/sum，LayerNorm 归约 mean/var。
同样用 shared memory 树形归约，两轮扫描（先算均值，再算方差），256 线程并行写回。

**实测：** GPU LayerNorm 约 0.038 ms（FP32，100 runs avg）；这里给出的是优化后延迟，不是加速比。

**缺失项：** 当前仓库里未保留对应的 CPU baseline 或 Naive GPU baseline 记录，因此不把“加速比”写得比实际更大。若后续补齐 baseline，就可以补到表格里。

→ [详细说明](layernorm/README.md)

---

## 模块 4：Flash Attention

**问题**：标准 Attention 需要把 seq×seq 的 S 矩阵写到全局内存，seq=4096 时 64MB，被三个 kernel 反复搬运共 ≈ 256MB。

**优化**：把 QK^T、softmax、SV 三步合并进一个 kernel。Q/K/V 分 tile 流式处理，attention score 只在 shared memory 里存活，永不写回全局内存。用 Online Softmax 的 (m, l, o_acc) 状态在 tile 间累积，最后一次性写回 O。

```
实测结果（seq=4096, head_dim=64，FP32，100 runs avg）：
  CPU:        873.34 ms
  Naive GPU:  21.10 ms
  Flash Attn: 10.70 ms
  Flash vs Naive: 2.0x
```

理论上，Flash Attention 可以把全局内存流量从 256 MB 降到约 8 MB，约 32x 节省；但在 RTX 3060 上，实际收益只有 2.0x，说明瓶颈已经从“DRAM 中间矩阵搬运”转移到**片上访存管线 / shared memory / LSU 访问模式**。这里不是“失败”，而是**瓶颈上移了一层**：你把大量流量从全局显存搬到了 tile 内的 shared memory 与寄存器重用路径；在这个规模下，片上访存本身已经成为限制因素。

这也解释了：理论上 32x 节省，实测只 2x，原因不在于优化思路错了，而在于它从“全局内存带宽问题”变成了“片上访存/共享内存效率问题”。

→ [详细说明](flash_attention/README.md)

---

## NCU Profiling（已保留真实报告）

已在 Windows + RTX 3060 / CUDA 13.2 环境下实际生成 Nsight Compute 报告，并保留了关键截图与分析结论。这里使用的是已验证的事实：**没有把 missing 值编成结论**。

```cmd
# 编译加 -lineinfo 支持 source correlation
nvcc -O2 -arch=sm_86 -lineinfo flash_attn.cu -o flash_attn.exe

# Profiling：同时观察 Flash Attn 与 Naive 的关键 kernel
ncu --set full ^
    --kernel-name flash_attn_kernel ^
    --kernel-name qk_dot_kernel ^
    --kernel-name softmax_kernel ^
    --kernel-name sv_dot_kernel ^
    -o flash_attn_report ^
    flash_attn.exe

ncu-ui flash_attn_report.ncu-rep
```

### 关键 NCU 指标（Flash Attention kernel）

| 指标 | 结果 |
|------|------|
| L2 Cache Hit Rate | 95.15% |
| Achieved Occupancy | 90.82% |
| Memory Busy | 99.73% |
| Compute Throughput | 22.55% |

这四个数字的真实含义是：

- **Memory Busy = 99.73%**，**Compute Throughput = 22.55%**，这是一条典型的 **memory-bound** 信号。
- **L2 Hit Rate = 95.15%**，说明这些访存没有打到 DRAM，而是被扔进了更高层次的片上缓存/共享内存路径。

所以，这个项目最关键的结论不是“Flash Attention 仍然是 pure DRAM memory-bound”，而是：

> **Flash Attention 没有消除访存瓶颈，而是把它从 DRAM 搬到了片上访存管线。**

也就是说，理论上的 32x 节省只计算了全局内存流量，却没有把 tile 内部大量的 shared memory load/store 与寄存器重用计入；所以 32x → 2.0x 的落差，是“瓶颈上移”而不是“优化无效”。

**这个判断有数据支撑：**

- 99.73% 的 Memory Busy 证明“访存是主瓶颈”
- 95.15% 的 L2 命中证明“这不是 DRAM 这层的内存吸收问题，而是片上访存路径的吞吐限制”
- 90.82% 的 Achieved Occupancy 说明 occupancy 不是主限制，而是 shared memory / LSU / register reuse 才是关键
- 22.55% 的 Compute Throughput 说明 ALU 没有被高峰值占满；即使算力存在剩余，程序仍被访存管线卡住

这里的结论非常明确：**不是 compute-bound，也不是“看起来像 memory-bound 就都一样”**，而是“瓶颈从 DRAM 转移到片上访存管线”。这个结论比“理论 roofline 说法更符合你手上的 NCU 实测”。

### 需要说明的缺失项

- **Naive kernel 的 NCU 对照值**：当前保留的报告中，最完整的指标是 flash kernel；没有找到一份同一时刻的 naive + flash 并排 NCU 表格，因此不能编造 naive 对照值。
- **cuBLAS / PyTorch SDPA baseline**：当前工程中未跑，故在 Limitations 中记录为“未测/未加入”，不伪造对照。
- **LayerNorm baseline**：当前仓库里未保留稳定的 CPU 或 naive GPU baseline，因此不在公开结果中误写“加速比”。

### 性能结论（总结）

1. 对于 Flash Attention，**收益真实存在**：CPU 873.34 ms → Naive 21.10 ms → Flash 10.70 ms，约 2.0x。  
2. 对于 NCU 来看，**瓶颈不是 compute-bound，而是从 DRAM 转移到了片上访存路径**。  
3. 我们已在仓库中保留关键 NCU 证据与截图，且不在公开结论中扩展到无根据的定性描述。  
4. 缺失项直接写“未测 / 未保留”，这比编造更可信，也更适合 GitHub 和简历展示。

---

## 进一步优化方向

下一步真正值得做的优化，不是再去“写一个更炫的结论”，而是继续压缩 tile 内部的 shared memory 访问次数：

- register blocking：每个 thread 处理更大的一小块输出，累加器留在寄存器而不是反复写回 smem
- float4 向量化：减少每次访存的指令数，提升 memory coalescing 效率
- bank conflict 消除：调节 shared memory 访问布局，减少 warp 内串行等待
- 更高阶路径：Tensor Core / `ldmatrix` 族操作，可进一步减少 smem 相关指令数

这也和 flash attention 与 matmul 共同撞到的那堵墙是同一类问题：**不是“算不出来”，而是“访存路径和 tile 组织方式还没压到最优”**。
