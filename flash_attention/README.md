# Flash Attention

从零实现 Flash Attention，对比 CPU / Naive GPU / Flash Attention 三种方式的性能。

## 实现思路

普通 Attention 需要把整张 `[seq_len, seq_len]` 的 attention 矩阵存到显存：

```
seq=4096, head_dim=64 → 4096 × 4096 × 4 bytes = 64MB（每个 head）
```

Flash Attention 把 Q/K/V 切成小 tile，在 shared memory 里做 online softmax，
始终只用 `O(tile_size)` 的显存，不需要存整张 attention matrix。

## 文件

| 文件 | 说明 |
|---|---|
| `flash_attn.cu` | CPU 参考实现 + Naive GPU（3 kernel）+ Flash Attention kernel |
| `CMakeLists.txt` | CMake 构建配置（sm_86, RTX 3060） |

## 编译运行

```cmd
# 在 x64 Native Tools Command Prompt for VS 2022 里运行
cd f:\AI\flash_attention
nvcc -O2 -arch=sm_86 flash_attn.cu -o flash_attn.exe
flash_attn.exe
```

## Benchmark 结果

**环境：** RTX 3060 12GB / CUDA 13.2 / seq=4096 / head_dim=64 / 100 runs avg

| 方案 | 正确性 | 平均耗时 | 对比 |
|---|---|---|---|
| Naive GPU | PASS | 20.4545 ms | baseline |
| Flash Attn V1 | PASS | 10.4941 ms | **1.95x vs Naive GPU** |
| Flash Attn V2 | PASS | 5.7011 ms | **1.84x vs V1；3.59x vs Naive GPU** |

V2 使用 8 个 warp/block，每个 warp 协作处理 Q 行：32 个 lane 并行完成 64 维 dot product，
并分别持有两个输出维度。它仍让每个 block 处理 32 行 Q，从而保持 V1 的 K/V tile 跨行复用粒度。
编译器资源统计从 V1 的 96 registers/thread 降到 V2 的 40 registers/thread，shared memory
也从 24 KB/block 降到 16 KB/block（RTX 3060，FP32，seq=4096，head_dim=64，100 runs）。
从 V1 的 10.4941 ms 降到 V2 的 5.7011 ms，等价于 **1.84x 加速**，或 **耗时降低 45.7%**。

---

## 为什么 Flash Attention 省带宽：量化分析

### Naive GPU 的问题：三次全局内存往返

Naive 实现拆成三个 kernel，中间结果 S（attention score 矩阵）必须写到全局内存再读回来：

```
Kernel 1: QK^T 写出 S        → 全局内存写  64MB
Kernel 2: softmax 读 S、写 S  → 全局内存读  64MB + 写 64MB
Kernel 3: SV 读 S             → 全局内存读  64MB
────────────────────────────────────────────────
全局内存总读写                    ≈ 256MB（每个 head，每次 forward）
```

S 矩阵本身不是最终结果，它只是中间变量。但因为三个 kernel 之间没有办法共享片上内存，
这 64MB 被反复搬运了三次。

### Flash Attention 的方案：tile 留在 shared memory

Flash Attention 把外层循环（遍历所有 K/V tile）全部放在**同一个 kernel** 里完成。
S 的每一小块（tile_size × tile_size）算完就用掉，**从不写回全局内存**：

```
Q 读一次                       → 全局内存读   2MB   （4096×64×4B）
K 读一次                       → 全局内存读   2MB
V 读一次                       → 全局内存读   2MB
O 写一次                       → 全局内存写   2MB
────────────────────────────────────────────────
全局内存总读写                    ≈ 8MB（每个 head，每次 forward）
```

带宽节省倍数：**256MB ÷ 8MB = 32x**（理论上限）。
实测 2x 加速，说明还有其他瓶颈（compute、kernel launch overhead、shared memory bank conflict 等），
但带宽节省的方向是正确的，seq 越长效果越明显。

> 直觉：S 矩阵大小是 O(N²)，Q/K/V/O 是 O(N·d)。当 N 很大时，
> 省掉 O(N²) 的全局内存读写是 Flash Attention 的核心价值。

### 为什么 seq 越长收益越大

| seq_len | S 矩阵大小 | Q/K/V/O 合计 | 理论带宽节省 |
|---------|-----------|-------------|------------|
| 512     | 1 MB      | 0.25 MB     | 5x         |
| 1024    | 4 MB      | 0.5 MB      | 9x         |
| 4096    | 64 MB     | 8 MB        | **32x**    |
| 8192    | 256 MB    | 32 MB       | **35x**    |

---

## Online Softmax 核心公式

传统 softmax 需要两遍扫描（先找 max，再算 sum）。
Online softmax 只需**一遍**，每处理一个新 score 就地更新状态：

```
# 状态：m = 当前最大值，l = 当前归一化分母，o_acc = 当前输出累积

m_new = max(m, score)
l_new = l * exp(m - m_new) + exp(score - m_new)
o_acc = o_acc * exp(m - m_new) + v * exp(score - m_new)

# 处理完所有 K/V tile 后，写回：
O = o_acc / l
```

**为什么这样可以？**

设处理到第 j 步时，已经见过 score₁, score₂, ..., scoreⱼ。
- `m` 维护 max(score₁..scoreⱼ)
- `l` 维护 Σ exp(scoreₖ - m)，即以当前 m 为基准的归一化分母
- 当新来一个 scoreⱼ₊₁，如果 m 更新了，之前的 `l` 和 `o_acc` 都乘以 `exp(m_old - m_new)` 来修正基准

这样 m、l、o_acc 始终处于一致的基准下，最终 `o_acc / l` 等价于标准 softmax 的结果，
但**中间不需要存任何历史 score**，整个过程 O(1) 额外空间（只有寄存器里的三个标量）。

---

## NCU Profiling（性能分析完成）

> **数据更正：** 下方旧截图和 `99.73% Memory Throughput / 22.55% Compute Throughput /
> 90.82% Occupancy` 实际选中的是 naive 路径的 `qk_dot_kernel`，不能归因于
> `flash_attn_kernel`。真正的 V1 Flash kernel 实测为：Compute Throughput 10.36%、
> Memory Throughput 53.20%、L1/TEX Throughput 75.60%、L2 Throughput 0.86%、
> DRAM Throughput 0.20%、Theoretical Occupancy 8.33%、Achieved Occupancy 7.25%，
> 且 Scheduler 的 No Eligible 为 88.22%、Short Scoreboard 约占指令间隔的 66%。
> 因此 V1 是低并行度造成的 latency-bound，不是 DRAM bandwidth-bound。
> V1/V2 的正确单次采集方法见 `RUN_NCU_PROFILING.md`；下方旧结论仅保留作错误数据选择案例，
> 不应继续作为 Flash kernel 的性能结论引用。

已完成基准测试和理论分析。关键指标推导如下：

### 内存优化量化

**Naive GPU（三个kernel）:**
```
全局内存读写总量 ≈ 256 MB
- QK^T kernel: 写 S 矩阵 64MB
- Softmax kernel: 读 S 64MB + 写 S 64MB  
- SV kernel: 读 S 64MB
- Q/K/V 读: 6MB
```

**Flash Attention（单个kernel）:**
```
全局内存读写总量 ≈ 8 MB
- Q 读 2MB + K 读 2MB + V 读 2MB + O 写 2MB
```

### 实测性能数据

| 指标 | Naive GPU | Flash V1 | Flash V2 |
|------|---:|---:|---:|
| 执行时间 | 20.4545 ms | 10.4941 ms | 5.7011 ms |
| 相对 V1 | — | baseline | **1.84x** |
| 相对 Naive | baseline | 1.95x | **3.59x** |
| 正确性 | PASS | PASS | PASS |

### V1 瓶颈与 V2 修改的证据链

| NCU 指标 | V1 | V2 | 含义 |
|------|---:|---:|------|
| Compute Throughput | 10.56% | 74.41% | V2 更能持续发射计算指令 |
| DRAM Throughput | 0.43% | 3.63% | V1 不是 DRAM 带宽打满 |
| Theoretical Occupancy | 8.33% | 83.33% | V1 受 block 并行度和资源限制 |
| Achieved Occupancy | 7.27% | 60.62% | V2 有更多 active warp 隐藏延迟 |
| Eligible Warps / Scheduler | 0.12 | 2.35 | V2 调度器更少无事可做 |
| No Eligible | 88.22% | 30.19% | V1 的关键 latency-bound 证据 |
| Registers / Thread | 96 | 40 | V2 降低每线程状态 |
| Dynamic Shared Memory / Block | 24.58KB | 16.38KB | V2 不再把 Q tile 放入 shared memory |

V1 每 block 只有 32 threads，即一个 warp；每线程串行计算一整行 dot product，并保留 64 维输出累加器。V2 改为 256 threads/block，由一个 warp 协作处理 Q 行，用 `__shfl_down_sync` 归约 dot product，每个 lane 只保留两个 Q/O 维度。因此这不是“凭经验调 block size”，而是直接针对 NCU 显示的低 occupancy 和缺少 eligible warp。

> `99.73% Memory Throughput / 90.82% Occupancy` 是 naive `qk_dot_kernel` 的指标，不是 Flash V1/V2 的指标。

### 如何查看完整的 NCU 报告

报告文件：`ncu_report/flash_attn_report.ncu-rep`（8.1GB，包含详细性能计数器）

**快速查看：**
```cmd
cd F:\AI.worktrees\todo-file-review\flash_attention
ncu --import ncu_report\flash_attn_report.ncu-rep --page details
```

**GUI 查看（推荐）：**
```cmd
ncu-ui ncu_report\flash_attn_report.ncu-rep
```

相关截图已保存到 `ncu_report/`：
- `L2_Cache_Hit_Rate.png` - 内存层级分析
- `Roofline_Analysis.png` - 性能特征与瓶颈
- `Achieved_Occupancy.png` - SM 占用率分析

---

**性能分析状态：** ✓ Benchmark 完成 | ✓ NCU Profiling 完成 | ✓ 性能诊断完成
