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

**环境：** RTX 3060 12GB / CUDA 12.1 / seq=4096 / head_dim=64 / 100 runs avg

| 方案 | 正确性 | 平均耗时 | 对比 |
|---|---|---|---|
| CPU | — | 1520 ms | baseline |
| Naive GPU | PASS | 22.5 ms | 67.6x vs CPU |
| Flash Attn | PASS | 11.4 ms | **2.0x vs Naive GPU** |

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

## NCU Profiling（TODO：明天在 Windows 机上补）

```cmd
# 编译时加 -lineinfo 以支持 source correlation
nvcc -O2 -arch=sm_86 -lineinfo flash_attn.cu -o flash_attn.exe

# 对 flash_attn_kernel 和 naive 三个 kernel 分别 profiling
ncu --set full ^
    --kernel-name flash_attn_kernel ^
    --kernel-name qk_dot_kernel ^
    --kernel-name softmax_kernel ^
    --kernel-name sv_dot_kernel ^
    -o flash_attn_report ^
    flash_attn.exe

# 用 GUI 打开报告
ncu-ui flash_attn_report.ncu-rep
```

重点关注的指标：

| 指标 | 预期 Flash vs Naive |
|------|-------------------|
| Memory Throughput (GB/s) | Flash 更低（省了大 buffer 搬运） |
| L2 Cache Hit Rate | Flash 更高（tile 在 SRAM 里复用） |
| Compute Bound vs Memory Bound | Naive 更 memory bound |
| Achieved Occupancy | 看 shared memory 是否限制了 occupancy |
