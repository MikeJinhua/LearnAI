# Softmax

从零实现 CUDA Softmax，三个版本逐步优化：串行 → 并行归约 → Online Softmax。

## 实现版本

| 版本 | 函数名 | 线程配置 | 核心改进 |
|------|--------|----------|----------|
| Naive | `softmax_naive` | `<<<M, 1>>>` | 单线程串行，逻辑最清晰 |
| V2 | `softmax_v2` | `<<<M, 256>>>` | 256 线程并行 + shared memory 树形归约 |
| V3 | `softmax_v3` | `<<<M, 256>>>` | Online softmax，max 和 sum 一遍合并 |

## 编译运行

```cmd
cd f:\AI\softmax
nvcc -O2 -arch=sm_86 softmax.cu -o softmax.exe
softmax.exe
```

## Benchmark 结果

**环境：** RTX 3060 12GB / CUDA 12.1 / M=1024 行 × N=1024 列 / 100 runs avg

| 版本 | 耗时 | 对比 |
|------|------|------|
| Naive | — ms | baseline |
| V2 | — ms | ~10x vs Naive |
| V3 | — ms | 与 V2 相近 |

> 待补：在 Windows 机上运行后填入实测数字。

---

## Naive → V2：10x 加速从哪来

### 问题：Naive 的内存访问效率极低

Naive 配置是 `<<<M, 1>>>`，每个 block 只有 1 个 thread：

```
单线程串行读: x[0], x[1], x[2], ..., x[1023]
```

GPU 每次内存事务读取 128 bytes（32 个 float），但只有 1 个 thread 用其中 1 个 float：

```
内存带宽利用率 = 1/32 ≈ 3%
```

### 改进：V2 用 256 个 thread，内存合并访问

V2 配置是 `<<<M, 256>>>`，同一个 warp 的 32 个 thread 访问连续地址：

```
Warp 0 (thread 0-31)  同时读: x[0],  x[1],  ..., x[31]   ← 一次事务全用上
Warp 1 (thread 32-63) 同时读: x[32], x[33], ..., x[63]   ← 同上
...
内存带宽利用率 ≈ 100%
```

### 改进：Shared Memory 树形归约

Naive 的 max/sum 是单线程 for 循环，N=1024 步串行。
V2 用 shared memory 做树形归约，只需 log₂(256) = **8 步**完成同样的计算：

```
Step 1: 256 threads 各自算局部 max，存入 smem[tid]
Step 2: 树形归约（stride=128,64,32,...,1），8 轮后 smem[0] = 全行 max
Step 3: 同样归约得到全行 sum
Step 4: 256 threads 并行写回归一化结果
```

### 总结

| 因素 | Naive | V2 |
|------|-------|----|
| 线程数/block | 1 | 256 |
| 内存带宽利用率 | ~3% | ~100% |
| 归约步数 | N=1024 步 | log₂(256)=8 步 |
| Warp 切换空间 | 无 | 8 个 warp 可互换隐藏延迟 |

---

## V3：Online Softmax — 把两遍扫描合并成一遍

V2 需要两遍扫描（第一遍找 max，第二遍算 sum），V3 用 online 算法一遍完成：

```
每处理一个新元素，维护 (local_max, local_sum)：

if val > local_max:
    local_sum = local_sum * exp(local_max - val) + 1.0   # 修正旧 sum 的基准
    local_max = val
else:
    local_sum += exp(val - local_max)
```

**核心思路**：max 更新时，把旧 sum 乘以修正因子 `exp(m_old - m_new)`，
让新旧 sum 始终在同一基准下，不需要单独一遍找 max。

归约阶段同样用修正公式合并两个 (max, sum) 对：

```cuda
if (m_a >= m_b):
    smem_sum[tid] = s_a + s_b * exp(m_b - m_a)
    smem_max[tid] = m_a
else:
    smem_sum[tid] = s_b + s_a * exp(m_a - m_b)
    smem_max[tid] = m_b
```

V3 的全局内存读次数从 V2 的 3 遍减为 2 遍（扫描 + 写回），理论上更省带宽，但实测与 V2 接近，原因是 N=1024 时瓶颈已在 compute 而非 memory。

---

## 与 Flash Attention 的关系

Online Softmax 是 Flash Attention 的核心子模块。
Flash Attention 用同样的 (m, l, o_acc) 状态在 tile 间流式更新，
整个 seq_len 的 softmax 不需要一次性加载到 shared memory。
