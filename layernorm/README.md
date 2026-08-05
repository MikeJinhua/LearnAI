# Layer Normalization

从零实现 CUDA LayerNorm，使用 Shared Memory + 树形归约并行计算均值和方差。

## 编译运行

```cmd
cd f:\AI\layernorm
nvcc -O2 -arch=sm_86 layernorm.cu -o layernorm.exe
layernorm.exe
```

## Benchmark 结果

**环境：** RTX 3060 12GB / CUDA 12.1 / M=1024 行 × N=1024 列 / 100 runs avg

| 版本 | 耗时 |
|------|------|
| GPU LayerNorm | — ms |

> 待补：在 Windows 机上运行后填入实测数字。

---

## LayerNorm 是什么

LayerNorm 是 Transformer 里每个 Block 的标配（Attention 和 FFN 后各一个），
作用是对**每一行**（每个 token 的特征向量）独立做归一化，让训练更稳定。

公式分三步：

```
μ  = (1/N) Σ xᵢ                          # 均值
σ² = (1/N) Σ (xᵢ - μ)²                   # 方差
yᵢ = (xᵢ - μ) / sqrt(σ² + ε) × γᵢ + βᵢ  # 归一化 + 仿射变换
```

其中 γ（scale）和 β（shift）是**可学习参数**，ε=1e-5 防止除零。

---

## 实现：两轮归约

LayerNorm 需要算 μ 再算 σ²，σ² 依赖 μ，所以必须串行两轮：

**Round 1：均值**

```
① 256 threads 各算局部 sum（步长循环覆盖 N=1024 个元素）
② shared memory 树形归约 → smem[0] = 全行 sum
③ mean = smem[0] / N
```

**Round 2：方差**

```
① 256 threads 各算局部 Σ(xᵢ - mean)²
② shared memory 树形归约 → smem[0] = 全行方差和
③ var = smem[0] / N
```

**写回**

```
④ 256 threads 并行写回：
   y[i] = (x[i] - mean) / sqrt(var + eps) * gamma[i] + beta[i]
```

全程对全局内存的访问：
- x 读两遍（round 1 + round 2）
- gamma、beta 读一遍
- y 写一遍

共 **4 次** 全局内存扫描，每次 N×4 bytes。

---

## 与 Softmax 的对比

两者结构非常相似，优化策略完全相同：

| | Softmax | LayerNorm |
|--|---------|-----------|
| 归约次数 | 2 次（max、sum） | 2 次（sum、var） |
| 可学习参数 | 无 | 有 γ, β |
| 数据依赖 | sum 依赖 max | var 依赖 mean |
| 优化手段 | smem 树形归约 | smem 树形归约（完全相同） |
| 扫描遍数 | 3 遍（max/sum/write） | 3 遍（sum/var/write） |
| 目的 | 输出概率分布 | 稳定特征分布 |

理解了 Softmax 的并行归约，LayerNorm 直接套用即可，差别只在数学公式。
