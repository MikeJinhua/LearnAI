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

seq=4096 时 attention 矩阵 = 64MB，Naive GPU 反复读写显存，
Flash Attention 通过 tiling 把中间结果留在 shared memory，显存带宽压力小，赢 2x。

## Online Softmax 核心公式

每处理一个新的 score，不需要存历史值，只维护两个标量：

```
m_new = max(m, score)
l_new = l * exp(m - m_new) + exp(score - m_new)
o_acc = o_acc * exp(m - m_new) + v * exp(score - m_new)

# 最终写回：
O = o_acc / l
```
