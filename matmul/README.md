# Matrix Multiplication

从零实现 CUDA 矩阵乘法，对比 Naive 和 Shared Memory Tiling 两个版本。

## 实现版本

| 版本 | 文件 | 核心思路 |
|------|------|----------|
| Naive | `matrixmul.cu` | 每个 thread 直接从全局内存读 A/B |
| Shared Memory | `matrixmul.cu` | 分块加载到 shared memory，复用数据 |

## 编译运行

```cmd
cd f:\AI\matmul
nvcc -O2 -arch=sm_86 matrixmul.cu -o matmul.exe
matmul.exe
```

## Benchmark 结果

**环境：** RTX 3060 12GB / CUDA 12.1 / N=1024×1024

| 版本 | 耗时 | 对比 |
|------|------|------|
| Naive | — ms | baseline |
| Shared Memory | — ms | —x |

> 待补：在 Windows 机上运行后填入实测数字。

---

## 优化动机：全局内存的重复读取问题

计算 C = A × B，矩阵大小 N×N，每个输出元素 C[i][j] 需要读 A 的第 i 行和 B 的第 j 列（各 N 个元素）：

```
Naive 的问题：
  计算 C[i][0] 读了 A 的第 i 行（N 次）
  计算 C[i][1] 又读了 A 的第 i 行（N 次）
  ...
  A 的每一行被重复读取 N 次，全局内存访问量 = 2N³
```

对 N=1024，全局内存读取量 ≈ **2GB**，而 RTX 3060 显存带宽 360 GB/s，理论下限就要 5ms 以上，且 Naive 还达不到峰值带宽。

## 优化方案：Shared Memory Tiling

核心思路：把 A 和 B 各切成 `TILE_SIZE × TILE_SIZE` 的小块，协作加载到 shared memory，block 内所有 thread 共享这份数据，每个元素只从全局内存读**一次**：

```
Shared Memory 版本：
  每个 block 负责 C 的一个 tile（16×16 = 256 个元素）
  外层循环每轮：
    ① 256 个 thread 协作，把 A 和 B 各一个 tile 加载进 shared memory（1 次全局读）
    ② 256 个 thread 各自用 shared memory 里的数据算点积（0 次全局读）
  
  A 的每块只从全局内存读 1 次，被复用 TILE_SIZE 次
  全局内存读取量 = 2N²（比 Naive 少 N 倍）
```

```
TILE_SIZE=16, N=1024 时，全局内存读取量：
  Naive:         2 × 1024³ ≈ 2147 MB
  Shared Memory: 2 × 1024² ≈    2 MB   （节省 ~1000x）
```

实际加速比低于理论，因为 shared memory 本身也有 bank conflict、同步开销等，但数量级上的改善是真实的。

## 关键代码解读

```cuda
// 每轮 tile，256 个 thread 协作加载
tileA[threadIdx.y][threadIdx.x] = A[row * n + t * TILE_SIZE + threadIdx.x];
tileB[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * n + col];
__syncthreads();  // 确保全部加载完再算

// 用 shared memory 里的数据做内积，不访问全局内存
for (int k = 0; k < TILE_SIZE; k++) {
    sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
}
__syncthreads();  // 下一轮 tile 开始前同步
```

## 与后续模块的关系

Matmul 的 tiling 思路直接延伸到 Flash Attention：
- Matmul tiling：把 A/B 分块放 shared memory，避免重复读全局内存
- Flash Attention tiling：把 Q/K/V 分块放 shared memory，避免写出 N² 的 attention 矩阵
