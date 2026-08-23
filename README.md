# CUDA Kernel from Scratch

This project implements several core CUDA kernels used in LLM inference from scratch and analyzes the bottlenecks and speedups of naive versus optimized versions.

**Environment:** RTX 3060 12GB / CUDA 13.2 / Windows 11

**Note:** Unless otherwise specified, all benchmark numbers below are for FP32 kernels measured with `cudaEvent`; most timings are averaged over 100 runs. If a comparison baseline was not preserved in the repo, it is explicitly marked as "not measured / not retained" instead of being guessed.

---

## Module Overview

| Module | Optimization | Measured result |
|--------|-------------|-----------------|
| [matmul](matmul/) | Shared Memory Tiling | 3.70 ms → 2.08 ms, 1.78x (N=1024, FP32, single-run code baseline) |
| [softmax](softmax/) | Memory coalescing + tree reduction + Online Softmax | Naive 0.415 ms, V2 0.036 ms, V3 0.037 ms (FP32, 100 runs avg) |
| [layernorm](layernorm/) | Shared Memory tree reduction | GPU LayerNorm 0.038 ms (FP32, 100 runs avg; original CPU baseline not retained) |
| [flash_attention](flash_attention/) | Warp collaboration + Online Softmax | V1 10.4941 ms → V2 5.7011 ms, **1.84x speedup / 45.7% lower latency** (FP32, seq=4096, head_dim=64, 100 runs avg) |

> This README uses the most defensible source data: the numbers backed by code and benchmark logs in the repo, without mixing different test conditions into one claim.

---

## Learning Path: The Optimizations Build on Each Other

The four modules are not independent; each one adds a new idea on top of the previous one:

```
matmul
  └─ Core problem: repeated global memory reads
     Solution: Shared Memory Tiling, so each tile is loaded from global memory once

softmax
  └─ Core problem: single-thread serial reduction and poor memory coalescing
     Solution: parallel threads + coalesced access + tree reduction
     Advanced step: Online Softmax, merging max and sum in one pass

layernorm
  └─ Similar to softmax
     Core idea: two tree reductions (mean then variance)

flash_attention
  └─ Combines matmul tiling and online softmax into one kernel
     Core problem: standard attention requires storing the O(N²) score matrix
     Solution: tile-by-tile streaming with intermediate values kept in shared memory
```

---

## Module 1: Matrix Multiplication

**Problem:** computing C = A × B for N=1024 means each output element reads one row from A and one column from B. The naive version repeatedly re-reads the same row/column data, causing global memory traffic of roughly 2N³ ≈ 2GB.

**Optimization:** Shared Memory Tiling. Split A and B into 16×16 tiles, have 256 threads inside a block collaboratively load the tile into shared memory, and reuse that data across the block before moving to the next tile.

```
Global memory traffic:
  Naive:         2 × 1024³ ≈ 2147 MB
  Shared Memory: 2 × 1024² ≈    2 MB
```

The measured benchmark from the main implementation in the repo is:

```
Naive = 3.70 ms
Shared = 2.08 ms
Speedup = 1.78x
```

A quick estimate:

```
2 × 1024³ / 2.08 ms ≈ 1.04 TFLOPS
RTX 3060 FP32 peak ≈ 12.7 TFLOPS
1.04 / 12.7 ≈ 8%
```

This shows the bottleneck is not simply “the kernel is compute-bound.” It is dominated by shared-memory traffic, tile granularity, and bank conflicts. This is the same bottleneck family seen later in Flash Attention: throughput is limited by the on-chip memory pipeline rather than the raw FP32 peak.

**Next-step optimizations:** register blocking (each thread computes a larger chunk and keeps accumulators in registers), float4 vectorized loads, bank-conflict elimination, and then tensor-core / `ldmatrix`-style paths once the memory path is improved.

→ [Detailed notes](matmul/README.md)

---

## Module 2: Softmax

**Problem:** in the naive version, each block has only one thread, so memory requests are poorly coalesced and bandwidth utilization is very low.

**V2 optimization:** use 256 threads per block, coalesced memory access, and tree reduction in shared memory to find max and sum more efficiently.

**V3 optimization:** Online Softmax keeps the running max and sum in one pass and avoids an extra full pass over the input.

```
Measured result:
  Naive: 0.415 ms
  V2:    0.036 ms
  V3:    0.037 ms
  Speedup: about 11.5x (V2 vs Naive)

Online Softmax update:
  m_new = max(m, score)
  l_new = l × exp(m - m_new) + exp(score - m_new)
  This lets the normalized denominator be computed in a single streaming pass without storing the full historical score matrix.
```

One important nuance: in this standalone softmax benchmark, V3 is slightly slower than V2 because the savings from removing one global read are partially offset by per-element rescaling and branch overhead. In other words, the real value of online softmax is not that it is always a standalone winner in a tiny kernel; its true strength shows up in Flash Attention, where tile-by-tile streaming is required.

→ [Detailed notes](softmax/README.md)

---

## Module 3: Layer Normalization

**Formula:** `y = (x - μ) / sqrt(σ² + ε) × γ + β`

The structure is similar to softmax. The difference is that softmax reduces max/sum, while LayerNorm reduces mean/variance. The implementation uses shared-memory tree reduction: two passes, first for mean and then for variance, followed by a final writeback.

**Measured:** GPU LayerNorm is around 0.038 ms (FP32, 100 runs avg). This is the optimized latency, not a speedup claim.

**Missing baseline:** the repo does not retain a stable CPU or naive-GPU baseline for LayerNorm, so this section intentionally avoids overstating a speedup ratio.

→ [Detailed notes](layernorm/README.md)

---

## Module 4: Flash Attention

**Problem:** standard attention writes the full seq×seq score matrix to global memory. For seq=4096, the score matrix alone is about 64MB per head, and the naive 3-kernel flow pushes roughly 256MB of global traffic.

**Optimization:** fuse QK^T, softmax, and SV into one kernel. Q/K/V are streamed in tiles, and the attention score is kept in shared memory instead of being written back to global memory. Online softmax tracks `(m, l, o_acc)` across tiles, then writes the final output once.

```
Measured results (seq=4096, head_dim=64, FP32, 100 runs avg):
  Naive GPU:     20.4545 ms
  Flash Attn V1: 10.4941 ms
  Flash Attn V2:  5.7011 ms
  V2 vs V1:       1.84x (45.7% lower latency)
  V2 vs Naive:    3.59x
```

V1 assigns one complete query row to one thread. V2 instead uses one warp to collaborate on a row, launches eight warps per block, and keeps only two output dimensions per lane. Registers fall from 96 to 40 per thread and dynamic shared memory from 24KB to 16KB per block. This raises achieved occupancy from 7.27% to 60.62% and removes much of V1's latency-hiding problem.

This is not a failure of the optimization idea; it is a shift in the actual bottleneck.

→ [Detailed notes](flash_attention/README.md)

---

## NCU Profiling (real report retained)

The Nsight Compute report was generated on Windows + RTX 3060 + CUDA 13.2, and the key screenshots/report files were retained in the repo. The important point is that we only keep facts that are supported by the profiling data and do not invent missing values.

```cmd
# compile with line-info support
nvcc -O2 -arch=sm_86 -lineinfo flash_attn.cu -o flash_attn.exe

# collect V1 and V2 separately; skip the warmup launch
ncu --set full --kernel-name regex:flash_attn_kernel --launch-skip 1 --launch-count 1 ^
    -o ncu_report\flash_v1 flash_attn.exe --profile-v1
ncu --set full --kernel-name regex:flash_attn_v2_kernel --launch-skip 1 --launch-count 1 ^
    -o ncu_report\flash_v2 flash_attn.exe --profile-v2

ncu-ui flash_attn_report.ncu-rep
```

### Key NCU metrics (V1 versus V2)

| Metric | V1 | V2 |
|--------|---:|---:|
| NCU duration | 11.05 ms | 6.90 ms |
| Compute throughput | 10.56% | 74.41% |
| DRAM throughput | 0.43% | 3.63% |
| Theoretical occupancy | 8.33% | 83.33% |
| Achieved occupancy | 7.27% | 60.62% |
| Eligible warps / scheduler | 0.12 | 2.35 |
| No eligible | 88.22% | 30.19% |
| Registers / thread | 96 | 40 |
| Dynamic shared memory / block | 24.58KB | 16.38KB |

V1 is latency-bound because it exposes only one warp per block and has high register/shared-memory usage. With almost no other eligible warp available, the scheduler cannot hide dependency and shared-memory latency. V2 directly addresses that evidence: eight warps collaborate within each block, warp shuffles parallelize the dot product, and lower per-thread state allows far higher occupancy.

The old `99.73% Memory Throughput / 90.82% Occupancy` values belong to the naive `qk_dot_kernel`; they must not be attributed to either Flash kernel.

### Important limitations

- **Naive-kernel NCU comparison:** the retained report contains the flash-kernel metrics, but not a clean naive-vs-flash side-by-side NCU table. We do not invent that missing comparison.
- **cuBLAS / PyTorch SDPA baseline:** not measured in this repo, so it is documented as “not measured / not included.”
- **LayerNorm baseline:** the repo does not retain a stable CPU or naive-GPU baseline for LayerNorm, so the public summary avoids overstating a speedup claim.

### Overall takeaway

1. Flash Attention V2 reduces latency from 10.4941 ms to 5.7011 ms: 1.84x faster, or 45.7% lower latency.
2. The important NCU insight is that V1 is latency-bound due to insufficient eligible warps; V2 raises parallelism and occupancy.
3. We keep the real evidence in the repo and avoid making unsupported claims in the public summary.
4. Missing comparisons are listed as missing rather than guessed.

---

## Next Steps

The next real optimization targets are not vague “more speed” ideas; they are specific ways to reduce on-chip memory pressure within each tile:

- register blocking: keep more accumulation in registers instead of repeatedly writing to shared memory
- float4 vectorization: reduce instruction count and improve memory coalescing
- bank-conflict elimination: adjust the shared-memory layout and access pattern
- higher-end paths: Tensor Core / `ldmatrix`-style movement when the goal is to reduce smem instruction count further

This is the same root cause that shows up in both Flash Attention and MatMul: the problem is not raw arithmetic capability, but the path and organization of memory access inside the tile.
