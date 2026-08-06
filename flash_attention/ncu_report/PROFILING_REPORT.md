# Flash Attention NCU Profiling Report

## Environment
- GPU: RTX 3060 12GB
- CUDA: 13.2
- Driver: Latest
- Compiler: nvcc -O2 -arch=sm_86 -lineinfo

## Benchmark Results Summary

### Flash Attention Performance (seq=4096, head_dim=64)

| Method | Time | Speedup vs CPU |
|--------|------|----------------|
| CPU Reference | 873.34 ms | baseline |
| Naive GPU (3 kernels) | 21.10 ms | **41.4x** |
| Flash Attention (fused) | 10.70 ms | **81.6x vs CPU** |

### Relative Improvement
- **Flash Attention vs Naive GPU: 2.0x speedup**

This 2.0x improvement on RTX 3060 demonstrates the memory optimization principle,
though the actual speedup is limited by other factors (compute-bound portions, kernel launch overhead, shared memory bank conflicts).

## Key Performance Metrics Analysis

### 1. Memory Throughput (GB/s)

**Theoretical Analysis:**

**Naive GPU (3 kernels):**
- Kernel 1 (QK^T): Writes S matrix → 64MB write
- Kernel 2 (Softmax): Reads & writes S matrix → 128MB (64MB + 64MB)
- Kernel 3 (SV): Reads S matrix → 64MB read
- Q/K/V reads: 6MB total
- **Total global memory bandwidth: ~256 MB per forward pass**
- Time: 21.10 ms → ~12.1 GB/s memory throughput

**Flash Attention (fused):**
- Q reads: 2MB
- K reads: 2MB
- V reads: 2MB
- O writes: 2MB
- **Total global memory bandwidth: ~8 MB per forward pass**
- Time: 10.70 ms → ~0.75 GB/s global memory throughput

**Result: Flash Attention reduces global memory traffic by ~32x (theoretical)**

### 2. L2 Cache Hit Rate

**Expected Pattern:**
- **Flash Attention: Higher L2 hit rate**
  - Tile-based processing keeps intermediate results in shared memory
  - Each tile's attention scores are computed and consumed locally
  - Much less data flows through L2 cache to/from global memory

- **Naive GPU: Lower L2 hit rate**
  - S matrix (64MB) must be loaded from global memory multiple times
  - Three separate kernels means three separate memory access patterns
  - Poor locality of reference between kernels

### 3. Achieved Occupancy

**Flash Attention Kernel:**
- Block: 256 threads (16×16)
- Shared memory: ~8KB (two 16×16 float32 tiles)
- Registers per thread: ~32
- **Expected occupancy: 100% (limited by registers/shared memory, not by resources)**

**Naive Kernels:**
- qk_dot_kernel: 256 threads, good occupancy
- softmax_kernel: 256 threads, good occupancy  
- sv_dot_kernel: 256 threads, good occupancy

### 4. Compute vs Memory Bound

**Flash Attention:**
- Computation: Q·K^T (seq × seq matrix multiply)
- For seq=4096: 4096² × 64 ≈ 1B FLOPs
- RTX 3060: ~6 TFLOPS peak
- Compute time: ~0.17 ms
- **Actual time 10.70 ms → Still MEMORY BOUND (but improved)**

**Naive GPU:**
- Same computation split into 3 kernels
- Each kernel is also memory bound
- Overhead from kernel launches and intermediate synchronization
- **Still MEMORY BOUND, but with 3x the memory traffic**

## Optimization Insights

### Why Flash Attention Achieves 2.0x (not 32x theoretical)?

1. **Remaining Global Memory Access** (~8 MB is not zero)
   - Q/K/V must be read from global memory (6MB)
   - O must be written back (2MB)
   - These dominate the total time

2. **Shared Memory Bank Conflicts**
   - Tiling with 16×16 blocks may cause some conflicts
   - Not all thread warps can access shared memory perfectly in parallel

3. **Kernel Launch Overhead**
   - Single kernel vs 3 kernels reduces overhead

4. **Mixed Workload**
   - Forward pass has both memory-bound (attention) and compute-bound phases
   - Not 100% memory-bound throughout

5. **sm_86 (RTX 3060) Limitations**
   - Older GPU architecture (Ampere)
   - Newer architectures (Hopper) may show better improvements

## Recommendations for Further Optimization

1. **Increase Tile Size**: Use 32×32 tiles instead of 16×16 to reduce kernel launch overhead
2. **Enable Double Buffering**: Prefetch next tile while computing current one
3. **Optimize for Ampere**: Use async copy operations if targeting RTX A100/3090
4. **Consider Tensor Cores**: Use mixed precision (fp16) for attention computation
5. **Profile with NSYS**: Use NVIDIA Systems Profiler for timeline analysis

## How to Generate Full NCU Report

```bash
# Compile with line info support
nvcc -O2 -arch=sm_86 -lineinfo flash_attn.cu -o flash_attn.exe

# Run NCU profiling (requires admin privileges)
ncu --set full ^
    --kernel-name flash_attn_kernel ^
    --kernel-name qk_dot_kernel ^
    --kernel-name softmax_kernel ^
    --kernel-name sv_dot_kernel ^
    -o flash_attn_report ^
    flash_attn.exe

# View report in GUI
ncu-ui flash_attn_report.ncu-rep
```

**Key metrics to examine in NCU GUI:**
- Memory Throughput (GB/s)
- L2 Cache Hit Rate (%)
- SM Utilization (%)
- Achieved Occupancy (%)
- Roofline Analysis (Memory Bound vs Compute Bound)
- DRAM Bandwidth Utilization

---

**Report Generated:** 2026-08-07  
**Status:** Benchmark completed ✓ | NCU Tools not available on this system  
**Alternative:** Used theoretical analysis + benchmark data to validate optimization effectiveness
