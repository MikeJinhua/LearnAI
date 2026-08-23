#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <string.h>

// -------------------------------------------------------
// Flash Attention v1 骨架
//
// 目标：计算 Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d)) @ V
//
// 普通做法的问题：
//   Q @ K^T 结果是 [seq_len, seq_len]，当 seq_len=4096 时
//   显存占用 = 4096 * 4096 * 4 bytes = 64MB（每个 head！）
//
// Flash Attention 的思路：
//   把 Q/K/V 切成小 tile，在 shared memory 里做 online softmax
//   始终只用 O(tile_size) 的显存，不需要存整张 attention matrix
// -------------------------------------------------------

// 超参数
constexpr int SEQ_LEN = 4096;    // 序列长度（先用小值调通）
constexpr int HEAD_DIM = 64;   // 每个 head 的维度

// -------------------------------------------------------
// Kernel: Flash Attention（单个 head）
//
// 输入：
//   Q, K, V  shape: [SEQ_LEN, HEAD_DIM]
// 输出：
//   O        shape: [SEQ_LEN, HEAD_DIM]
//
// Grid:  (ceil_div(seq_len, 32),)  — 每个 block 负责 Q 的一个 tile
// Block: (32,)                     — 这版实现要求 blockDim.x == 32
// Smem:  3 * 32 * HEAD_DIM * sizeof(float)
//
// 注意：
// 1) 这版 demo 把 HEAD_DIM 固定成编译期常量，要求传入的 head_dim == HEAD_DIM。
// 2) O 的中间状态保存在寄存器里，所有 K/V tile 处理完后再一次性写回全局内存。
// -------------------------------------------------------
// 普通 GPU Attention（三个 kernel，不做 tiling 优化）
// -------------------------------------------------------

// Kernel 1: S = Q @ K^T * scale
// Grid:  (ceil_div(seq_len,16), ceil_div(seq_len,16))  Block: (16, 16)
// 线程 (i, j) 负责计算 S[i][j]
__global__ void qk_dot_kernel(
    const float* Q, const float* K, float* S,
    int seq_len, int head_dim, float scale
) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= seq_len || j >= seq_len) return;

    float val = 0.0f;
    for (int d = 0; d < head_dim; d++) {
        val += Q[i * head_dim + d] * K[j * head_dim + d];
    }
    S[i * seq_len + j] = val * scale;
}

// Kernel 2: softmax over each row
// Grid:  (seq_len,)  Block: (1,)
// 每个线程独立处理一整行（seq_len 小时够用）
__global__ void softmax_kernel(float* S, int seq_len) {
    int i = blockIdx.x;
    if (i >= seq_len) return;

    float* row = S + i * seq_len;

    // 找最大值（数值稳定）
    float max_val = -FLT_MAX;
    for (int j = 0; j < seq_len; j++)
        max_val = fmaxf(max_val, row[j]);

    // exp 并求和
    float sum = 0.0f;
    for (int j = 0; j < seq_len; j++) {
        row[j] = expf(row[j] - max_val);
        sum += row[j];
    }

    // 归一化
    for (int j = 0; j < seq_len; j++)
        row[j] /= sum;
}

// Kernel 3: O = S @ V
// Grid:  (ceil_div(head_dim,16), ceil_div(seq_len,16))  Block: (16, 16)
// 线程 (i, d) 负责计算 O[i][d]
__global__ void sv_dot_kernel(
    const float* S, const float* V, float* O,
    int seq_len, int head_dim
) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= seq_len || d >= head_dim) return;

    float val = 0.0f;
    for (int j = 0; j < seq_len; j++) {
        val += S[i * seq_len + j] * V[j * head_dim + d];
    }
    O[i * head_dim + d] = val;
}


// -------------------------------------------------------
// Flash Attention kernel（online softmax 简化版）
// -------------------------------------------------------
__global__ void flash_attn_kernel(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    int seq_len,
    int head_dim,
    float scale  // = 1 / sqrt(head_dim)
) {
    // 这版 kernel 把每个线程绑定到一行 Q，并把 O 暂存在寄存器里，
    // 因此要求 launch 时使用固定的线程数和 head_dim。
    if (blockDim.x != 32 || head_dim != HEAD_DIM) return;

    // 每个 block 负责一个 Q 的 tile
    int tile_row = blockIdx.x * 32;
    int tid = threadIdx.x;
    int q_row = tile_row + tid;
    bool active = (q_row < seq_len);

    // 申请 shared memory
    extern __shared__ float smem[];
    float* Qi = smem;                          // [block_size, head_dim]
    float* Kj = Qi + 32 * HEAD_DIM;    // [block_size, head_dim]
    float* Vj = Kj + 32 * HEAD_DIM;    // [block_size, head_dim]

    // 把当前 q_row 的输出累积在寄存器里，最后一次性写回全局内存。
    float o_reg[HEAD_DIM];
#pragma unroll
    for (int d = 0; d < HEAD_DIM; d++) {
        o_reg[d] = 0.0f;
    }

    // 加载 Q tile 到 shared memory（SRAM 是 block 内所有线程共享的）
    // 用步长循环让所有 32 个线程合作加载，提高内存带宽利用率
    for (int idx = tid; idx < 32 * HEAD_DIM; idx += blockDim.x) {
        int row = idx / HEAD_DIM;
        int col = idx % HEAD_DIM;
        int global_row = tile_row + row;
        Qi[idx] = (global_row < seq_len) ? Q[global_row * HEAD_DIM + col] : 0.0f;
    }
    __syncthreads();  // 确保所有数据都加载完成，之后所有线程都能访问完整的 Qi tile

    // 初始化 m, l, O
    float m = -FLT_MAX;
    float l = 0.0f;

    // 外层循环：遍历所有 K/V 块
    for (int j = 0; j < seq_len; j += 32) {

        // 加载当前 K/V tile 到 shared memory（同样是所有线程协作加载，具体分配无所谓）
        for (int idx = tid; idx < 32 * HEAD_DIM; idx += blockDim.x) {
            int row = idx / HEAD_DIM;
            int col = idx % HEAD_DIM;
            int global_row = j + row;
            Kj[idx] = (global_row < seq_len) ? K[global_row * HEAD_DIM + col] : 0.0f;
            Vj[idx] = (global_row < seq_len) ? V[global_row * HEAD_DIM + col] : 0.0f;
        }
        __syncthreads();  // 数据准备完成，所有线程可以安全地访问 Kj 和 Vj

        // 当前线程负责 q_row 这一行
        if (active) {
            for (int row = 0; row < 32; row++) {
                int k_row = j + row;
                if (k_row >= seq_len) continue;

                // 1) 计算当前 score = Q[q_row] · K[k_row]
                float score = 0.0f;
#pragma unroll
                for (int d = 0; d < HEAD_DIM; d++) {
                    score += Qi[tid * HEAD_DIM + d] * Kj[row * HEAD_DIM + d];
                }
                score *= scale;

                // 2) online softmax 更新
                float m_new = fmaxf(m, score);
                float exp_old = expf(m - m_new);
                float exp_new = expf(score - m_new);
                float l_new = l * exp_old + exp_new;

                // 3) 更新输出 O[q_row, :]
#pragma unroll
                for (int d = 0; d < HEAD_DIM; d++) {
                    float v = Vj[row * HEAD_DIM + d];
                    o_reg[d] = o_reg[d] * exp_old + v * exp_new;
                }

                // 4) 更新状态
                m = m_new;
                l = l_new;
            }
        }

        __syncthreads();
    }

    if (active) {
#pragma unroll
        for (int d = 0; d < HEAD_DIM; d++) {
            O[q_row * HEAD_DIM + d] = o_reg[d] / l;
        }
    }
}

// -------------------------------------------------------
// Flash Attention v2：一个 warp 协作处理一行 Q
//
// 思路（和 v1 的区别）：
// 1) 依然按 Q/K/V tile 流式处理
// 2) 更强调在 block/warp 内做并行归约（max/sum）
// 3) 尽量把中间量留在寄存器，减少对全局内存读写
// -------------------------------------------------------
__global__ void flash_attn_v2_kernel(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    int seq_len,
    int head_dim,
    float scale
) {
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    int tile_row = blockIdx.x * 32;

    extern __shared__ float smem[];
    float* Kj = smem;
    float* Vj = Kj + 32 * head_dim;

    // 每个 warp 负责 4 行；每个 lane 保存每行的两个 Q/O 元素。
    float q0[4];
    float q1[4];
    float o0[4] = {0.0f};
    float o1[4] = {0.0f};
    float m[4];
    float l[4] = {0.0f};

#pragma unroll
    for (int r = 0; r < 4; ++r) {
        int q_row = tile_row + warp + r * 8;
        bool active = q_row < seq_len;
        q0[r] = active ? Q[q_row * HEAD_DIM + lane] : 0.0f;
        q1[r] = active ? Q[q_row * HEAD_DIM + lane + 32] : 0.0f;
        m[r] = -FLT_MAX;
    }

    // 每个 block 仍处理 32 行 Q，K/V tile 的跨行复用粒度与 V1 相同。
    for (int j = 0; j < seq_len; j += 32) {
        for (int idx = tid; idx < 32 * head_dim; idx += blockDim.x) {
            int row = idx / head_dim;
            int col = idx % head_dim;
            int global_row = j + row;
            Kj[idx] = (global_row < seq_len) ? K[global_row * head_dim + col] : 0.0f;
            Vj[idx] = (global_row < seq_len) ? V[global_row * head_dim + col] : 0.0f;
        }
        __syncthreads();

        for (int k = 0; k < 32 && j + k < seq_len; ++k) {
            float k0 = Kj[k * HEAD_DIM + lane];
            float k1 = Kj[k * HEAD_DIM + lane + 32];
            float v0 = Vj[k * HEAD_DIM + lane];
            float v1 = Vj[k * HEAD_DIM + lane + 32];

#pragma unroll
            for (int r = 0; r < 4; ++r) {
                float score = q0[r] * k0 + q1[r] * k1;
#pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1) {
                    score += __shfl_down_sync(0xffffffff, score, offset);
                }
                score = __shfl_sync(0xffffffff, score, 0) * scale;

                float m_new = fmaxf(m[r], score);
                float exp_old = expf(m[r] - m_new);
                float exp_new = expf(score - m_new);
                l[r] = l[r] * exp_old + exp_new;
                o0[r] = o0[r] * exp_old + v0 * exp_new;
                o1[r] = o1[r] * exp_old + v1 * exp_new;
                m[r] = m_new;
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int r = 0; r < 4; ++r) {
        int q_row = tile_row + warp + r * 8;
        if (q_row < seq_len) {
            O[q_row * HEAD_DIM + lane] = o0[r] / l[r];
            O[q_row * HEAD_DIM + lane + 32] = o1[r] / l[r];
        }
    }
}


// -------------------------------------------------------
// CPU 参考实现（朴素 attention，用来验证正确性）
// -------------------------------------------------------
void attention_cpu(
    const float* Q, const float* K, const float* V,
    float* O, int seq_len, int head_dim
) {
    float scale = 1.0f / sqrtf((float)head_dim);
    float* scores = new float[seq_len * seq_len];

    // S = Q @ K^T * scale
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < seq_len; j++) {
            float s = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                s += Q[i * head_dim + d] * K[j * head_dim + d];
            }
            scores[i * seq_len + j] = s * scale;
        }
    }

    // softmax over each row
    for (int i = 0; i < seq_len; i++) {
        float max_val = -FLT_MAX;
        for (int j = 0; j < seq_len; j++) max_val = fmaxf(max_val, scores[i * seq_len + j]);
        float sum = 0.0f;
        for (int j = 0; j < seq_len; j++) {
            scores[i * seq_len + j] = expf(scores[i * seq_len + j] - max_val);
            sum += scores[i * seq_len + j];
        }
        for (int j = 0; j < seq_len; j++) scores[i * seq_len + j] /= sum;
    }

    // O = softmax(S) @ V
    for (int i = 0; i < seq_len; i++) {
        for (int d = 0; d < head_dim; d++) {
            float val = 0.0f;
            for (int j = 0; j < seq_len; j++) {
                val += scores[i * seq_len + j] * V[j * head_dim + d];
            }
            O[i * head_dim + d] = val;
        }
    }

    delete[] scores;
}


// -------------------------------------------------------
// 验证
// -------------------------------------------------------
bool verify(const float* ref, const float* out, int total, float eps = 1e-4f) {
    for (int i = 0; i < total; i++) {
        if (fabsf(ref[i] - out[i]) > eps) {
            printf("MISMATCH at %d: ref=%.6f, got=%.6f\n", i, ref[i], out[i]);
            return false;
        }
    }
    return true;
}


int main(int argc, char** argv) {
    const int total = SEQ_LEN * HEAD_DIM;
    const int RUNS  = 100;
    float scale = 1.0f / sqrtf((float)HEAD_DIM);

    // Host 内存
    float* h_Q   = new float[total];
    float* h_K   = new float[total];
    float* h_V   = new float[total];
    float* h_ref = new float[total];
    float* h_out = new float[total];

    srand(42);
    for (int i = 0; i < total; i++) {
        h_Q[i] = (float)rand() / RAND_MAX - 0.5f;
        h_K[i] = (float)rand() / RAND_MAX - 0.5f;
        h_V[i] = (float)rand() / RAND_MAX - 0.5f;
    }

    bool profile_v1 = argc > 1 && strcmp(argv[1], "--profile-v1") == 0;
    bool profile_v2 = argc > 1 && strcmp(argv[1], "--profile-v2") == 0;

    // -------------------------------------------------------
    // CPU 计时
    // -------------------------------------------------------
    double cpu_ms = 0.0;
    if (!profile_v1 && !profile_v2) {
        attention_cpu(h_Q, h_K, h_V, h_ref, SEQ_LEN, HEAD_DIM); // warmup + 生成参考结果
        clock_t t0 = clock();
        for (int r = 0; r < RUNS; r++)
            attention_cpu(h_Q, h_K, h_V, h_out, SEQ_LEN, HEAD_DIM);
        cpu_ms = (double)(clock() - t0) / CLOCKS_PER_SEC * 1000.0 / RUNS;
    }

    // Device 内存
    float *d_Q, *d_K, *d_V, *d_O;
    cudaMalloc(&d_Q, total * sizeof(float));
    cudaMalloc(&d_K, total * sizeof(float));
    cudaMalloc(&d_V, total * sizeof(float));
    cudaMalloc(&d_O, total * sizeof(float));
    cudaMemcpy(d_Q, h_Q, total * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K, total * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V, total * sizeof(float), cudaMemcpyHostToDevice);

    int grid_fa = (SEQ_LEN + 32 - 1) / 32;
    size_t smem_v1 = 3 * 32 * HEAD_DIM * sizeof(float);
    size_t smem_v2 = 2 * 32 * HEAD_DIM * sizeof(float);

    // NCU 专用模式：一次 warmup + 一次待采集 launch；配合 --launch-skip 1。
    if (profile_v1 || profile_v2) {
        for (int launch = 0; launch < 2; ++launch) {
            if (profile_v1) {
                flash_attn_kernel<<<grid_fa, 32, smem_v1>>>(
                    d_Q, d_K, d_V, d_O, SEQ_LEN, HEAD_DIM, scale);
            } else {
                flash_attn_v2_kernel<<<grid_fa, 256, smem_v2>>>(
                    d_Q, d_K, d_V, d_O, SEQ_LEN, HEAD_DIM, scale);
            }
        }
        cudaDeviceSynchronize();
        printf("NCU target completed: %s\n", profile_v1 ? "flash_attn_kernel" : "flash_attn_v2_kernel");
        delete[] h_Q; delete[] h_K; delete[] h_V; delete[] h_ref; delete[] h_out;
        cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_O);
        return 0;
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float ms = 0;

    // -------------------------------------------------------
    // 普通 GPU Attention：正确性 + 计时
    // -------------------------------------------------------
    float* d_S;
    cudaMalloc(&d_S, SEQ_LEN * SEQ_LEN * sizeof(float));
    dim3 block2d(16, 16);
    dim3 grid_qk((SEQ_LEN + 15) / 16, (SEQ_LEN + 15) / 16);
    dim3 grid_sv((HEAD_DIM + 15) / 16, (SEQ_LEN + 15) / 16);

    auto run_naive = [&]() {
        qk_dot_kernel<<<grid_qk, block2d>>>(d_Q, d_K, d_S, SEQ_LEN, HEAD_DIM, scale);
        softmax_kernel<<<SEQ_LEN, 1>>>(d_S, SEQ_LEN);
        sv_dot_kernel<<<grid_sv, block2d>>>(d_S, d_V, d_O, SEQ_LEN, HEAD_DIM);
    };

    run_naive(); cudaDeviceSynchronize(); // warmup
    cudaMemcpy(h_out, d_O, total * sizeof(float), cudaMemcpyDeviceToHost);
    bool naive_ok = verify(h_ref, h_out, total);

    cudaEventRecord(start);
    for (int r = 0; r < RUNS; r++) run_naive();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    double naive_ms = ms / RUNS;

    cudaFree(d_S);

    // -------------------------------------------------------
    // Flash Attention：正确性 + 计时
    // -------------------------------------------------------
    auto run_flash = [&]() {
        flash_attn_kernel<<<grid_fa, 32, smem_v1>>>(
            d_Q, d_K, d_V, d_O, SEQ_LEN, HEAD_DIM, scale);
    };

    run_flash(); cudaDeviceSynchronize(); // warmup
    float* h_out2 = new float[total];
    cudaMemcpy(h_out2, d_O, total * sizeof(float), cudaMemcpyDeviceToHost);
    bool flash_ok = verify(h_ref, h_out2, total);
    delete[] h_out2;

    cudaEventRecord(start);
    for (int r = 0; r < RUNS; r++) run_flash();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    double flash_ms = ms / RUNS;

    auto run_flash_v2 = [&]() {
        flash_attn_v2_kernel<<<grid_fa, 256, smem_v2>>>(
            d_Q, d_K, d_V, d_O, SEQ_LEN, HEAD_DIM, scale);
    };

    run_flash_v2(); cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_O, total * sizeof(float), cudaMemcpyDeviceToHost);
    bool flash_v2_ok = verify(h_ref, h_out, total);

    cudaEventRecord(start);
    for (int r = 0; r < RUNS; r++) run_flash_v2();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    double flash_v2_ms = ms / RUNS;

    // -------------------------------------------------------
    // 结果汇总
    // -------------------------------------------------------
    printf("\n========== Attention Benchmark (seq=%d, d=%d, runs=%d) ==========\n",
           SEQ_LEN, HEAD_DIM, RUNS);
    printf("%-20s  %s   %8s\n", "Method", "Correct", "Avg time");
    printf("%-20s  %s   %7.4f ms\n", "CPU",          "----", cpu_ms);
    printf("%-20s  %s   %7.4f ms\n", "Naive GPU",    naive_ok ? "PASS" : "FAIL", naive_ms);
    printf("%-20s  %s   %7.4f ms\n", "Flash Attn",   flash_ok ? "PASS" : "FAIL", flash_ms);
    printf("%-20s  %s   %7.4f ms\n", "Flash Attn V2", flash_v2_ok ? "PASS" : "FAIL", flash_v2_ms);
    printf("  Naive GPU vs CPU:  %.1fx\n", cpu_ms / naive_ms);
    printf("  Flash  vs Naive:   %.1fx\n", naive_ms / flash_ms);
    printf("  V2 vs V1:          %.1fx\n", flash_ms / flash_v2_ms);
    printf("=================================================================\n");

    // Cleanup
    cudaEventDestroy(start); cudaEventDestroy(stop);
    delete[] h_Q; delete[] h_K; delete[] h_V; delete[] h_ref; delete[] h_out;
    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_O);
    return 0;
}
