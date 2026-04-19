#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>
#include <float.h>

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
const int SEQ_LEN = 64;    // 序列长度（先用小值调通）
const int HEAD_DIM = 64;   // 每个 head 的维度
const int BLOCK_SIZE = 16; // tile 大小

// -------------------------------------------------------
// Kernel: Flash Attention（单个 head）
//
// 输入：
//   Q, K, V  shape: [SEQ_LEN, HEAD_DIM]
// 输出：
//   O        shape: [SEQ_LEN, HEAD_DIM]
//
// Grid:  (ceil_div(seq_len, BLOCK_SIZE),)  — 每个 block 负责 Q 的一个 tile
// Block: (BLOCK_SIZE,)                     — 这版实现要求 blockDim.x == BLOCK_SIZE
// Smem:  3 * BLOCK_SIZE * HEAD_DIM * sizeof(float)
//
// 注意：
// 1) 这版 demo 把 HEAD_DIM 固定成编译期常量，要求传入的 head_dim == HEAD_DIM。
// 2) O 的中间状态保存在寄存器里，所有 K/V tile 处理完后再一次性写回全局内存。
// -------------------------------------------------------
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
    if (blockDim.x != BLOCK_SIZE || head_dim != HEAD_DIM) return;

    // 每个 block 负责一个 Q 的 tile
    int tile_row = blockIdx.x * BLOCK_SIZE;
    int tid = threadIdx.x;
    int q_row = tile_row + tid;
    bool active = (q_row < seq_len);

    // 申请 shared memory
    extern __shared__ float smem[];
    float* Qi = smem;                          // [block_size, head_dim]
    float* Kj = Qi + BLOCK_SIZE * HEAD_DIM;    // [block_size, head_dim]
    float* Vj = Kj + BLOCK_SIZE * HEAD_DIM;    // [block_size, head_dim]

    // 把当前 q_row 的输出累积在寄存器里，最后一次性写回全局内存。
    float o_reg[HEAD_DIM];
#pragma unroll
    for (int d = 0; d < HEAD_DIM; d++) {
        o_reg[d] = 0.0f;
    }

    // 加载 Q tile 到 shared memory（SRAM 是 block 内所有线程共享的）
    // 具体哪个线程加载哪个元素无所谓，只要把 BLOCK_SIZE 行的数据都搬进来即可
    // 用步长循环让所有 16 个线程合作加载 1024 个元素，提高内存带宽利用率
    for (int idx = tid; idx < BLOCK_SIZE * HEAD_DIM; idx += blockDim.x) {
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
    for (int j = 0; j < seq_len; j += BLOCK_SIZE) {

        // 加载当前 K/V tile 到 shared memory（同样是所有线程协作加载，具体分配无所谓）
        for (int idx = tid; idx < BLOCK_SIZE * HEAD_DIM; idx += blockDim.x) {
            int row = idx / HEAD_DIM;
            int col = idx % HEAD_DIM;
            int global_row = j + row;
            Kj[idx] = (global_row < seq_len) ? K[global_row * HEAD_DIM + col] : 0.0f;
            Vj[idx] = (global_row < seq_len) ? V[global_row * HEAD_DIM + col] : 0.0f;
        }
        __syncthreads();  // 数据准备完成，所有线程可以安全地访问 Kj 和 Vj

        // 当前线程负责 q_row 这一行
        if (active) {
            for (int row = 0; row < BLOCK_SIZE; row++) {
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
                    o_reg[d] = (o_reg[d] * l * exp_old + v * exp_new) / l_new;
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
            O[q_row * HEAD_DIM + d] = o_reg[d];
        }
    }
}

// -------------------------------------------------------
// Flash Attention v2 骨架（仅结构，不含完整实现）
//
// 思路（和 v1 的区别）：
// 1) 依然按 Q/K/V tile 流式处理
// 2) 更强调在 block/warp 内做并行归约（max/sum）
// 3) 尽量把中间量留在寄存器，减少对全局内存读写
// -------------------------------------------------------
__global__ void flash_attn_v2_kernel_skeleton(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    int seq_len,
    int head_dim,
    float scale
) {
    int tile_row = blockIdx.x * BLOCK_SIZE;
    int tid = threadIdx.x;
    int q_row = tile_row + tid;
    bool active = (q_row < seq_len);

    extern __shared__ float smem[];
    float* Qi = smem;
    float* Kj = Qi + BLOCK_SIZE * head_dim;
    float* Vj = Kj + BLOCK_SIZE * head_dim;

    // TODO(v2): 也可以把 O 的部分累积放到寄存器数组，最后一次性写回。

    // 1) 载入当前 Q tile
    for (int idx = tid; idx < BLOCK_SIZE * head_dim; idx += blockDim.x) {
        int row = idx / head_dim;
        int col = idx % head_dim;
        int global_row = tile_row + row;
        Qi[idx] = (global_row < seq_len) ? Q[global_row * head_dim + col] : 0.0f;
    }
    __syncthreads();

    // 2) 初始化在线 softmax 状态
    float m = -FLT_MAX;
    float l = 0.0f;
    if (active) {
        for (int d = 0; d < head_dim; d++) {
            O[q_row * head_dim + d] = 0.0f;
        }
    }

    // 3) 流式遍历所有 K/V tile
    for (int j = 0; j < seq_len; j += BLOCK_SIZE) {
        for (int idx = tid; idx < BLOCK_SIZE * head_dim; idx += blockDim.x) {
            int row = idx / head_dim;
            int col = idx % head_dim;
            int global_row = j + row;
            Kj[idx] = (global_row < seq_len) ? K[global_row * head_dim + col] : 0.0f;
            Vj[idx] = (global_row < seq_len) ? V[global_row * head_dim + col] : 0.0f;
        }
        __syncthreads();

        // TODO(v2-step1): 每个线程/warp 计算局部 score 片段。
        // TODO(v2-step2): 对 score 在 block/warp 内做并行 max 归约，得到 m_tile。
        // TODO(v2-step3): 对 exp(score - m_tile) 做并行 sum 归约，得到 l_tile。
        // TODO(v2-step4): 用 m/l 与 m_tile/l_tile 合并，并并行累积 O。

        // 占位：避免 scale 参数未使用
        if (active && scale < 0.0f) {
            O[q_row * head_dim] = O[q_row * head_dim];
        }

        __syncthreads();
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


int main() {
    const int total = SEQ_LEN * HEAD_DIM;
    float scale = 1.0f / sqrtf((float)HEAD_DIM);

    // Host 内存
    float* h_Q = new float[total];
    float* h_K = new float[total];
    float* h_V = new float[total];
    float* h_ref = new float[total];
    float* h_out = new float[total];

    // 随机初始化
    srand(42);
    for (int i = 0; i < total; i++) {
        h_Q[i] = (float)rand() / RAND_MAX - 0.5f;
        h_K[i] = (float)rand() / RAND_MAX - 0.5f;
        h_V[i] = (float)rand() / RAND_MAX - 0.5f;
    }

    // CPU 参考结果
    attention_cpu(h_Q, h_K, h_V, h_ref, SEQ_LEN, HEAD_DIM);

    // Device 内存
    float *d_Q, *d_K, *d_V, *d_O;
    cudaMalloc(&d_Q, total * sizeof(float));
    cudaMalloc(&d_K, total * sizeof(float));
    cudaMalloc(&d_V, total * sizeof(float));
    cudaMalloc(&d_O, total * sizeof(float));

    cudaMemcpy(d_Q, h_Q, total * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K, total * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V, total * sizeof(float), cudaMemcpyHostToDevice);

    // -------------------------------------------------------
    // 普通 GPU Attention
    // -------------------------------------------------------
    float* d_S;
    cudaMalloc(&d_S, SEQ_LEN * SEQ_LEN * sizeof(float));

    dim3 block2d(16, 16);
    dim3 grid_qk((SEQ_LEN + 15) / 16, (SEQ_LEN + 15) / 16);  // Kernel 1
    dim3 grid_sv((HEAD_DIM + 15) / 16, (SEQ_LEN + 15) / 16); // Kernel 3

    // Kernel 1: S = Q @ K^T * scale
    qk_dot_kernel<<<grid_qk, block2d>>>(d_Q, d_K, d_S, SEQ_LEN, HEAD_DIM, scale);

    // Kernel 2: softmax over each row
    softmax_kernel<<<SEQ_LEN, 1>>>(d_S, SEQ_LEN);

    // Kernel 3: O = S @ V
    sv_dot_kernel<<<grid_sv, block2d>>>(d_S, d_V, d_O, SEQ_LEN, HEAD_DIM);

    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_O, total * sizeof(float), cudaMemcpyDeviceToHost);

    if (verify(h_ref, h_out, total))
        printf("普通 GPU Attention: PASS\n");

    cudaFree(d_S);

    // -------------------------------------------------------
    // Flash Attention
    // -------------------------------------------------------
    int grid_fa = (SEQ_LEN + BLOCK_SIZE - 1) / BLOCK_SIZE;
    size_t smem_size = 3 * BLOCK_SIZE * HEAD_DIM * sizeof(float);

    flash_attn_kernel<<<grid_fa, BLOCK_SIZE, smem_size>>>(
        d_Q, d_K, d_V, d_O, SEQ_LEN, HEAD_DIM, scale
    );

    cudaDeviceSynchronize();
    float* h_out2 = new float[total];
    cudaMemcpy(h_out2, d_O, total * sizeof(float), cudaMemcpyDeviceToHost);

    if (verify(h_ref, h_out2, total))
        printf("Flash Attention:    PASS\n");

    delete[] h_out2;

    // Cleanup
    delete[] h_Q; delete[] h_K; delete[] h_V; delete[] h_ref; delete[] h_out;
    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_O);
    return 0;
}
