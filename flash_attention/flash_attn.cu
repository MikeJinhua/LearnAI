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
// Grid:  (SEQ_LEN / BLOCK_SIZE,)  — 每个 block 负责 Q 的一个 tile
// Block: (BLOCK_SIZE,)
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
    // 每个 block 负责一个 Q 的 tile
    int tile_row = blockIdx.x * BLOCK_SIZE;
    int tid = threadIdx.x;

    // 申请 shared memory
    extern __shared__ float smem[];
    float* Qi = smem;                          // [block_size, head_dim]
    float* Kj = Qi + BLOCK_SIZE * head_dim;    // [block_size, head_dim]
    float* Vj = Kj + BLOCK_SIZE * head_dim;    // [block_size, head_dim]

    // 初始化 m, l, O
    float m = -FLT_MAX;
    float l = 0.0f;
    // O 初始化为 0
    for (int i = tid; i < head_dim; i += blockDim.x) {
        O[tile_row * head_dim + i] = 0.0f;
    }

    // 外层循环：遍历所有 K/V 块
    for (int j = 0; j < seq_len; j += BLOCK_SIZE) {

        // 把 Kj, Vj 从 global memory 加载到 shared memory
        // ???

        // 计算 Sij = Qi × Kj^T
        // ???

        // online softmax 更新 m, l, O
        // ???
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

    // Launch（TODO: 填完 kernel 后取消注释）
    // int grid  = SEQ_LEN / BLOCK_SIZE;
    // int block = BLOCK_SIZE;
    // flash_attn_kernel<<<grid, block>>>(d_Q, d_K, d_V, d_O, SEQ_LEN, HEAD_DIM, scale);
    // cudaDeviceSynchronize();

    // cudaMemcpy(h_out, d_O, total * sizeof(float), cudaMemcpyDeviceToHost);
    // if (verify(h_ref, h_out, total)) printf("✅ Flash Attention correct!\n");

    printf("骨架搭好了，开始填 TODO 吧！\n");

    // Cleanup
    delete[] h_Q; delete[] h_K; delete[] h_V; delete[] h_ref; delete[] h_out;
    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_O);
    return 0;
}
