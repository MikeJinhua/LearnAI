#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>
#include <float.h>

// -------------------------------------------------------
// Naive Softmax: 每个 thread 负责一整行
// Grid:  (M,)    — M 个 block，每个 block 一个 thread
// Block: (1,)
// 缺点：完全没有并行，纯串行，但逻辑最清晰
// -------------------------------------------------------
__global__ void softmax_naive(float* input, float* output, int M, int N) {
    int row = blockIdx.x;  // 这个 thread 负责第 row 行
    if (row >= M) return;

    float* x = input  + row * N;
    float* y = output + row * N;

    // Pass 1: 找最大值（numerical stability）
    float max_val = x[0];
    for (int i = 1; i < N; i++) {
        max_val = fmaxf(max_val, x[i]);
    }

    // Pass 2: 计算 exp 之和
    float sum = 0.0f;
    for (int i = 0; i < N; i++) {
        sum += expf(x[i] - max_val);
    }

    // Pass 3: normalize
    for (int i = 0; i < N; i++) {
        y[i] = expf(x[i] - max_val) / sum;
    }
}

__global__ void softmax_v3(float* input, float* output, int M, int N) {
    __shared__ float smem_max[256];
    __shared__ float smem_sum[256];

    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= M) return;

    float* x = input  + row * N;
    float* y = output + row * N;

    // Step 1: 每个 thread 扫自己的元素，同时维护局部 max 和 sum
    float local_max = -FLT_MAX;
    float local_sum = 0.0f;
    for (int i = tid; i < N; i += blockDim.x) {
        float val = x[i];
        if (val > local_max) {
            
        // max 更新了，sum 需要修正
        local_sum = local_sum * expf(local_max - val) + 1.0f;  // +1 是因为 exp(val-val)=1
        local_max = val;
        } else {
            // max 没变，直接累加
            local_sum += expf(val - local_max);  // 直接加，不除
        }
    }
    smem_max[tid] = local_max;
    smem_sum[tid] = local_sum;
    __syncthreads();

    // Step 2: 树形归约，同时归约 max 和 sum
    // 注意：合并两个 thread 的 (max, sum) 时，也要用修正公式
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            float m_a = smem_max[tid];
            float m_b = smem_max[tid + stride];
            float s_a = smem_sum[tid];
            float s_b = smem_sum[tid + stride];
            // ??? 合并 (m_a, s_a) 和 (m_b, s_b)
            if (m_a > m_b) {
                
                // max 更新了，sum 需要修正
                smem_max[tid] = m_a;
                smem_sum[tid] = s_a + s_b * expf(m_b - m_a);
            } else {
                // max 没变，直接累加
                smem_max[tid] = m_b;
                smem_sum[tid] = s_b + s_a * expf(m_a - m_b);
            }
        }
        __syncthreads();
    }

    // Step 3: normalize 写回
    float max_val = smem_max[0];
    float sum_val = smem_sum[0];
    for (int i = tid; i < N; i += blockDim.x) {
        y[i] = expf(x[i] - max_val) / sum_val;
    }
}

__global__ void softmax_v2(float* input, float* output, int M, int N) {
     __shared__ float smem[256];  // shared memory

    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= M) return;

    float* x = input  + row * N;
    float* y = output + row * N;

    // Step 1: 每个 thread 先算自己负责的元素的局部 max
    float local_max = -FLT_MAX;
    for (int i = tid; i < N; i += blockDim.x) {
        local_max = fmaxf(local_max, x[i]);
    }
    smem[tid] = local_max;
    __syncthreads();

    // Step 2: 树形归约找全行 max
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem[tid] = fmaxf(smem[tid], smem[tid + stride]);
        }
        __syncthreads();
    }

    float  max_val = smem[0];
   
     __syncthreads();
    smem[tid] = 0;
    // Step 3: 每个 thread 算局部 sum（exp）
    float local_sum = 0;
    for (int i = tid; i < N; i += blockDim.x) {
         local_sum += expf(x[i] - max_val);
    }
    smem[tid] = local_sum;
    __syncthreads();
    // Step 4: 树形归约找全行 sum
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem[tid] = smem[tid] + smem[tid + stride];
        }
        __syncthreads();
    }
 
    // Step 5: normalize 写回
     for (int i = tid; i < N; i += blockDim.x) {
        y[i] = expf(x[i] - max_val) / smem[0];
    }
}


// -------------------------------------------------------
// CPU 参考实现，用来验证正确性
// -------------------------------------------------------
void softmax_cpu(float* input, float* output, int M, int N) {
    for (int row = 0; row < M; row++) {
        float* x = input  + row * N;
        float* y = output + row * N;

        float max_val = x[0];
        for (int i = 1; i < N; i++) max_val = fmaxf(max_val, x[i]);

        float sum = 0.0f;
        for (int i = 0; i < N; i++) sum += expf(x[i] - max_val);

        for (int i = 0; i < N; i++) y[i] = expf(x[i] - max_val) / sum;
    }
}

// -------------------------------------------------------
// 验证结果
// -------------------------------------------------------
bool verify(float* ref, float* out, int total, float eps = 1e-5f) {
    for (int i = 0; i < total; i++) {
        if (fabsf(ref[i] - out[i]) > eps) {
            printf("MISMATCH at %d: ref=%.6f, got=%.6f\n", i, ref[i], out[i]);
            return false;
        }
    }
    return true;
}

int main() {
    const int M = 1024;   // 行数（比如 batch * seq_len）
    const int N = 1024;   // 列数（比如 vocab_size 或 head_dim）
    const int total = M * N;

    // Host 内存
    float* h_input  = new float[total];
    float* h_ref    = new float[total];
    float* h_output = new float[total];

    // 随机初始化
    
    for (int i = 0; i < total; i++) h_input[i] = (float)rand() / RAND_MAX * 10.0f - 5.0f;

    // CPU 参考
    softmax_cpu(h_input, h_ref, M, N);

    // Device 内存
    float *d_input, *d_output;
    cudaMalloc(&d_input,  total * sizeof(float));
    cudaMalloc(&d_output, total * sizeof(float));
    cudaMemcpy(d_input, h_input, total * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    // Grid = M 个 block，每个 block 1 个 thread（naive，后面会改）
    softmax_naive<<<M, 1>>>(d_input, d_output, M, N);
    cudaDeviceSynchronize();

    // 拷回验证
    cudaMemcpy(h_output, d_output, total * sizeof(float), cudaMemcpyDeviceToHost);

    if (verify(h_ref, h_output, total)) {
        printf("✅ Naive softmax correct!\n");
    }

    // 简单计时
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < 100; i++)
        softmax_naive<<<M, 1>>>(d_input, d_output, M, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    printf("⏱  Naive: %.3f ms (avg over 100 runs)\n", ms / 100.0f);

    // V2 验证
    softmax_v2<<<M, 256>>>(d_input, d_output, M, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_output, d_output, total * sizeof(float), cudaMemcpyDeviceToHost);
    if (verify(h_ref, h_output, total)) printf("✅ V2 softmax correct!\n");

    // V2 计时
    cudaEventRecord(start);
    for (int i = 0; i < 100; i++)
        softmax_v2<<<M, 256>>>(d_input, d_output, M, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf("⏱  V2: %.3f ms (avg over 100 runs)\n", ms / 100.0f);

    softmax_v3<<<M, 256>>>(d_input, d_output, M, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_output, d_output, total * sizeof(float), cudaMemcpyDeviceToHost);
    if (verify(h_ref, h_output, total)) printf("✅ V3 softmax correct!\n");

    cudaEventRecord(start);
    for (int i = 0; i < 100; i++)
        softmax_v3<<<M, 256>>>(d_input, d_output, M, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf("⏱  V3: %.3f ms (avg over 100 runs)\n", ms / 100.0f);

    // Cleanup
    delete[] h_input; delete[] h_ref; delete[] h_output;
    cudaFree(d_input); cudaFree(d_output);
    return 0;
}