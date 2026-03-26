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
    // ??? 这里你来填

    // Step 3: 每个 thread 算局部 sum（exp）
    // ???

    // Step 4: 树形归约找全行 sum
    // ???

    // Step 5: normalize 写回
    float max_val = smem[0];  // 归约结果在 smem[0]
    // ???
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

    // Cleanup
    delete[] h_input; delete[] h_ref; delete[] h_output;
    cudaFree(d_input); cudaFree(d_output);
    return 0;
}