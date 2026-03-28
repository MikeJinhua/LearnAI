#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>
#include <float.h>


__global__ void layernorm(float* input, float* output, 
                          float* gamma, float* beta,
                          int M, int N, float eps) {
    __shared__ float smem[256];

    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= M) return;

    float* x = input  + row * N;
    float* y = output + row * N;

    // Step 1: 每个 thread 算局部 sum
    float local_sum = 0.0f;
    for (int i = tid; i < N; i += blockDim.x) {
        local_sum += x[i];
    }
    smem[tid] = local_sum;
    __syncthreads();

    // Step 2: 树形归约得到 sum → 算 mean
    // ??? (和 V2 一样，把 fmaxf 换成 +)
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem[tid] =  smem[tid] +smem[tid + stride];
        }
        __syncthreads();
    }
    float mean = smem[0] / N;
    __syncthreads();

    // Step 3: 每个 thread 算局部 variance sum
    float local_var = 0.0f;
    for (int i = tid; i < N; i += blockDim.x) {
        float diff = x[i] - mean;
        local_var += diff * diff;
    }
    smem[tid] = local_var;
    __syncthreads();

    // Step 4: 树形归约得到 var sum → 算 var
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem[tid] = smem[tid] + smem[tid + stride]; 
        }
        __syncthreads();
    }
 

    float var = smem[0] / N;

    // Step 5: normalize + scale + shift
    //y    = (x - mean) / sqrt(var + eps) * gamma + beta
    for (int i = tid; i < N; i += blockDim.x) {
        y[i] = (x[i] - mean) / sqrtf(var + eps) * gamma[i] + beta[i];  
      }
}

// CPU 参考实现
void layernorm_cpu(float* input, float* output, float* gamma, float* beta,
                   int M, int N, float eps) {
    for (int row = 0; row < M; row++) {
        float* x = input  + row * N;
        float* y = output + row * N;
        float mean = 0.0f, var = 0.0f;
        for (int i = 0; i < N; i++) mean += x[i];
        mean /= N;
        for (int i = 0; i < N; i++) var += (x[i]-mean)*(x[i]-mean);
        var /= N;
        for (int i = 0; i < N; i++)
            y[i] = (x[i]-mean) / sqrtf(var+eps) * gamma[i] + beta[i];
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
    const int M = 1024;
    const int N = 1024;
    const int total = M * N;

    float* h_input  = new float[total];
    float* h_ref    = new float[total];
    float* h_output = new float[total];
    for (int i = 0; i < total; i++) h_input[i] = (float)rand() / RAND_MAX * 10.0f - 5.0f;

    float* h_gamma = new float[N];
    float* h_beta  = new float[N];
    for (int i = 0; i < N; i++) { h_gamma[i] = 1.0f; h_beta[i] = 0.0f; }

    layernorm_cpu(h_input, h_ref, h_gamma, h_beta, M, N, 1e-5f);

    float *d_input, *d_output, *d_gamma, *d_beta;
    cudaMalloc(&d_input,  total * sizeof(float));
    cudaMalloc(&d_output, total * sizeof(float));
    cudaMalloc(&d_gamma,  N * sizeof(float));
    cudaMalloc(&d_beta,   N * sizeof(float));
    cudaMemcpy(d_input,  h_input,  total * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_gamma,  h_gamma,  N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_beta,   h_beta,   N * sizeof(float), cudaMemcpyHostToDevice);

    layernorm<<<M, 256>>>(d_input, d_output, d_gamma, d_beta, M, N, 1e-5f);
    cudaDeviceSynchronize();
    cudaMemcpy(h_output, d_output, total * sizeof(float), cudaMemcpyDeviceToHost);
    if (verify(h_ref, h_output, total)) printf("✅ LayerNorm correct!\n");

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < 100; i++)
        layernorm<<<M, 256>>>(d_input, d_output, d_gamma, d_beta, M, N, 1e-5f);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    printf("⏱  LayerNorm: %.3f ms (avg over 100 runs)\n", ms / 100.0f);

    delete[] h_input; delete[] h_ref; delete[] h_output;
    delete[] h_gamma; delete[] h_beta;
    cudaFree(d_input); cudaFree(d_output);
    cudaFree(d_gamma); cudaFree(d_beta);
    return 0;
}