#include <stdio.h>
#include <cuda_runtime.h>
#include <windows.h>

#define N 1024        // 矩阵大小 N×N
#define TILE_SIZE 16  // shared memory 分块大小

// ============================================================
// Naive版本
// 每个thread计算输出矩阵C的一个元素
// 直接从显存读取数据，没有任何缓存优化
// ============================================================
__global__ void matmul_naive(float* A, float* B, float* C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; k++) {
            // 每次循环都从显存读A和B
            // 大量重复读取，带宽浪费严重
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

// ============================================================
// Shared Memory优化版本
// 把数据分块加载到片上shared memory
// 和你熟悉的TBDR on-chip memory是同一个思路
// ============================================================
__global__ void matmul_shared(float* A, float* B, float* C, int n) {
    // 声明shared memory，整个block共享
    // 对应CS里的 groupshared
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    // 分块处理，每次加载一个TILE到shared memory
    for (int t = 0; t < n / TILE_SIZE; t++) {

        // 每个thread负责加载一个元素到shared memory
        // 从显存加载一次，block内所有thread都能用
        tileA[threadIdx.y][threadIdx.x] = A[row * n + t * TILE_SIZE + threadIdx.x];
        tileB[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * n + col];

        // 等待block内所有thread加载完毕
        // 对应CS里的 GroupMemoryBarrierWithGroupSync()
        __syncthreads();

        // 用shared memory里的数据计算，不访问显存
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }

        // 下一轮tile开始前同步
        __syncthreads();
    }

    if (row < n && col < n) {
        C[row * n + col] = sum;
    }
}

// ============================================================
// 计时工具
// ============================================================
float run_kernel(void (*kernel)(float*, float*, float*, int),
                 float* d_A, float* d_B, float* d_C,
                 dim3 grid, dim3 block, int n) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    kernel<<<grid, block>>>(d_A, d_B, d_C, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms;
}

int main() {
    SetConsoleOutputCP(65001);

    size_t size = N * N * sizeof(float);

    // CPU内存
    float* h_A = (float*)malloc(size);
    float* h_B = (float*)malloc(size);

    // 初始化
    for (int i = 0; i < N * N; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 1.0f;
    }

    // GPU显存
    float* d_A, * d_B, * d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Naive版本
    dim3 block_naive(16, 16);
    dim3 grid_naive(N / 16, N / 16);
    float ms_naive = run_kernel(matmul_naive, d_A, d_B, d_C,
                                grid_naive, block_naive, N);

    // Shared memory版本
    dim3 block_shared(TILE_SIZE, TILE_SIZE);
    dim3 grid_shared(N / TILE_SIZE, N / TILE_SIZE);
    float ms_shared = run_kernel(matmul_shared, d_A, d_B, d_C,
                                 grid_shared, block_shared, N);

    printf("矩阵大小: %d x %d\n\n", N, N);
    printf("Naive版本:         %.2f ms\n", ms_naive);
    printf("Shared memory版本: %.2f ms\n", ms_shared);
    printf("加速比:            %.2f x\n", ms_naive / ms_shared);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);

    return 0;
}