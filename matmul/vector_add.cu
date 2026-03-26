#include <stdio.h>
#include <cuda_runtime.h>
#include <windows.h>

// ============================================================
// GPU 核函数（Kernel）
// __global__ 表示这个函数运行在 GPU 上，由 CPU 调用
// 参数：a、b 是输入数组，c 是输出数组，n 是数组长度
// ============================================================
__global__ void vec_add(float* a, float* b, float* c, int n) {
    // 每个 GPU 线程负责计算一个元素
    // blockIdx.x  = 当前线程块的编号
    // blockDim.x  = 每个线程块包含多少个线程
    // threadIdx.x = 当前线程在块内的编号
    // 三者组合算出这个线程对应的数组下标 i
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // 防止越界：总线程数可能比 n 多，多余的线程什么也不做
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    // 设置控制台输出为 UTF-8，防止中文乱码（Windows 专用）
    SetConsoleOutputCP(65001);

    int n = 1024;                    // 数组元素个数
    size_t size = n * sizeof(float); // 数组占用的字节数

    // ---- 第一步：在 CPU（Host）分配内存 ----
    float* h_a = (float*)malloc(size); // 输入数组 a（CPU 侧）
    float* h_b = (float*)malloc(size); // 输入数组 b（CPU 侧）
    float* h_c = (float*)malloc(size); // 输出数组 c（CPU 侧，用于接收结果）

    // 初始化输入数据：a[i] = i，b[i] = i*2
    for (int i = 0; i < n; i++) {
        h_a[i] = (float)i;
        h_b[i] = (float)(i * 2);
    }

    // ---- 第二步：在 GPU（Device）分配显存 ----
    float* d_a, * d_b, * d_c; // d_ 前缀代表 Device（GPU）上的指针
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    // ---- 第三步：把数据从 CPU 内存复制到 GPU 显存 ----
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice); // CPU -> GPU
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice); // CPU -> GPU

    // ---- 第四步：启动 GPU 核函数 ----
    int blockSize = 256;                           // 每个线程块 256 个线程
    int gridSize = (n + blockSize - 1) / blockSize; // 需要多少个线程块（向上取整）
    // <<<gridSize, blockSize>>> 是 CUDA 特有语法，指定线程网格结构
    vec_add<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);

    // ---- 第五步：把结果从 GPU 显存复制回 CPU 内存 ----
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost); // GPU -> CPU

    // 打印前 10 个结果验证正确性
    printf("验证前10个结果:\n");
    for (int i = 0; i < 10; i++) {
        printf("c[%d] = %.0f (期望: %.0f)\n", i, h_c[i], h_a[i] + h_b[i]);
    }

    // ---- 第六步：释放内存 ----
    cudaFree(d_a);  // 释放 GPU 显存
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);      // 释放 CPU 内存
    free(h_b);
    free(h_c);

    return 0;
}
