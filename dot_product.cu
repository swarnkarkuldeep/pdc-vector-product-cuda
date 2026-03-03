#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <omp.h>
#include <cuda_runtime.h>

#define THREADS_PER_BLOCK 256

// =======================
// OpenMP Dot Product
// =======================
double dotProductOMP(const std::vector<double>& A,
                     const std::vector<double>& B)
{
    double sum = 0.0;
    size_t N = A.size();

    #pragma omp parallel for reduction(+:sum)
    for (size_t i = 0; i < N; i++)
        sum += A[i] * B[i];

    return sum;
}

// =======================
// CUDA Kernel
// =======================
__global__ void dotProductKernel(double* A, double* B,
                                 double* partialSum, size_t N)
{
    __shared__ double cache[THREADS_PER_BLOCK];

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int cacheIndex = threadIdx.x;

    double temp = 0.0;

    while (tid < N)
    {
        temp += A[tid] * B[tid];
        tid += blockDim.x * gridDim.x;
    }

    cache[cacheIndex] = temp;
    __syncthreads();

    // Reduction in shared memory
    int i = blockDim.x / 2;
    while (i != 0)
    {
        if (cacheIndex < i)
            cache[cacheIndex] += cache[cacheIndex + i];
        __syncthreads();
        i /= 2;
    }

    if (cacheIndex == 0)
        partialSum[blockIdx.x] = cache[0];
}

// =======================
// CUDA Dot Product
// =======================
double dotProductCUDA(const std::vector<double>& A,
                      const std::vector<double>& B,
                      float &kernelTime)
{
    size_t N = A.size();
    size_t bytes = N * sizeof(double);

    double *d_A, *d_B;
    double *d_partial;

    int blocks = 1024;
    int threads = THREADS_PER_BLOCK;

    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_partial, blocks * sizeof(double));

    // Copy input to device
    cudaMemcpy(d_A, A.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data(), bytes, cudaMemcpyHostToDevice);

    // CUDA events for kernel timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    dotProductKernel<<<blocks, threads>>>(d_A, d_B, d_partial, N);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&kernelTime, start, stop);

    // Copy partial sums back
    std::vector<double> h_partial(blocks);
    cudaMemcpy(h_partial.data(), d_partial,
               blocks * sizeof(double),
               cudaMemcpyDeviceToHost);

    double finalSum = 0.0;
    for (int i = 0; i < blocks; i++)
        finalSum += h_partial[i];

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_partial);

    return finalSum;
}

// =======================
// Main
// =======================
int main()
{
    const size_t N = 100000000; // ~100M elements

    std::vector<double> A(N);
    std::vector<double> B(N);

    std::mt19937 gen(42);
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    for (size_t i = 0; i < N; i++)
    {
        A[i] = dist(gen);
        B[i] = dist(gen);
    }

    // ===== OpenMP timing =====
    auto t0 = std::chrono::high_resolution_clock::now();
    double ompResult = dotProductOMP(A, B);
    auto t1 = std::chrono::high_resolution_clock::now();

    double ompTime =
        std::chrono::duration<double>(t1 - t0).count();

    // ===== CUDA timing (total time including transfer) =====
    float kernelTime = 0.0f;

    auto t2 = std::chrono::high_resolution_clock::now();
    double cudaResult = dotProductCUDA(A, B, kernelTime);
    auto t3 = std::chrono::high_resolution_clock::now();

    double cudaTotalTime =
        std::chrono::duration<double>(t3 - t2).count();

    // ===== Results =====
    std::cout << "OMP Result:   " << ompResult << "\n";
    std::cout << "CUDA Result:  " << cudaResult << "\n\n";

    std::cout << "OpenMP Time:  " << ompTime << " sec\n";
    std::cout << "CUDA Total Time (with transfer): "
              << cudaTotalTime << " sec\n";
    std::cout << "CUDA Kernel Time only: "
              << kernelTime / 1000.0 << " sec\n";

    std::cout << "\nSpeedup (GPU vs OMP): "
              << ompTime / cudaTotalTime << "x\n";

    return 0;
}
