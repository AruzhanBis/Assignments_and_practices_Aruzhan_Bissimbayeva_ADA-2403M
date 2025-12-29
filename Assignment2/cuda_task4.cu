#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <climits>

#define BLOCK_SIZE 256

__global__ void blockSort(int* data, int n) {
    __shared__ int shared[BLOCK_SIZE];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    if (gid < n)
        shared[tid] = data[gid];
    else
        shared[tid] = INT_MAX;

    __syncthreads();

    for (int k = 2; k <= blockDim.x; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            int ixj = tid ^ j;
            if (ixj > tid) {
                if ((tid & k) == 0) {
                    if (shared[tid] > shared[ixj]) {
                        int tmp = shared[tid];
                        shared[tid] = shared[ixj];
                        shared[ixj] = tmp;
                    }
                } else {
                    if (shared[tid] < shared[ixj]) {
                        int tmp = shared[tid];
                        shared[tid] = shared[ixj];
                        shared[ixj] = tmp;
                    }
                }
            }
            __syncthreads();
        }
    }

    if (gid < n)
        data[gid] = shared[tid];
}

__global__ void mergeKernel(int* input, int* output, int width, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int start = tid * width * 2;
    if (start >= n) return;

    int mid = min(start + width, n);
    int end = min(start + 2 * width, n);

    int i = start, j = mid, k = start;

    while (i < mid && j < end)
        output[k++] = (input[i] < input[j]) ? input[i++] : input[j++];

    while (i < mid) output[k++] = input[i++];
    while (j < end) output[k++] = input[j++];
}

void gpuMergeSort(int* d_data, int n) {
    int* d_temp;
    cudaMalloc(&d_temp, n * sizeof(int));

    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    blockSort<<<blocks, BLOCK_SIZE>>>(d_data, n);
    cudaDeviceSynchronize();

    for (int width = BLOCK_SIZE; width < n; width *= 2) {
        int numThreads = (n + 2 * width - 1) / (2 * width);
        int threadsPerBlock = 256;
        int numBlocks = (numThreads + threadsPerBlock - 1) / threadsPerBlock;

        mergeKernel<<<numBlocks, threadsPerBlock>>>(d_data, d_temp, width, n);
        cudaDeviceSynchronize();
        std::swap(d_data, d_temp);
    }

    cudaFree(d_temp);
}

void runTest(int n) {
    std::vector<int> h_data(n);
    srand(time(nullptr));

    for (int i = 0; i < n; i++)
        h_data[i] = rand() % 100000;

    int* d_data;
    cudaMalloc(&d_data, n * sizeof(int));
    cudaMemcpy(d_data, h_data.data(), n * sizeof(int), cudaMemcpyHostToDevice);

    std::cout << "\n1. Создание входного массива\n";
    std::cout << "Размер массива: " << n << " элементов\n";

    std::cout << "\n2. Разбиение массива на подмассивы\n";
    std::cout << "Каждый подмассив обрабатывается отдельным CUDA-блоком\n";
    std::cout << "Размер подмассива (block size): " << BLOCK_SIZE << "\n";

    std::cout << "\n3. Параллельное слияние отсортированных подмассивов\n";
    std::cout << "Выполняется на GPU (CUDA)\n";

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    gpuMergeSort(d_data, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    std::cout << "\n4. Замер производительности\n";
    std::cout << "Время сортировки на GPU: " << ms << " мс\n";

    cudaFree(d_data);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    std::cout << "Сортировка на GPU с использованием CUDA\n";

    runTest(10000);
    std::cout << "\n-----------------------------------------------\n";
    runTest(100000);

    return 0;
}
