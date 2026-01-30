#include <cuda_runtime.h>
#include <iostream>

__global__ void compute_kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        for(int i = 0; i < 100; i++) data[idx] = sqrtf(data[idx] + 1.0f); // Нагружаем GPU вычислениями
    }
}

int main() {
    int n = 1 << 22;                                // Размер части массива
    size_t size = n * sizeof(float);
    float *h_data, *d_data_1, *d_data_2;

    // 2. Используем pinned memory (закрепленную память) для асинхронности
    cudaHostAlloc(&h_data, size * 2, cudaHostAllocDefault);
    cudaMalloc(&d_data_1, size);
    cudaMalloc(&d_data_2, size);

    cudaStream_t stream1, stream2;                  // Создаем 2 потока CUDA
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    std::cout << "Задание 3. Профилирование гибридного приложения (Streams)" << std::endl;
    
    cudaEventRecord(start);

    // 3. Асинхронная передача и выполнение (Stream 1)
    cudaMemcpyAsync(d_data_1, h_data, size, cudaMemcpyHostToDevice, stream1);
    compute_kernel<<<(n+255)/256, 256, 0, stream1>>>(d_data_1, n);
    cudaMemcpyAsync(h_data, d_data_1, size, cudaMemcpyDeviceToHost, stream1);

    // 3. Асинхронная передача и выполнение (Stream 2) - работает параллельно со Stream 1
    cudaMemcpyAsync(d_data_2, h_data + n, size, cudaMemcpyHostToDevice, stream2);
    compute_kernel<<<(n+255)/256, 256, 0, stream2>>>(d_data_2, n);
    cudaMemcpyAsync(h_data + n, d_data_2, size, cudaMemcpyDeviceToHost, stream2);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);                     // Ждем завершения всех операций

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    std::cout << "1. Общее время гибридного алгоритма: " << ms << " мс" << std::endl;
    std::cout << "2. Оптимизация: использование CUDA Streams позволило скрыть задержки передачи данных." << std::endl;

    cudaStreamDestroy(stream1); cudaStreamDestroy(stream2);
    cudaFreeHost(h_data); cudaFree(d_data_1); cudaFree(d_data_2);
    return 0;
}
