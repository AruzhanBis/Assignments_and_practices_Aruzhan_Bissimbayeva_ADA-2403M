#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>

// 1a. Эффективный (коалесцированный) доступ: потоки читают соседние элементы
__global__ void coalesced_kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // Вычисляем глобальный индекс
    if (idx < n) data[idx] *= 2.0f;                 // Каждый поток берет свой элемент подряд
}

// 1b. Неэффективный доступ: доступ со смещением (stride)
__global__ void stride_kernel(float* data, int n, int stride) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * stride; // Индекс с большим прыжком
    if (idx < n) data[idx] *= 2.0f;                             // Это заставляет контроллер памяти делать лишние запросы
}

// 3a. Оптимизация с использованием разделяемой (Shared) памяти
__global__ void shared_mem_kernel(float* data, int n) {
    __shared__ float temp[256];                     // Резервируем быструю память внутри блока
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    if (idx < n) temp[tid] = data[idx];             // Загружаем данные из глобальной памяти в shared
    __syncthreads();                                // Ждем, пока все потоки блока закончат загрузку

    temp[tid] *= 2.0f;                              // Работаем с быстрой памятью

    if (idx < n) data[idx] = temp[tid];             // Выгружаем результат обратно
}

int main() {
    int n = 1 << 24;                                // Размер массива (около 16 млн элементов)
    size_t size = n * sizeof(float);
    float *h_data = (float*)malloc(size);           // Память на CPU
    float *d_data;                                  // Память на GPU
    cudaMalloc(&d_data, size);                      // Выделяем память на видеокарте

    cudaEvent_t start, stop;                        // 2. Инструменты для замера времени
    cudaEventCreate(&start); cudaEventCreate(&stop);
    float milliseconds = 0;

    std::cout << "Задание 2. Оптимизация доступа к памяти на GPU" << std::endl;

    // ТЕСТ 1: Коалесцированный доступ
    cudaEventRecord(start);
    coalesced_kernel<<<(n + 255) / 256, 256>>>(d_data, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "1. Коалесцированный доступ: " << milliseconds << " мс" << std::endl;

    // ТЕСТ 2: Неэффективный доступ (шаг 32)
    cudaEventRecord(start);
    stride_kernel<<<(n / 32 + 255) / 256, 256>>>(d_data, n, 32);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms_stride;
    cudaEventElapsedTime(&ms_stride, start, stop);
    std::cout << "2. Неэффективный доступ (stride 32): " << ms_stride << " мс" << std::endl;

    // ТЕСТ 3: Оптимизация через Shared Memory
    cudaEventRecord(start);
    shared_mem_kernel<<<(n + 255) / 256, 256>>>(d_data, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "3. Оптимизация (Shared Memory): " << milliseconds << " мс" << std::endl;

    // 4. Выводы
    std::cout << "\nВывод: Неэффективный доступ медленнее коалесцированного в " << ms_stride / milliseconds << " раз." << std::endl;
    std::cout << "Это доказывает важность выравнивания запросов к глобальной памяти." << std::endl;

    cudaFree(d_data); free(h_data);
    return 0;
}
