#include <cuda_runtime.h> // Подключение CUDA Runtime API
#include <iostream>       // Для вывода в консоль
#include <climits>        // Для INT_MAX и INT_MIN

// Ядро CUDA для нахождения минимума и максимума блока
__global__ void reduceMinMax(int *data, int *minOut, int *maxOut, int n) {
    __shared__ int smin[256]; // Shared память для минимума блока
    __shared__ int smax[256]; // Shared память для максимума блока

    int tid = threadIdx.x; // Индекс потока внутри блока
    int i = blockIdx.x * blockDim.x + tid; // Глобальный индекс элемента

    smin[tid] = (i < n) ? data[i] : INT_MAX; // Инициализация минимума
    smax[tid] = (i < n) ? data[i] : INT_MIN; // Инициализация максимума
    __syncthreads(); // Синхронизация потоков в блоке

    // Редукция минимума и максимума внутри блока
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            smin[tid] = min(smin[tid], smin[tid + s]); // Сравнение и редукция минимума
            smax[tid] = max(smax[tid], smax[tid + s]); // Сравнение и редукция максимума
        }
        __syncthreads(); // Синхронизация после каждой итерации
    }

    if (tid == 0) { // Первый поток блока сохраняет результаты блока
        minOut[blockIdx.x] = smin[0];
        maxOut[blockIdx.x] = smax[0];
    }
}

int main() {
    int N = 1 << 20;           // 1 048 576 элементов
    int *h = new int[N];       // Выделяем память на CPU

    for (int i = 0; i < N; i++)
        h[i] = rand() % 1000; // Инициализация случайными числами

    int *d, *dmin, *dmax;                // Указатели на память GPU
    cudaMalloc(&d, N * sizeof(int));     // Выделение памяти на GPU для массива
    cudaMemcpy(d, h, N * sizeof(int), cudaMemcpyHostToDevice); // Копируем данные на GPU

    int blocks = 4096;                   // Количество блоков
    cudaMalloc(&dmin, blocks * sizeof(int)); // Память для частичных минимумов
    cudaMalloc(&dmax, blocks * sizeof(int)); // Память для частичных максимумов

    reduceMinMax<<<blocks, 256>>>(d, dmin, dmax, N); // Запуск ядра

    // Копируем частичные минимумы и максимумы на CPU
    int *hmin = new int[blocks];
    int *hmax = new int[blocks];
    cudaMemcpy(hmin, dmin, blocks * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(hmax, dmax, blocks * sizeof(int), cudaMemcpyDeviceToHost);

    // Финальная редукция на CPU
    int finalMin = INT_MAX;
    int finalMax = INT_MIN;
    for (int i = 0; i < blocks; i++) {
        finalMin = std::min(finalMin, hmin[i]); // Находим глобальный минимум
        finalMax = std::max(finalMax, hmax[i]); // Находим глобальный максимум
    }

    std::cout << "GPU Reduction Result:" << std::endl;
    std::cout << "Minimum value = " << finalMin << std::endl; // Вывод минимума
    std::cout << "Maximum value = " << finalMax << std::endl; // Вывод максимума

    cudaFree(d);     // Освобождаем память GPU
    cudaFree(dmin);
    cudaFree(dmax);
    delete[] h;      // Освобождаем память CPU
    delete[] hmin;
    delete[] hmax;

    return 0; // Завершение программы
}
