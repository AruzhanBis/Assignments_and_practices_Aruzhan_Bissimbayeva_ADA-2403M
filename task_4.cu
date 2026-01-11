#include <cuda_runtime.h>        // Подключение CUDA Runtime API
#include <iostream>             // Подключение стандартного ввода-вывода
#include <chrono>               // Подключение библиотеки для измерения времени

#define N 1000000               // Размер массивов (1 миллион элементов)

// Макрос для проверки ошибок CUDA
#define CUDA_CHECK(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error in " << __FILE__ << " line " << __LINE__ << ": " \
                  << cudaGetErrorString(err) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

// CUDA-ядро для сложения массивов
__global__ void addArrays(const float* a, const float* b, float* c, int n) { // Объявление ядра
    int idx = blockIdx.x * blockDim.x + threadIdx.x;  // Вычисление глобального индекса потока
    if (idx < n) {                                   // Проверка выхода за границы массива
        c[idx] = a[idx] + b[idx];                    // Поэлементное сложение
    }                                                // Конец условия
}                                                    // Конец ядра

int main() {                                        // Главная функция программы
    std::cout << "Оптимизация конфигурации CUDA-сетки и блоков\n"; // Вывод заголовка

    float* h_a = new float[N];                      // Выделение памяти под массив a на CPU
    float* h_b = new float[N];                      // Выделение памяти под массив b на CPU
    float* h_c = new float[N];                      // Выделение памяти под массив c на CPU

    for (int i = 0; i < N; i++) {                   // Цикл инициализации массивов
        h_a[i] = static_cast<float>(i) * 0.001f;    // Заполнение массива a
        h_b[i] = static_cast<float>(i) * 0.002f;    // Заполнение массива b
    }                                               // Конец цикла

    float *d_a, *d_b, *d_c;                         // Указатели на массивы в памяти GPU
    CUDA_CHECK(cudaMalloc(&d_a, N * sizeof(float))); // Выделение памяти под a на GPU
    CUDA_CHECK(cudaMalloc(&d_b, N * sizeof(float))); // Выделение памяти под b на GPU
    CUDA_CHECK(cudaMalloc(&d_c, N * sizeof(float))); // Выделение памяти под c на GPU

    CUDA_CHECK(cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice)); // Копирование a на GPU
    CUDA_CHECK(cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice)); // Копирование b на GPU

    int blockSize_bad = 32;                         // Малый размер блока (неоптимальный)
    int gridSize_bad = (N + blockSize_bad - 1) / blockSize_bad; // Число блоков в сетке

    auto start_bad = std::chrono::high_resolution_clock::now(); // Начало замера времени
    addArrays<<<gridSize_bad, blockSize_bad>>>(d_a, d_b, d_c, N); // Запуск ядра с плохой конфигурацией
    CUDA_CHECK(cudaDeviceSynchronize());             // Ожидание завершения вычислений
    auto end_bad = std::chrono::high_resolution_clock::now();   // Конец замера
    double time_bad = std::chrono::duration<double, std::milli>(end_bad - start_bad).count(); // Время

    int blockSize_opt = 256;                        // Оптимальный размер блока
    int gridSize_opt = (N + blockSize_opt - 1) / blockSize_opt; // Число блоков

    auto start_opt = std::chrono::high_resolution_clock::now(); // Начало замера
    addArrays<<<gridSize_opt, blockSize_opt>>>(d_a, d_b, d_c, N); // Запуск ядра с оптимальной конфигурацией
    CUDA_CHECK(cudaDeviceSynchronize());             // Синхронизация
    auto end_opt = std::chrono::high_resolution_clock::now();   // Конец замера
    double time_opt = std::chrono::duration<double, std::milli>(end_opt - start_opt).count(); // Время

    CUDA_CHECK(cudaMemcpy(h_c, d_c, N * sizeof(float), cudaMemcpyDeviceToHost)); // Копирование результата на CPU

    std::cout << "Результаты выполнения:\n";        // Вывод заголовка
    std::cout << "Неоптимальная конфигурация: " << time_bad << " мс\n"; // Время плохой конфигурации
    std::cout << "Оптимальная конфигурация: " << time_opt << " мс\n";   // Время оптимальной конфигурации

    delete[] h_a;                                  // Освобождение памяти массива a
    delete[] h_b;                                  // Освобождение памяти массива b
    delete[] h_c;                                  // Освобождение памяти массива c
    CUDA_CHECK(cudaFree(d_a));                     // Освобождение памяти GPU для a
    CUDA_CHECK(cudaFree(d_b));                     // Освобождение памяти GPU для b
    CUDA_CHECK(cudaFree(d_c));                     // Освобождение памяти GPU для c

    return 0;                                      // Завершение программы
}
