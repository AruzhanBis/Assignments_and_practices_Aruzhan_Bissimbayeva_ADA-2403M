#include <cuda_runtime.h>                // Подключение CUDA Runtime
#include <iostream>                     // Подключение библиотеки ввода-вывода
#include <chrono>                       // Подключение библиотеки для измерения времени

#define N 1000000                       // Размер массивов (1 000 000 элементов)

// CUDA-ядро для поэлементного сложения двух массивов
__global__ void addVectors(float* a, float* b, float* c) {   // Объявление ядра
    int idx = blockIdx.x * blockDim.x + threadIdx.x;         // Вычисление глобального индекса

    if (idx < N)                                            // Проверка выхода за границы
        c[idx] = a[idx] + b[idx];                           // Поэлементное сложение
}

int main() {                                                // Главная функция
    float *h_a = new float[N];                              // Выделение памяти под первый массив на CPU
    float *h_b = new float[N];                              // Выделение памяти под второй массив на CPU
    float *h_c = new float[N];                              // Выделение памяти под результат на CPU

    for (int i = 0; i < N; i++) {                           // Инициализация массивов
        h_a[i] = 1.0f;                                      // Заполнение первого массива
        h_b[i] = 2.0f;                                      // Заполнение второго массива
    }

    float *d_a, *d_b, *d_c;                                 // Указатели на память GPU

    cudaMalloc(&d_a, N * sizeof(float));                    // Выделение памяти под a на GPU
    cudaMalloc(&d_b, N * sizeof(float));                    // Выделение памяти под b на GPU
    cudaMalloc(&d_c, N * sizeof(float));                    // Выделение памяти под c на GPU

    cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice); // Копирование a на GPU
    cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice); // Копирование b на GPU

    int blockSizes[3] = {128, 256, 512};                    // Три варианта размера блока

    std::cout << "Исследование влияния размера блока потоков на производительность\n";
    std::cout << "Размер массива: " << N << " элементов\n\n";

    for (int i = 0; i < 3; i++) {                            // Цикл по трем размерам блоков
        int blockSize = blockSizes[i];                      // Текущий размер блока
        int blocks = (N + blockSize - 1) / blockSize;       // Вычисление количества блоков

        auto start = std::chrono::high_resolution_clock::now(); // Начало замера времени
        addVectors<<<blocks, blockSize>>>(d_a, d_b, d_c);   // Запуск ядра
        cudaDeviceSynchronize();                            // Синхронизация с GPU
        auto end = std::chrono::high_resolution_clock::now();   // Конец замера

        double time = std::chrono::duration<double, std::milli>(end - start).count(); // Время выполнения

        std::cout << "Размер блока: " << blockSize << " потоков\n"; // Вывод размера блока
        std::cout << "Время выполнения: " << time << " мс\n\n";     // Вывод времени
    }

    std::cout << "Вывод:\n";
    std::cout << "Производительность CUDA-программы зависит от размера блока потоков.\n";
    std::cout << "Слишком малый размер блока приводит к недостаточной загрузке GPU,\n";
    std::cout << "а слишком большой может вызывать неэффективное использование ресурсов.\n";
    std::cout << "Оптимальный размер блока обеспечивает наилучший баланс между\n";
    std::cout << "параллелизмом и использованием аппаратных ресурсов.\n";
    std::cout << "В данном эксперименте были исследованы размеры блоков 128, 256 и 512.\n";

    cudaFree(d_a);                                          // Освобождение памяти GPU
    cudaFree(d_b);                                          // Освобождение памяти GPU
    cudaFree(d_c);                                          // Освобождение памяти GPU

    delete[] h_a;                                          // Освобождение памяти CPU
    delete[] h_b;                                          // Освобождение памяти CPU
    delete[] h_c;                                          // Освобождение памяти CPU

    return 0;                                              // Завершение программы
}
