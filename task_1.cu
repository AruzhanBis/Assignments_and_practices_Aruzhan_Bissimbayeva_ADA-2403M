#include <cuda_runtime.h>                // Подключение библиотеки CUDA Runtime
#include <iostream>                     // Подключение библиотеки для ввода-вывода
#include <chrono>                       // Подключение библиотеки для измерения времени

#define N 1000000                       // Определение размера массива (1 000 000 элементов)
#define BLOCK_SIZE 256                  // Определение количества потоков в одном блоке

// Ядро 1: использование только глобальной памяти 
__global__ void multiplyGlobal(float* data, float factor) {   // Объявление CUDA-ядра для работы с глобальной памятью
    int idx = blockIdx.x * blockDim.x + threadIdx.x;          // Вычисление глобального индекса потока

    if (idx < N) {                                           // Проверка, что индекс не выходит за границы массива
        data[idx] *= factor;                                 // Умножение элемента массива на коэффициент
    }                                                        // Конец условия
}                                                            // Конец ядра multiplyGlobal

// Ядро 2: использование разделяемой (shared) памяти 
__global__ void multiplyShared(float* data, float factor) {  // Объявление CUDA-ядра с использованием shared memory
    __shared__ float shmem[BLOCK_SIZE];                      // Объявление массива в разделяемой памяти блока

    int idx = blockIdx.x * blockDim.x + threadIdx.x;         // Вычисление глобального индекса элемента
    int tid = threadIdx.x;                                   // Получение локального индекса потока в блоке

    if (idx < N)                                             // Проверка выхода за границы массива
        shmem[tid] = data[idx];                              // Копирование элемента из глобальной памяти в shared
    else                                                     // В противном случае
        shmem[tid] = 0.0f;                                   // Запись нулевого значения

    __syncthreads();                                        // Синхронизация всех потоков в блоке

    shmem[tid] *= factor;                                   // Умножение элемента в shared memory

    __syncthreads();                                        // Повторная синхронизация потоков

    if (idx < N)                                             // Проверка выхода за границы массива
        data[idx] = shmem[tid];                              // Запись результата обратно в глобальную память
}                                                            // Конец ядра multiplyShared

int main() {                                                 // Точка входа в программу
    float *h_data = new float[N];                            // Выделение памяти на CPU под массив

    for (int i = 0; i < N; i++)                              // Цикл инициализации массива
        h_data[i] = 1.0f;                                    // Заполнение массива единицами

    float *d_data;                                          // Объявление указателя на память GPU

    cudaMalloc(&d_data, N * sizeof(float));                 // Выделение памяти на GPU

    cudaMemcpy(d_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice); // Копирование данных с CPU на GPU

    int blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;         // Вычисление количества блоков

    float factor = 2.5f;                                    // Задание коэффициента умножения

    auto start1 = std::chrono::high_resolution_clock::now(); // Начало замера времени (global memory)
    multiplyGlobal<<<blocks, BLOCK_SIZE>>>(d_data, factor); // Запуск ядра с глобальной памятью
    cudaDeviceSynchronize();                                // Ожидание завершения всех потоков GPU
    auto end1 = std::chrono::high_resolution_clock::now();  // Конец замера времени
    double timeGlobal = std::chrono::duration<double, std::milli>(end1 - start1).count(); // Вычисление времени

    cudaMemcpy(d_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice); // Сброс данных на GPU

    auto start2 = std::chrono::high_resolution_clock::now(); // Начало замера времени (shared memory)
    multiplyShared<<<blocks, BLOCK_SIZE>>>(d_data, factor);  // Запуск ядра с shared memory
    cudaDeviceSynchronize();                                // Ожидание завершения всех потоков GPU
    auto end2 = std::chrono::high_resolution_clock::now();  // Конец замера времени
    double timeShared = std::chrono::duration<double, std::milli>(end2 - start2).count(); // Вычисление времени

    std::cout << "Результаты измерения времени выполнения\n"; // Вывод заголовка
    std::cout << "Размер массива: " << N << " элементов\n";  // Вывод размера массива
    std::cout << "Время выполнения с использованием только глобальной памяти: "
              << timeGlobal << " мс\n";                      // Вывод времени для global memory
    std::cout << "Время выполнения с использованием разделяемой (shared) памяти: "
              << timeShared << " мс\n";                      // Вывод времени для shared memory

    if (timeShared < timeGlobal)                            // Сравнение времен
        std::cout << "Вывод: версия с использованием разделяемой памяти работает быстрее,\n"
                  << "так как доступ к shared memory имеет меньшую задержку по сравнению\n"
                  << "с глобальной памятью GPU.\n";         // Текст вывода при преимуществе shared memory
    else                                                    // В противном случае
        std::cout << "Вывод: версия с использованием глобальной памяти оказалась быстрее\n"
                  << "для данной конфигурации, однако в большинстве случаев shared memory\n"
                  << "обеспечивает более высокую производительность.\n"; // Альтернативный вывод

    cudaFree(d_data);                                       // Освобождение памяти GPU
    delete[] h_data;                                       // Освобождение памяти CPU

    return 0;                                              // Завершение программы
}
