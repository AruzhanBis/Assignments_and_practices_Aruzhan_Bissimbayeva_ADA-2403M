#include <cuda_runtime.h>              // Основная библиотека CUDA
#include <device_launch_parameters.h> // threadIdx, blockIdx и др.
#include <iostream>                   // Потоки ввода-вывода
#include <vector>                     // Контейнер vector для CPU

#define BLOCK_SIZE 256                // Количество потоков в блоке
#define REPEATS 500                   // Число повторов для усреднения времени


// --- 1. Глобальная память ---
__global__ void reduce_global(float* input, float* output, int n) { // Редукция с доступом только к global memory
    int idx = blockIdx.x * blockDim.x + threadIdx.x;                // Глобальный индекс элемента
    if (idx < n)                                                    // Проверка выхода за границы
        atomicAdd(output, input[idx]);                             // Атомарное сложение в глобальную память
}

// --- 2. Разделяемая память ---
__global__ void reduce_shared(float* input, float* output, int n) { // Редукция с использованием shared memory
    __shared__ float sdata[BLOCK_SIZE];                             // Массив в разделяемой памяти блока
    int tid = threadIdx.x;                                          // Номер потока в блоке
    int idx = blockIdx.x * blockDim.x + tid;                        // Глобальный индекс

    sdata[tid] = (idx < n) ? input[idx] : 0.0f;                    // Копирование из глобальной в shared
    __syncthreads();                                               // Синхронизация потоков

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {                 // Параллельная редукция в shared
        if (tid < s)
            sdata[tid] += sdata[tid + s];                         // Сложение пар элементов
        __syncthreads();                                          // Синхронизация после каждого шага
    }

    if (tid == 0)                                                  // Один поток записывает сумму блока
        atomicAdd(output, sdata[0]);                               // Атомарное добавление в global memory
}

// --- 3. Локальная память (регистры) ---
__global__ void reduce_local(float* input, float* output, int n) { // Редукция с использованием локальных регистров
    int idx = blockIdx.x * blockDim.x + threadIdx.x;               // Глобальный индекс
    float local_sum = 0.0f;                                        // Переменная в регистрах (local memory)
    if (idx < n)
        local_sum = input[idx];                                   // Загрузка элемента в локальную переменную

    atomicAdd(output, local_sum);                                  // Запись результата в глобальную память
}

// Функция измерения времени с усреднением
float measure(int N, void (*kernel)(float*, float*, int)) {        // Универсальная функция замера времени
    std::vector<float> h_data(N, 1.0f);                            // Массив на CPU, заполненный единицами
    float *d_data, *d_result;                                      // Указатели на память GPU
    cudaMalloc(&d_data, N * sizeof(float));                       // Выделение памяти под массив
    cudaMalloc(&d_result, sizeof(float));                         // Память под результат
    cudaMemcpy(d_data, h_data.data(), N * sizeof(float), cudaMemcpyHostToDevice); // Копирование на GPU

    dim3 block(BLOCK_SIZE);                                        // Размер блока
    dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE);                 // Количество блоков

    cudaEvent_t start, stop;                                       // CUDA-события для замера времени
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float total_ms = 0.0f;                                         // Суммарное время всех запусков

    for (int i = 0; i < REPEATS; ++i) {                            // Многократный запуск для усреднения
        cudaMemset(d_result, 0, sizeof(float));                   // Обнуление результата

        cudaEventRecord(start);                                   // Начало измерения
        kernel<<<grid, block>>>(d_data, d_result, N);             // Запуск ядра
        cudaEventRecord(stop);                                    // Конец измерения
        cudaEventSynchronize(stop);                               // Ожидание завершения

        float ms;
        cudaEventElapsedTime(&ms, start, stop);                   // Вычисление времени в мс
        total_ms += ms;                                           // Суммирование времени
    }

    cudaFree(d_data);                                             // Освобождение памяти GPU
    cudaFree(d_result);
    cudaEventDestroy(start);                                      // Удаление событий
    cudaEventDestroy(stop);

    return total_ms / REPEATS;                                    // Среднее время выполнения
}

int main() {
    int sizes[3] = {10000, 100000, 1000000};                      // Тестируемые размеры массивов

    std::cout << "N, Глобальная(ms), Разделяемая(ms), Локальная (ms)\n";

    for (int i = 0; i < 3; i++) {                                  // Цикл по размерам
        int N = sizes[i];                                         // Текущий размер массива

        float t_global = measure(N, reduce_global);               // Замер для global memory
        float t_shared = measure(N, reduce_shared);               // Замер для shared memory
        float t_local  = measure(N, reduce_local);                // Замер для local memory

        std::cout << N << ", "
                  << t_global << ", "
                  << t_shared << ", "
                  << t_local << "\n";                             // Вывод результатов в таблицу
    }

    return 0;                                                     // Завершение программы
}
