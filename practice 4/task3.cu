#include <cuda_runtime.h>              // Библиотека для работы с CUDA
#include <device_launch_parameters.h> // Определения для threadIdx, blockIdx и т.д.
#include <iostream>                   // Ввод-вывод в C++
#include <cstdlib>                    // Функции rand(), srand()

#define N 10000                       // Размер массива
#define BLOCK_SIZE 256                // Размер блока потоков

// Пузырьковая сортировка в локальной памяти (на уровне потока)
__device__ void bubbleSortLocal(int* data, int size) {   // Функция, выполняемая на GPU в одном потоке
    for (int i = 0; i < size - 1; i++) {                  // Внешний цикл пузырьковой сортировки
        for (int j = 0; j < size - i - 1; j++) {          // Внутренний цикл сравнений
            if (data[j] > data[j + 1]) {                 // Если элементы стоят в неправильном порядке
                int tmp = data[j];                      // Временная переменная для обмена
                data[j] = data[j + 1];                  // Меняем элементы местами
                data[j + 1] = tmp;                      // Завершаем обмен
            }
        }
    }
}

// Сортировка подмассивов + подготовка к слиянию
__global__ void sortSubarrays(int* d_array, int subSize) { // Ядро GPU для сортировки подмассивов
    int tid = threadIdx.x;                                // Номер потока внутри блока
    int start = blockIdx.x * subSize;                     // Начальный индекс подмассива в глобальной памяти

    // локальный массив потока (local memory / registers)
    int local[4];                                         // Массив в локальной памяти потока

    if (tid < subSize && start + tid < N) {               // Проверка выхода за границы массива
        local[0] = d_array[start + tid];                  // Копируем элемент из глобальной памяти в локальную
        bubbleSortLocal(local, 1);                        // Сортируем локальный подмассив (1 элемент)
        d_array[start + tid] = local[0];                  // Записываем результат обратно в глобальную память
    }
}

// Параллельное слияние в shared memory
__global__ void mergeSubarrays(int* input, int* output, int size) { // Ядро слияния подмассивов
    __shared__ int sdata[BLOCK_SIZE * 2];                 // Разделяемая память для двух подмассивов

    int tid = threadIdx.x;                                // Номер потока в блоке
    int start = blockIdx.x * BLOCK_SIZE * 2;              // Начало обрабатываемого сегмента

    if (start + tid < size)                               // Копируем первую половину в shared memory
        sdata[tid] = input[start + tid];
    if (start + BLOCK_SIZE + tid < size)                  // Копируем вторую половину в shared memory
        sdata[BLOCK_SIZE + tid] = input[start + BLOCK_SIZE + tid];

    __syncthreads();                                      // Синхронизация всех потоков блока

    // Простое слияние двух отсортированных половин
    if (tid == 0) {                                      // Слияние выполняет только нулевой поток блока
        int i = 0, j = BLOCK_SIZE, k = 0;                // Индексы для двух подмассивов и выходного массива
        while (i < BLOCK_SIZE && j < 2 * BLOCK_SIZE && start + k < size) {
            if (sdata[i] < sdata[j])                    // Сравниваем элементы двух половин
                output[start + k++] = sdata[i++];       // Записываем меньший элемент
            else
                output[start + k++] = sdata[j++];       // Или второй элемент
        }
    }
}

int main() {
    int* h_array = new int[N];                           // Выделение памяти на CPU
    for (int i = 0; i < N; i++)                          // Заполнение массива случайными числами
        h_array[i] = rand() % 1000;

    int *d_array, *d_temp;                               // Указатели на память GPU
    cudaMalloc(&d_array, N * sizeof(int));               // Выделение глобальной памяти на GPU
    cudaMalloc(&d_temp, N * sizeof(int));                // Временный массив для слияния
    cudaMemcpy(d_array, h_array, N * sizeof(int), cudaMemcpyHostToDevice); // Копирование на GPU

    std::cout << "Задание 3: \n";
    std::cout << "Сортировка на GPU с использованием разных типов памяти\n\n";

    std::cout << "1. Сортировка подмассивов пузырьком (локальная память)\n";
    sortSubarrays<<<(N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_array, BLOCK_SIZE); // Запуск ядра
    cudaDeviceSynchronize();                              // Ожидание завершения GPU

    std::cout << "2. Хранение массива в глобальной памяти\n";
    std::cout << "Размер массива: " << N << " элементов\n";

    std::cout << "3. Слияние подмассивов с использованием shared memory\n\n";
    mergeSubarrays<<<(N + 2 * BLOCK_SIZE - 1) / (2 * BLOCK_SIZE), BLOCK_SIZE>>>(d_array, d_temp, N); // Слияние
    cudaDeviceSynchronize();                              // Синхронизация

    std::cout << "Сортировка и слияние завершены.\n";

    cudaFree(d_array);                                   // Освобождение памяти GPU
    cudaFree(d_temp);                                    // Освобождение временной памяти
    delete[] h_array;                                    // Освобождение памяти CPU
    return 0;                                            // Завершение программы
}
