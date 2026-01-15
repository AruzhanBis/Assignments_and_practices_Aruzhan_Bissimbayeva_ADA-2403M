
#include <cuda_runtime.h>              // Подключение CUDA Runtime API
#include <device_launch_parameters.h> // Описания blockIdx, threadIdx и т.д.
#include <iostream>                   // Ввод-вывод в консоль

#define N 1000000                     // Размер массива: 1 000 000 элементов
#define THREADS 256                   // Количество потоков в одном CUDA-блоке

// ------------------- Версия 1: только глобальная память -------------------
__global__ void reduceGlobal(int* input, int* output, int n) {
    int tid = threadIdx.x;            // Локальный индекс потока в блоке
    int idx = blockIdx.x * blockDim.x + tid; // Глобальный индекс элемента массива

    if (idx < n) {                   // Проверка выхода за границы массива
        atomicAdd(output, input[idx]); // Атомарное добавление к общей сумме в глобальной памяти
    }
}

// ------------------- Версия 2: глобальная + разделяемая память -------------------
__global__ void reduceShared(int* input, int* output, int n) {
    __shared__ int sdata[THREADS];   // Разделяемая память блока для частичных сумм

    int tid = threadIdx.x;           // Индекс потока внутри блока
    int idx = blockIdx.x * blockDim.x + tid; // Глобальный индекс элемента

    sdata[tid] = (idx < n) ? input[idx] : 0; // Копирование данных из глобальной памяти в shared
    __syncthreads();                 // Синхронизация всех потоков блока

    for (int s = blockDim.x / 2; s > 0; s >>= 1) { // Параллельная редукция в shared памяти
        if (tid < s)
            sdata[tid] += sdata[tid + s]; // Сложение пар элементов
        __syncthreads();             // Синхронизация после каждого шага
    }

    if (tid == 0)                   // Только нулевой поток блока
        atomicAdd(output, sdata[0]); // Добавляет частичную сумму блока в глобальную сумму
}

int main() {
    int *d_array, *d_result;         // Указатели на массив и результат в памяти GPU
    size_t size = N * sizeof(int);   // Размер массива в байтах

    cudaMalloc(&d_array, size);      // Выделение памяти под массив на GPU
    cudaMalloc(&d_result, sizeof(int)); // Выделение памяти под результат суммы

    int blocks = (N + THREADS - 1) / THREADS; // Расчёт количества CUDA-блоков

    // Инициализация массива
    cudaMemset(d_array, 1, size);    // Заполнение массива значениями (все элементы = 1)

    std::cout << "Задание 2: \n";
    std::cout << "Редукция суммы массива (N = " << N << ")\n\n";

    // -------- Версия с глобальной памятью --------
    cudaMemset(d_result, 0, sizeof(int)); // Обнуление результата
    cudaEvent_t start1, stop1;            // События для замера времени
    cudaEventCreate(&start1);             // Создание события начала
    cudaEventCreate(&stop1);              // Создание события конца

    cudaEventRecord(start1);              // Запуск таймера
    reduceGlobal<<<blocks, THREADS>>>(d_array, d_result, N); // Запуск ядра
    cudaEventRecord(stop1);               // Остановка таймера
    cudaEventSynchronize(stop1);          // Ожидание завершения ядра

    float timeGlobal;                     // Переменная для хранения времени
    cudaEventElapsedTime(&timeGlobal, start1, stop1); // Получение времени в мс

    // -------- Версия с разделяемой памятью --------
    cudaMemset(d_result, 0, sizeof(int)); // Сброс результата
    cudaEvent_t start2, stop2;            // События таймера
    cudaEventCreate(&start2);             // Создание события начала
    cudaEventCreate(&stop2);              // Создание события конца

    cudaEventRecord(start2);              // Запуск таймера
    reduceShared<<<blocks, THREADS>>>(d_array, d_result, N); // Запуск shared-редукции
    cudaEventRecord(stop2);               // Остановка таймера
    cudaEventSynchronize(stop2);          // Синхронизация

    float timeShared;                     // Переменная для времени shared-версии
    cudaEventElapsedTime(&timeShared, start2, stop2); // Время в миллисекундах

    // Вывод результатов
    std::cout << "a. Редукция с использованием только глобальной памяти\n";
    std::cout << "Время: " << timeGlobal << " мс\n\n";

    std::cout << "b. Редукция с использованием глобальной + shared памяти\n";
    std::cout << "Время: " << timeShared << " мс\n\n";

    std::cout << "Вывод: использование разделяемой памяти уменьшает число обращений\n";
    std::cout << "к глобальной памяти и ускоряет редукцию.\n";

    cudaFree(d_array);                    // Освобождение памяти массива
    cudaFree(d_result);                   // Освобождение памяти результата
    return 0;                             // Завершение программы
}
