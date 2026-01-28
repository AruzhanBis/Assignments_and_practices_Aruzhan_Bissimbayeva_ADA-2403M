#include <cuda_runtime.h>          // Подключаем библиотеку CUDA для работы с GPU
#include <iostream>               // Подключаем библиотеку для вывода в консоль

#define BLOCK_SIZE 256            // Размер блока потоков (256 потоков в одном блоке)

__global__ void reductionKernel(int *input, int *blockSums, int n) { // CUDA-ядро для редукции (суммирования)
    __shared__ int sdata[BLOCK_SIZE]; // Объявляем разделяемую память для хранения данных блока

    int tid = threadIdx.x;        // Локальный индекс потока внутри блока
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // Глобальный индекс элемента в массиве

    // Загружаем данные в shared memory
    if (idx < n)                 // Если поток не выходит за границы массива
        sdata[tid] = input[idx]; // Копируем элемент из глобальной памяти в shared memory
    else
        sdata[tid] = 0;          // Если вышли за границы — записываем 0

    __syncthreads();             // Синхронизация всех потоков блока

    // Параллельная редукция в блоке
    for (int s = blockDim.x / 2; s > 0; s >>= 1) { // Шаг редукции: делим активные потоки пополам
        if (tid < s)             // Только половина потоков участвует в суммировании
            sdata[tid] += sdata[tid + s]; // Складываем пары элементов
        __syncthreads();         // Синхронизация после каждого шага
    }

    // Первый поток каждого блока сохраняет сумму блока
    if (tid == 0)                // Только поток с номером 0
        blockSums[blockIdx.x] = sdata[0]; // Записывает сумму всего блока в глобальный массив
}

int main() {
    int N = 1 << 20;             // Размер массива: 2^20 = 1 048 576 элементов
    int size = N * sizeof(int); // Размер массива в байтах

    int *h_input = new int[N];  // Выделяем память под массив на CPU
    for (int i = 0; i < N; i++) h_input[i] = 1; // Заполняем массив единицами

    int *d_input, *d_blockSums; // Указатели на массивы в памяти GPU
    cudaMalloc(&d_input, size); // Выделяем память на GPU под входной массив

    int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE; // Вычисляем количество блоков
    cudaMalloc(&d_blockSums, numBlocks * sizeof(int)); // Память под суммы блоков

    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice); // Копируем массив с CPU на GPU

    // Запуск ядра
    reductionKernel<<<numBlocks, BLOCK_SIZE>>>(d_input, d_blockSums, N); // Запускаем редукцию на GPU
    cudaDeviceSynchronize(); // Ждём завершения всех потоков

    // Копируем частичные суммы на CPU
    int *h_blockSums = new int[numBlocks]; // Массив для хранения сумм блоков на CPU
    cudaMemcpy(h_blockSums, d_blockSums, numBlocks * sizeof(int), cudaMemcpyDeviceToHost); // Копируем с GPU

    // Финальная редукция на CPU
    long long finalSum = 0;     // Переменная для общей суммы
    for (int i = 0; i < numBlocks; i++) // Суммируем результаты всех блоков
        finalSum += h_blockSums[i];

    std::cout << "GPU Reduction Sum = " << finalSum << std::endl; // Выводим итоговую сумму

    cudaFree(d_input);          // Освобождаем память GPU для входного массива
    cudaFree(d_blockSums);      // Освобождаем память GPU для сумм блоков
    delete[] h_input;           // Освобождаем память CPU для входного массива
    delete[] h_blockSums;       // Освобождаем память CPU для сумм блоков

    return 0;                  // Завершение программы
}
