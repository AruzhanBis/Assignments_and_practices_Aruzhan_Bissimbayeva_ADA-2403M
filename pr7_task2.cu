#include <cuda_runtime.h>          // Подключаем CUDA Runtime для работы с GPU
#include <iostream>               // Подключаем библиотеку для вывода в консоль

__global__ void scanKernel(int *data, int n) { // CUDA-ядро для префиксной суммы (scan)
    __shared__ int temp[256];     // Разделяемая память для хранения элементов одного блока

    int tid = threadIdx.x;        // Локальный индекс потока в блоке
    int i = blockIdx.x * blockDim.x + tid; // Глобальный индекс элемента массива

    temp[tid] = (i < n) ? data[i] : 0; // Загружаем данные в shared memory или 0, если вышли за границы
    __syncthreads();             // Синхронизация всех потоков блока

    for (int offset = 1; offset < blockDim.x; offset <<= 1) { // Итерации префиксного суммирования
        int val = 0;             // Временная переменная для хранения добавляемого значения
        if (tid >= offset)       // Если поток имеет соседа слева на расстоянии offset
            val = temp[tid - offset]; // Берем его значение
        __syncthreads();         // Синхронизация перед обновлением
        temp[tid] += val;        // Прибавляем значение соседа (формируем префиксную сумму)
        __syncthreads();         // Синхронизация после обновления
    }

    if (i < n)                   // Проверка выхода за границы массива
        data[i] = temp[tid];     // Записываем результат из shared memory обратно в глобальную память
}

int main() {
    int N = 256;                 // Размер массива (один блок)
    int h_data[256];             // Массив на CPU

    for (int i = 0; i < N; i++) h_data[i] = 1; // Заполняем массив единицами

    int *d_data;                 // Указатель на память GPU
    cudaMalloc(&d_data, N * sizeof(int)); // Выделяем память на GPU
    cudaMemcpy(d_data, h_data, N * sizeof(int), cudaMemcpyHostToDevice); // Копируем массив на GPU

    scanKernel<<<1, 256>>>(d_data, N); // Запускаем ядро (1 блок, 256 потоков)

    cudaMemcpy(h_data, d_data, N * sizeof(int), cudaMemcpyDeviceToHost); // Копируем результат обратно на CPU

    std::cout << "Prefix sum result: "; // Выводим первые элементы результата
    for (int i = 0; i < 10; i++)
        std::cout << h_data[i] << " ";  // Печатаем первые 10 значений префиксной суммы
    std::cout << std::endl;

    cudaFree(d_data);            // Освобождаем память GPU
    return 0;                   // Завершение программы
}

