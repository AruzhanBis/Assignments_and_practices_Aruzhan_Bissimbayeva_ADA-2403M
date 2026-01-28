
#include <cuda_runtime.h>      // Подключаем CUDA Runtime для работы с GPU
#include <iostream>           // Для вывода в консоль
#include <vector>             // Для использования std::vector на CPU
#include <chrono>             // Для измерения времени на CPU

__global__ void reduceKernel(int *data, int *out, int n) { // CUDA-ядро редукции (суммирование)
    __shared__ int s[256];    // Разделяемая память блока для частичных сумм
    int tid = threadIdx.x;   // Индекс потока внутри блока
    int i = blockIdx.x * blockDim.x + tid; // Глобальный индекс элемента массива
    s[tid] = (i < n) ? data[i] : 0; // Загружаем элемент в shared memory или 0, если за пределами
    __syncthreads();         // Синхронизация всех потоков блока

    for (int s2 = blockDim.x / 2; s2 > 0; s2 >>= 1) { // Параллельная редукция (делим шаг пополам)
        if (tid < s2)         // Только первая половина потоков работает
            s[tid] += s[tid + s2]; // Складываем пары элементов
        __syncthreads();     // Синхронизация после каждого шага
    }

    if (tid == 0)            // Первый поток блока
        out[blockIdx.x] = s[0]; // Записывает сумму блока в выходной массив
}

int main() {
    int N = 1 << 20;         // Размер массива: 2^20 = 1 048 576 элементов
    std::vector<int> h(N, 1); // Массив на CPU, заполненный единицами

    int *d, *d_out;         // Указатели на память GPU
    cudaMalloc(&d, N * sizeof(int));          // Выделяем память под входной массив на GPU
    cudaMalloc(&d_out, sizeof(int) * 4096);   // Память под суммы блоков (4096 блоков)
    cudaMemcpy(d, h.data(), N * sizeof(int), cudaMemcpyHostToDevice); // Копируем данные на GPU

    auto start_cpu = std::chrono::high_resolution_clock::now(); // Старт таймера CPU
    long long cpu_sum = 0;  // Переменная для суммы на CPU
    for (int x : h) cpu_sum += x; // Последовательное суммирование на CPU
    auto end_cpu = std::chrono::high_resolution_clock::now();   // Конец таймера CPU

    cudaEvent_t start, stop; // CUDA-события для замера времени GPU
    cudaEventCreate(&start); // Создаем событие старта
    cudaEventCreate(&stop);  // Создаем событие окончания

    cudaEventRecord(start);  // Запускаем таймер GPU
    reduceKernel<<<4096, 256>>>(d, d_out, N); // Запускаем ядро (4096 блоков по 256 потоков)
    cudaEventRecord(stop);   // Останавливаем таймер GPU
    cudaEventSynchronize(stop); // Ждем завершения ядра

    float gpu_time;         // Переменная для времени GPU
    cudaEventElapsedTime(&gpu_time, start, stop); // Вычисляем время выполнения ядра

    std::cout << "CPU sum = " << cpu_sum << std::endl; // Вывод суммы на CPU
    std::cout << "GPU reduction time = " << gpu_time << " ms" << std::endl; // Время GPU

    cudaFree(d);            // Освобождаем память входного массива на GPU
    cudaFree(d_out);        // Освобождаем память выходного массива на GPU
    return 0;               // Завершение программы
}
