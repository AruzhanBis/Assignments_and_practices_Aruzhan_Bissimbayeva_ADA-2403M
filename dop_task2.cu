#include <cuda_runtime.h> // CUDA Runtime API
#include <iostream>       // Для ввода-вывода

// Ядро CUDA для алгоритма Blelloch Scan
__global__ void blellochScan(int *data) {
    __shared__ int temp[256]; // Shared память для сканирования
    int tid = threadIdx.x;    // Индекс потока в блоке

    temp[tid] = data[tid];    // Загружаем данные в shared память
    __syncthreads();          // Синхронизация потоков

    // Up-sweep (редукция суммы)
    for (int offset = 1; offset < 256; offset <<= 1) {
        int idx = (tid + 1) * offset * 2 - 1;
        if (idx < 256)
            temp[idx] += temp[idx - offset]; // Суммируем элементы
        __syncthreads(); // Синхронизация
    }

    if (tid == 0) temp[255] = 0; // Сброс последнего элемента для down-sweep
    __syncthreads();

    // Down-sweep (вычисление префиксной суммы)
    for (int offset = 128; offset > 0; offset >>= 1) {
        int idx = (tid + 1) * offset * 2 - 1;
        if (idx < 256) {
            int t = temp[idx - offset];
            temp[idx - offset] = temp[idx];
            temp[idx] += t; // Распространяем суммы
        }
        __syncthreads();
    }

    data[tid] = temp[tid]; // Сохраняем результат обратно в глобальную память
}

int main() {
    int h[256];
    for (int i = 0; i < 256; i++) h[i] = 1; // Инициализация массива

    int *d;
    cudaMalloc(&d, 256 * sizeof(int)); // Память GPU
    cudaMemcpy(d, h, 256 * sizeof(int), cudaMemcpyHostToDevice); // Копируем на GPU

    blellochScan<<<1, 256>>>(d); // Запуск ядра
    cudaMemcpy(h, d, 256 * sizeof(int), cudaMemcpyDeviceToHost); // Копируем результат на CPU

    std::cout << "Blelloch Scan result (first 10): ";
    for (int i = 0; i < 10; i++) std::cout << h[i] << " "; // Вывод первых 10 элементов
    std::cout << std::endl;

    cudaFree(d); // Освобождаем память GPU
    return 0;
}
