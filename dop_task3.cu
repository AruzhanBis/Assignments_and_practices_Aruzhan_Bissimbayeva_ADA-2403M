#include <cuda_runtime.h> // CUDA Runtime API
#include <iostream>       // Для вывода

// Ядро CUDA для редукции массива
__global__ void reduce(int *data, int *out, int n) {
    __shared__ int s[1024]; // Shared память для блока
    int tid = threadIdx.x;  // Индекс потока в блоке
    int i = blockIdx.x * blockDim.x + tid; // Глобальный индекс
    s[tid] = (i < n) ? data[i] : 0;       // Загружаем данные или 0, если вышли за предел
    __syncthreads();

    // Редукция суммы в блоке
    for (int s2 = blockDim.x / 2; s2 > 0; s2 >>= 1) {
        if (tid < s2)
            s[tid] += s[tid + s2]; // Суммируем элементы
        __syncthreads();
    }

    if (tid == 0)
        out[blockIdx.x] = s[0]; // Сохраняем частичную сумму блока
}

int main() {
    int N = 1 << 20;           // Размер массива
    int *h = new int[N];       // Массив на CPU
    for (int i = 0; i < N; i++) h[i] = 1; // Инициализация единицами

    int *d, *dout;
    cudaMalloc(&d, N * sizeof(int)); // Память GPU
    cudaMemcpy(d, h, N * sizeof(int), cudaMemcpyHostToDevice); // Копируем на GPU

    for (int blockSize : {128, 256, 512, 1024}) { // Тестируем разные размеры блока
        int blocks = (N + blockSize - 1) / blockSize; // Вычисляем количество блоков
        cudaMalloc(&dout, blocks * sizeof(int)); // Память для частичных сумм

        cudaEvent_t start, stop;
        cudaEventCreate(&start); // Создаем событие для замера времени
        cudaEventCreate(&stop);

        cudaEventRecord(start);
        reduce<<<blocks, blockSize>>>(d, dout, N); // Запуск ядра
        cudaEventRecord(stop);
        cudaEventSynchronize(stop); // Ждем завершения

        float time;
        cudaEventElapsedTime(&time, start, stop); // Замеряем время
        std::cout << "Block size " << blockSize << ": " << time << " ms" << std::endl;

        cudaFree(dout); // Освобождаем память
    }

    cudaFree(d);
    delete[] h;
    return 0;
}
