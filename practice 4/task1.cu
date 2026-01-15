
#include <cuda_runtime.h>              // Подключение библиотеки CUDA Runtime
#include <device_launch_parameters.h> // Подключение описаний grid, block, thread
#include <iostream>                   // Библиотека для вывода в консоль
#include <cstdlib>                    // Стандартная библиотека C (rand, etc.)

#define N 1000000                     // Размер массива: 1 000 000 элементов
#define THREADS 256                   // Число потоков в одном CUDA-блоке

// CUDA-ядро для генерации псевдослучайных чисел
__global__ void generateRandom(int* data, int n, unsigned int seed) {

    // Вычисление глобального индекса потока
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Проверка, чтобы не выйти за границы массива
    if (idx < n) {

        // Инициализация локальной переменной генератора
        unsigned int x = seed ^ idx;  // Комбинация seed и индекса потока

        // Линейный конгруэнтный генератор (LCG)
        x = (1103515245 * x + 12345) & 0x7fffffff;

        // Запись числа в глобальную память GPU (диапазон 0..999)
        data[idx] = x % 1000;
    }
}

int main() {

    int* d_array;                     // Указатель на массив в глобальной памяти GPU
    size_t size = N * sizeof(int);    // Объём памяти в байтах

    // Выделение памяти на видеокарте
    cudaMalloc(&d_array, size);

    // Вычисление количества блоков
    int blocks = (N + THREADS - 1) / THREADS;

    // Информационный вывод
    std::cout << "Задача 1: \n";
    std::cout << "Генерация массива в глобальной памяти CUDA\n";
    std::cout << "Размер массива: " << N << " элементов\n";
    std::cout << "Блоков: " << blocks << ", Потоков в блоке: " << THREADS << "\n\n";

    // Запуск CUDA-ядра
    generateRandom<<<blocks, THREADS>>>(d_array, N, 1234);

    // Ожидание завершения всех потоков GPU
    cudaDeviceSynchronize();

    // Подтверждение завершения вычислений
    std::cout << "Массив успешно создан на GPU \n";

    // Освобождение памяти на видеокарте
    cudaFree(d_array);

    return 0;                         // Успешное завершение программы
}
