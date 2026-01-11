#include <iostream>                                            // Подключение библиотеки для ввода-вывода
#include <cuda_runtime.h>                                      // Подключение CUDA Runtime
#include <device_launch_parameters.h>                          // Подключение параметров запуска устройств CUDA
#include <chrono>                                              // Подключение библиотеки для измерения времени
#include <cmath>                                               // Подключение библиотеки для математических функций

// Константы
#define N 1000000                                              // Размер массива
#define BLOCK_SIZE 256                                         // Размер блока потоков
#define NUM_ITERATIONS 100                                     // Количество итераций для замера времени

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error in " << __FILE__ << " line " << __LINE__ << ": " \
                  << cudaGetErrorString(err) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while(0)
                                                       // Конец макроса

// Ядро с коалесцированным доступом
__global__ void coalescedAccess(float* data, float factor) {  // Объявление ядра для coalesced access
    int idx = blockIdx.x * blockDim.x + threadIdx.x;          // Вычисление глобального индекса потока
    if (idx < N)                                              // Проверка выхода за границы массива
        data[idx] = data[idx] * factor;                       // Умножение элемента массива на коэффициент
}

// Ядро с некоалесцированным доступом
__global__ void nonCoalescedAccess(float* data, float factor) { // Объявление ядра для non-coalesced access
    int tid = threadIdx.x;                                     // Локальный индекс потока в блоке
    int bid = blockIdx.x;                                      // Индекс блока
    int idx = tid * gridDim.x + bid;                           // Вычисление «разреженного» индекса для некоалесцированного доступа
    if (idx < N)                                               // Проверка выхода за границы массива
        data[idx] = data[idx] * factor;                        // Умножение элемента массива
}

// Инициализация массива на хосте
void initializeArray(float* h_data) {                          // Функция для заполнения массива данными
    for (int i = 0; i < N; i++) {                              // Цикл по всем элементам
        h_data[i] = static_cast<float>(i % 100) * 0.1f;        // Присвоение значения (0, 0.1, 0.2 ... 9.9, 0, 0.1 ...)
    }                                                           // Конец цикла
}                                                               // Конец функции

// Проверка результатов
bool verifyResults(float* h_data, float* h_reference, float factor) { // Функция для проверки корректности вычислений
    for (int i = 0; i < N; i++) {                                 // Цикл по всем элементам
        float expected = h_reference[i] * factor;                // Вычисление ожидаемого результата
        if (fabs(h_data[i] - expected) > 1e-5) {                 // Проверка с допустимой погрешностью
            std::cout << "Ошибка проверки на элементе " << i      // Вывод сообщения об ошибке
                      << ": " << h_data[i] << " != " << expected << std::endl;
            return false;                                         // Возврат false, если есть ошибка
        }                                                         // Конец условия
    }                                                             // Конец цикла
    return true;                                                  // Если ошибок нет, возвращаем true
}                                                                 // Конец функции

int main() {                                                     // Главная функция программы
    std::cout << "Сравнение коалесцированного и некоалесцированного доступа к памяти" << std::endl; // Вывод заголовка
    std::cout << "Размер массива: " << N << " элементов (" 
              << N * sizeof(float) / (1024.0 * 1024.0) << " МБ)" << std::endl; // Вывод размера массива в МБ
    std::cout << "Блоков по " << BLOCK_SIZE << " потоков" << std::endl; // Вывод размера блока

    // Выделение памяти
    float* h_data = new float[N];                               // Выделение памяти на CPU под рабочий массив
    float* h_reference = new float[N];                          // Выделение памяти на CPU под массив-эталон
    float* d_data;                                              // Указатель на память GPU

    CUDA_CHECK(cudaMalloc(&d_data, N * sizeof(float)));         // Выделение памяти на GPU

    // Инициализация данных
    initializeArray(h_data);                                    // Заполнение массива h_data
    memcpy(h_reference, h_data, N * sizeof(float));            // Копирование в эталонный массив

    float factor = 2.0f;                                       // Задаем коэффициент умножения

    int gridSize = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;           // Вычисляем количество блоков

    // Тестирование разных типов доступа
    cudaEvent_t start, stop;                                    // Создаем события CUDA для точного замера времени
    CUDA_CHECK(cudaEventCreate(&start));                        // Инициализация события start
    CUDA_CHECK(cudaEventCreate(&stop));                         // Инициализация события stop

    // Массив для хранения результатов
    struct Result {                                             // Структура для хранения результата теста
        const char* name;                                       // Название теста
        float time_ms;                                          // Время выполнения
        bool correct;                                           // Корректность результатов
    };

    Result results[2];                                          // Массив из двух результатов (coalesced/non-coalesced)

    // 1. Коалесцированный доступ
    float totalTime = 0;                                        // Суммарное время
    bool correct = true;                                        // Флаг корректности

    for (int iter = 0; iter < NUM_ITERATIONS; iter++) {         // Цикл по NUM_ITERATIONS для усреднения времени
        CUDA_CHECK(cudaMemcpy(d_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice)); // Сброс данных на GPU

        CUDA_CHECK(cudaEventRecord(start));                     // Запуск события начала
        coalescedAccess<<<gridSize, BLOCK_SIZE>>>(d_data, factor); // Запуск ядра
        CUDA_CHECK(cudaDeviceSynchronize());                    // Синхронизация GPU
        CUDA_CHECK(cudaEventRecord(stop));                      // Запуск события окончания
        CUDA_CHECK(cudaEventSynchronize(stop));                 // Синхронизация события

        float milliseconds = 0;                                  // Переменная для хранения времени
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop)); // Вычисление времени
        totalTime += milliseconds;                               // Добавление времени к сумме

        if (iter == NUM_ITERATIONS - 1) {                        // На последней итерации проверяем результат
            CUDA_CHECK(cudaMemcpy(h_data, d_data, N * sizeof(float), cudaMemcpyDeviceToHost)); // Копируем обратно
            correct = verifyResults(h_data, h_reference, factor); // Проверка результатов
        }                                                         // Конец условия
    }                                                             // Конец цикла

    results[0] = {"Коалесцированный доступ", totalTime / NUM_ITERATIONS, correct}; // Сохраняем результат

    // 2. Некоалесцированный доступ
    totalTime = 0;                                               // Сброс времени
    correct = true;                                              // Сброс флага корректности

    for (int iter = 0; iter < NUM_ITERATIONS; iter++) {          // Цикл по NUM_ITERATIONS
        CUDA_CHECK(cudaMemcpy(d_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice)); // Сброс данных на GPU

        CUDA_CHECK(cudaEventRecord(start));                      // Запуск события начала
        nonCoalescedAccess<<<gridSize, BLOCK_SIZE>>>(d_data, factor); // Запуск ядра non-coalesced
        CUDA_CHECK(cudaDeviceSynchronize());                     // Синхронизация GPU
        CUDA_CHECK(cudaEventRecord(stop));                       // Запуск события окончания
        CUDA_CHECK(cudaEventSynchronize(stop));                  // Синхронизация события

        float milliseconds = 0;                                   // Время итерации
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop)); // Вычисление времени
        totalTime += milliseconds;                                // Суммирование времени

        if (iter == NUM_ITERATIONS - 1) {                         // Проверка на последней итерации
            CUDA_CHECK(cudaMemcpy(h_data, d_data, N * sizeof(float), cudaMemcpyDeviceToHost)); // Копируем обратно
            correct = verifyResults(h_data, h_reference, factor); // Проверка результатов
        }                                                          // Конец условия
    }                                                              // Конец цикла

    results[1] = {"Некоалесцированный доступ", totalTime / NUM_ITERATIONS, correct}; // Сохраняем результат

    // Вывод результатов
    std::cout << "\n Результаты сравнения:" << std::endl;         // Заголовок

    float fastest = results[0].time_ms;                            // Начальное значение для вычисления самой быстрой версии
    if (results[1].time_ms < fastest) fastest = results[1].time_ms; // Находим минимальное время

    for (int i = 0; i < 2; i++) {                                 // Цикл по двум тестам
        std::cout << "\n" << results[i].name << ":" << std::endl; // Название теста
        std::cout << "  Время: " << results[i].time_ms << " мс" << std::endl; // Вывод времени
        std::cout << "  Относительная скорость: " << results[i].time_ms / fastest << "x" << std::endl; // Относительная скорость
    }

    // Сравнение двух вариантов
    std::cout << "\n Сравнение:" << std::endl;                     // Заголовок сравнения

    float coalesced_time = results[0].time_ms;                     // Время коалесцированного доступа
    float noncoalesced_time = results[1].time_ms;                  // Время некоалесцированного доступа

    if (coalesced_time < noncoalesced_time) {                      // Если coalesced быстрее
        std::cout << "Коалесцированный доступ быстрее в " 
                  << noncoalesced_time / coalesced_time << " раз." << std::endl; // Соотношение скорости
        std::cout << "Время выполнения:" << std::endl;            // Заголовок
        std::cout << "- Коалесцированный: " << coalesced_time << " мс" << std::endl; // Вывод
        std::cout << "- Некоалесцированный: " << noncoalesced_time << " мс" << std::endl; // Вывод
        std::cout << "Разница: " << (noncoalesced_time - coalesced_time) << " мс" << std::endl; // Разница времени
    } else {                                                        // Если non-coalesced быстрее
        std::cout << "Некоалесцированный доступ быстрее в " 
                  << coalesced_time / noncoalesced_time << " раз." << std::endl; // Соотношение скорости
        std::cout << "Время выполнения:" << std::endl;             // Заголовок
        std::cout << "- Коалесцированный: " << coalesced_time << " мс" << std::endl; // Вывод
        std::cout << "- Некоалесцированный: " << noncoalesced_time << " мс" << std::endl; // Вывод
        std::cout << "Разница: " << (coalesced_time - noncoalesced_time) << " мс" << std::endl; // Разница времени
    }

    // Освобождение памяти
    delete[] h_data;                                               // Освобождение рабочего массива на CPU
    delete[] h_reference;                                          // Освобождение массива эталона на CPU
    CUDA_CHECK(cudaFree(d_data));                                  // Освобождение памяти GPU
    CUDA_CHECK(cudaEventDestroy(start));                           // Удаление события start
    CUDA_CHECK(cudaEventDestroy(stop));                            // Удаление события stop

    return 0;                                                      // Завершение программы
}
