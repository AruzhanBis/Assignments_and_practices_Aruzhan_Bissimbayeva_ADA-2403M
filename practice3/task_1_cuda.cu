

#include <iostream>
#include <vector>
#include <cuda_runtime.h>

// Функция для проверки ошибок CUDA - завершает программу при ошибке
void cudaCheck(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {  // Если возникла ошибка CUDA
        std::cerr << "CUDA ошибка: " << msg << " - " << cudaGetErrorString(err) << std::endl;
        exit(1);  // Завершаем программу с кодом ошибки
    }
}

// 1. Ядро CUDA для сортировки одного блока данных
__global__ void sortBlock(int* data, int n, int block_size) {
    // Объявляем динамически выделяемую разделяемую память (shared memory)
    extern __shared__ int block_data[];
    
    // Получаем идентификатор текущего блока и потока
    int block_id = blockIdx.x;    // Номер блока (0, 1, 2, ...)
    int thread_id = threadIdx.x;  // Номер потока внутри блока (0, 1, ..., block_size-1)
    
    // Вычисляем границы блока данных, который обрабатывает этот блок потоков
    int start = block_id * block_size;          // Начало блока в массиве
    int end = min(start + block_size, n);       // Конец блока (не превышаем размер массива)
    int elements = end - start;                 // Количество элементов в этом блоке
    
    // Загрузка данных блока в разделяемую память (ускоренный доступ)
    if (thread_id < elements) {  // Если поток должен обрабатывать элемент
        block_data[thread_id] = data[start + thread_id];  // Копируем элемент в shared memory
    }
    __syncthreads();  // Синхронизация всех потоков блока - ждем загрузки всех данных
    
    // Сортировка блока методом пузырька (простой, но понятный алгоритм)
    for (int i = 0; i < elements - 1; i++) {              // Проходы по массиву
        for (int j = 0; j < elements - i - 1; j++) {      // Сравнение соседних элементов
            if (block_data[j] > block_data[j + 1]) {      // Если порядок неправильный
                int temp = block_data[j];                 // Меняем местами
                block_data[j] = block_data[j + 1];
                block_data[j + 1] = temp;
            }
        }
        __syncthreads();  // Синхронизация после каждого прохода
    }
    
    // Сохранение отсортированного блока обратно в глобальную память
    if (thread_id < elements) {  // Если поток обрабатывает элемент
        data[start + thread_id] = block_data[thread_id];  // Копируем из shared memory
    }
}

// 2. Ядро CUDA для слияния двух отсортированных блоков
__global__ void mergeBlocks(int* src, int* dst, int n, int block_size, int merge_step) {
    // pair_id - номер пары блоков для слияния (0, 1, ...)
    int pair_id = blockIdx.x;      // Номер блока = номер пары
    int thread_id = threadIdx.x;   // Номер потока внутри блока
    
    // Вычисляем границы двух блоков, которые нужно слить
    int block1_start = pair_id * 2 * merge_step * block_size;  // Начало первого блока
    int block1_end = min(block1_start + merge_step * block_size, n);  // Конец первого
    int block2_start = block1_end;                               // Начало второго блока
    int block2_end = min(block2_start + merge_step * block_size, n);  // Конец второго
    
    // Размеры блоков
    int size1 = block1_end - block1_start;  // Количество элементов в первом блоке
    int size2 = block2_end - block2_start;  // Количество элементов во втором блоке
    
    // Каждый поток обрабатывает один элемент из объединенного массива
    int idx = thread_id;  // Индекс элемента, который обрабатывает этот поток
    if (idx >= size1 + size2) return;  // Если поток не нужен (слишком много потоков)
    
    // Определяем позицию элемента в результирующем отсортированном массиве
    if (idx < size1) {
        // Элемент из первого блока
        int elem = src[block1_start + idx];  // Берем элемент из первого блока
        
        // Бинарный поиск: сколько элементов из второго блока меньше нашего элемента
        int left = 0, right = size2;  // Границы поиска во втором блоке
        while (left < right) {  // Бинарный поиск
            int mid = left + (right - left) / 2;  // Средний элемент
            if (src[block2_start + mid] < elem) {  // Если элемент из второго блока меньше
                left = mid + 1;  // Сдвигаем левую границу
            } else {
                right = mid;     // Сдвигаем правую границу
            }
        }
        // left - количество элементов из второго блока, которые меньше нашего элемента
        dst[block1_start + idx + left] = elem;  // Записываем элемент на правильную позицию
        
    } else {
        // Элемент из второго блока
        int elem = src[block2_start + (idx - size1)];  // Берем элемент из второго блока
        
        // Бинарный поиск: сколько элементов из первого блока <= нашему элементу
        int left = 0, right = size1;  // Границы поиска в первом блоке
        while (left < right) {  // Бинарный поиск
            int mid = left + (right - left) / 2;  // Средний элемент
            if (src[block1_start + mid] <= elem) {  // Если элемент из первого блока <=
                left = mid + 1;  // Сдвигаем левую границу
            } else {
                right = mid;     // Сдвигаем правую границу
            }
        }
        // left - количество элементов из первого блока, которые <= нашему элементу
        dst[block1_start + (idx - size1) + left] = elem;  // Записываем на правильную позицию
    }
}

// Главная функция параллельной сортировки слиянием на CUDA
void cudaMergeSort(std::vector<int>& arr) {
    int n = arr.size();  // Получаем размер массива
    if (n <= 1) return;  // Если массив пустой или из одного элемента - ничего не делаем
    
    // 1. Выделение памяти на GPU (устройстве)
    int* d_arr;   // Указатель на массив в памяти GPU
    int* d_temp;  // Указатель на временный массив в памяти GPU
    cudaCheck(cudaMalloc(&d_arr, n * sizeof(int)), "Выделение памяти d_arr");   // Выделяем память
    cudaCheck(cudaMalloc(&d_temp, n * sizeof(int)), "Выделение памяти d_temp"); // Выделяем память
    
    // 2. Копирование данных с CPU (хоста) на GPU (устройство)
    cudaCheck(cudaMemcpy(d_arr, arr.data(), n * sizeof(int), cudaMemcpyHostToDevice), 
              "Копирование на GPU");  // Копируем исходный массив
    
    // 3. Настройка параметров параллельного выполнения
    int block_size = 4;  // Количество элементов в одном блоке данных (можно менять)
    int num_blocks = (n + block_size - 1) / block_size;  // Количество блоков данных
    
    // Информация о параметрах
    std::cout << "Этап 1: Разделение на " << num_blocks << " блоков по " << block_size << " элементов" << std::endl;
    
    // 4. Этап 1: Параллельная сортировка всех блоков
    std::cout << "Этап 2: Параллельная сортировка блоков" << std::endl;
    // Запускаем ядро: num_blocks блоков потоков, block_size потоков в каждом блоке
    sortBlock<<<num_blocks, block_size, block_size * sizeof(int)>>>(d_arr, n, block_size);
    cudaCheck(cudaGetLastError(), "Ошибка sortBlock");  // Проверяем ошибки ядра
    cudaDeviceSynchronize();  // Ждем завершения всех потоков
    
    // 5. Показываем промежуточный результат (после сортировки блоков)
    std::vector<int> intermediate(n);  // Временный массив на CPU
    cudaMemcpy(intermediate.data(), d_arr, n * sizeof(int), cudaMemcpyDeviceToHost);  // Копируем
    
    std::cout << "Результат после сортировки блоков:" << std::endl;
    std::cout << "  ";
    for (int i = 0; i < n; i++) {
        std::cout << intermediate[i] << " ";  // Выводим элемент
        if ((i + 1) % block_size == 0 && i + 1 < n) std::cout << "| ";  // Разделитель между блоками
    }
    std::cout << std::endl;
    
    // 6. Этап 3: Итеративное слияние блоков по парам
    std::cout << "Этап 3: Слияние блоков по парам" << std::endl;
    
    // merge_step - текущий размер сливаемых блоков (увеличивается в 2 раза на каждом шаге)
    for (int merge_step = 1; merge_step < num_blocks; merge_step *= 2) {
        int num_pairs = (num_blocks + 1) / 2;  // Количество пар блоков для слияния
        int threads = min(512, 2 * merge_step * block_size);  // Потоков на пару (не более 512)
        
        std::cout << "  Шаг " << merge_step << ": слияние " << num_pairs << " пар" << std::endl;
        
        // Запускаем ядро слияния
        mergeBlocks<<<num_pairs, threads>>>(d_arr, d_temp, n, block_size, merge_step);
        cudaCheck(cudaGetLastError(), "Ошибка mergeBlocks");  // Проверяем ошибки
        cudaDeviceSynchronize();  // Ждем завершения
        
        // Копируем результат обратно для следующей итерации
        cudaMemcpy(d_arr, d_temp, n * sizeof(int), cudaMemcpyDeviceToDevice);
    }
    
    // 7. Копирование финального результата с GPU на CPU
    cudaCheck(cudaMemcpy(arr.data(), d_arr, n * sizeof(int), cudaMemcpyDeviceToHost),
              "Копирование результата");  // Копируем отсортированный массив
    
    // 8. Освобождение памяти GPU
    cudaFree(d_arr);   // Освобождаем память основного массива
    cudaFree(d_temp);  // Освобождаем память временного массива
}

// Функция проверки правильности сортировки массива
bool isSorted(const std::vector<int>& arr) {
    // Проверяем каждый элемент с предыдущим
    for (size_t i = 1; i < arr.size(); i++) {
        if (arr[i] < arr[i - 1]) {  // Если нарушен порядок сортировки
            std::cout << "Найдена ошибка: " << arr[i-1] << " > " << arr[i] << std::endl;
            return false;  // Массив не отсортирован
        }
    }
    return true;  // Все элементы в правильном порядке
}

// Главная функция программы
int main() {
    // Вывод заголовка программы
    std::cout << "Реализация параллельной сортировки слиянием на CUDA" << std::endl;
    
    // 1. Создание тестового массива (16 элементов)
    std::vector<int> array = {38, 27, 43, 3, 9, 82, 10, 15, 7, 22, 56, 41, 18, 33, 29, 5};
    
    // 2. Вывод исходного массива
    std::cout << "\nИсходный массив:" << std::endl;
    for (int i = 0; i < array.size(); i++) {
        std::cout << array[i] << " ";  // Выводим каждый элемент
    }
    std::cout << std::endl;
    
    // 3. Запуск алгоритма сортировки
    std::cout << "\n Выполнение алгоритма:" << std::endl;
    cudaMergeSort(array);  // Вызов функции CUDA сортировки
    
    // 4. Вывод результата
    std::cout << "\n Результат:" << std::endl;
    std::cout << "Отсортированный массив:" << std::endl;
    for (int i = 0; i < array.size(); i++) {
        std::cout << array[i] << " ";  // Выводим отсортированный массив
    }
    std::cout << std::endl;
    
    // 5. Проверка правильности сортировки
    std::cout << "\n ПРОВЕРКА " << std::endl;
    if (isSorted(array)) {  // Если массив отсортирован
        std::cout << " Массив успешно отсортирован!" << std::endl;
    } else {  // Если есть ошибки
        std::cout << "Ошибка в сортировке!" << std::endl;
    }
    
    // 6. Завершение программы
    std::cout << "Алгоритм выполнен успешно!" << std::endl;
    
    return 0;  // Успешное завершение программы
}
