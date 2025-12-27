
// Подключаем необходимые библиотеки
#include <iostream>      // Для ввода-вывода (cout, cin, cerr)
#include <vector>        // Для использования динамического массива vector
#include <cuda_runtime.h> // Основная библиотека CUDA для работы с GPU

// Функция для проверки ошибок CUDA - завершает программу при обнаружении ошибки
void cudaCheck(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {  // Если функция CUDA вернула ошибку
        std::cerr << "CUDA Ошибка: " << msg << " - " << cudaGetErrorString(err) << std::endl;
        exit(1);  // Завершаем программу с кодом ошибки 1
    }
}

// 1. Ядро CUDA для параллельного разделения массива по опорному элементу
__global__ void partitionKernel(int* data, int n, int pivot, int* left_count, int* right_count, int* temp_left, int* temp_right) {
    // Вычисляем глобальный индекс потока: номер_блока * размер_блока + номер_потока_в_блоке
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {  // Проверяем, что индекс находится в пределах массива
        int elem = data[idx];  // Читаем элемент массива, который обрабатывает этот поток
        
        if (elem < pivot) {  // Если элемент меньше опорного
            // Атомарно увеличиваем счетчик левых элементов и получаем позицию для записи
            int pos = atomicAdd(left_count, 1);  // atomicAdd предотвращает гонки данных
            temp_left[pos] = elem;  // Записываем элемент во временный массив левой части
        } else {  // Если элемент больше или равен опорному
            // Атомарно увеличиваем счетчик правых элементов
            int pos = atomicAdd(right_count, 1);
            temp_right[pos] = elem;  // Записываем элемент во временный массив правой части
        }
    }
}

// 2. Ядро CUDA для объединения разделенных частей обратно в один массив
__global__ void mergeKernel(int* data, int* temp_left, int* temp_right, int left_count, int right_count) {
    // Вычисляем глобальный индекс потока
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < left_count) {  // Если поток должен обработать элемент левой части
        data[idx] = temp_left[idx];  // Копируем элемент из временного левого массива
    } else if (idx - left_count < right_count) {  // Если поток должен обработать элемент правой части
        data[idx] = temp_right[idx - left_count];  // Копируем элемент из временного правого массива
    }
    // Потоки с индексом >= left_count + right_count ничего не делают
}

// 3. Функция для выбора опорного элемента (стратегия "медиана трех")
int selectPivot(const std::vector<int>& arr, int start, int end) {
    // Вычисляем индексы трех элементов: первого, среднего и последнего
    int mid = start + (end - start) / 2;  // Средний индекс (защита от переполнения)
    int a = arr[start];   // Первый элемент
    int b = arr[mid];     // Средний элемент
    int c = arr[end-1];   // Последний элемент (end указывает за последний элемент)
    
    // Находим медиану трех значений (элемент, который будет между двумя другими)
    if (a < b) {  // Если a < b
        if (b < c) return b;      // a < b < c → медиана b
        else if (a < c) return c; // a < c ≤ b → медиана c
        else return a;            // c ≤ a < b → медиана a
    } else {  // Если b ≤ a
        if (a < c) return a;      // b ≤ a < c → медиана a
        else if (b < c) return c; // b < c ≤ a → медиана c
        else return b;            // c ≤ b ≤ a → медиана b
    }
}

// 4. Рекурсивная быстрая сортировка на CPU (используется для маленьких массивов)
void quickSortCPU(std::vector<int>& arr, int start, int end) {
    if (end - start <= 1) return;  // Базовый случай рекурсии: пустой или 1 элемент
    
    // Выбираем опорный элемент (средний элемент части массива)
    int pivot = arr[(start + end) / 2];
    int i = start;      // Индекс для прохода слева
    int j = end - 1;    // Индекс для прохода справа
    
    // Процесс разделения (partition): переставляем элементы относительно pivot
    while (i <= j) {
        while (arr[i] < pivot) i++;  // Ищем элемент >= pivot слева
        while (arr[j] > pivot) j--;  // Ищем элемент <= pivot справа
        if (i <= j) {  // Если индексы не пересеклись
            std::swap(arr[i], arr[j]);  // Меняем элементы местами
            i++;  // Двигаем левый индекс вправо
            j--;  // Двигаем правый индекс влево
        }
    }
    
    // Рекурсивные вызовы для левой и правой частей
    if (start < j) quickSortCPU(arr, start, j + 1);      // Сортировка левой части
    if (i < end - 1) quickSortCPU(arr, i, end);          // Сортировка правой части
}

// 5. Основная функция параллельной быстрой сортировки (гибридная CPU-GPU реализация)
void parallelQuickSort(std::vector<int>& arr) {
    int n = arr.size();  // Получаем размер массива
    if (n <= 32) {  // Если массив маленький, сортируем на CPU (эмпирическое правило)
        quickSortCPU(arr, 0, n);  // Вызываем CPU версию быстрой сортировки
        return;  // Выходим из функции
    }
    
    // Объявляем указатели на память GPU
    int* d_data;          // Основной массив на GPU
    int* d_temp_left;     // Временный массив для элементов < pivot
    int* d_temp_right;    // Временный массив для элементов >= pivot
    int* d_left_count;    // Счетчик элементов в левой части (в памяти GPU)
    int* d_right_count;   // Счетчик элементов в правой части (в памяти GPU)
    
    // Выделяем память на GPU для основного массива
    cudaCheck(cudaMalloc(&d_data, n * sizeof(int)), "cudaMalloc d_data");
    // Выделяем память для временных массивов (максимум n элементов в каждом)
    cudaCheck(cudaMalloc(&d_temp_left, n * sizeof(int)), "cudaMalloc d_temp_left");
    cudaCheck(cudaMalloc(&d_temp_right, n * sizeof(int)), "cudaMalloc d_temp_right");
    // Выделяем память для счетчиков (по одному целому числу на каждый счетчик)
    cudaCheck(cudaMalloc(&d_left_count, sizeof(int)), "cudaMalloc d_left_count");
    cudaCheck(cudaMalloc(&d_right_count, sizeof(int)), "cudaMalloc d_right_count");
    
    // Копируем данные из оперативной памяти (CPU) в память GPU
    cudaCheck(cudaMemcpy(d_data, arr.data(), n * sizeof(int), cudaMemcpyHostToDevice), "Копирование на GPU");
    
    // Выбираем опорный элемент с помощью стратегии "медиана трех"
    int pivot = selectPivot(arr, 0, n);
    
    // Инициализируем счетчики на GPU нулями
    int zero = 0;  // Значение для инициализации
    cudaCheck(cudaMemcpy(d_left_count, &zero, sizeof(int), cudaMemcpyHostToDevice), "Инициализация left_count");
    cudaCheck(cudaMemcpy(d_right_count, &zero, sizeof(int), cudaMemcpyHostToDevice), "Инициализация right_count");
    
    // Настраиваем параметры запуска ядра CUDA
    int threads = 256;  // Количество потоков в одном блоке (стандартное значение)
    int blocks = (n + threads - 1) / threads;  // Вычисляем количество блоков для покрытия всех элементов
    
    // Запускаем ядро для параллельного разделения массива на GPU
    partitionKernel<<<blocks, threads>>>(d_data, n, pivot, d_left_count, d_right_count, d_temp_left, d_temp_right);
    cudaCheck(cudaGetLastError(), "partitionKernel");  // Проверяем ошибки при запуске ядра
    cudaDeviceSynchronize();  // Ждем завершения всех потоков GPU
    
    // Копируем результаты счетчиков из памяти GPU в память CPU
    int left_count, right_count;  // Переменные для хранения счетчиков на CPU
    cudaCheck(cudaMemcpy(&left_count, d_left_count, sizeof(int), cudaMemcpyDeviceToHost), "Получение left_count");
    cudaCheck(cudaMemcpy(&right_count, d_right_count, sizeof(int), cudaMemcpyDeviceToHost), "Получение right_count");
    
    // Запускаем ядро для объединения разделенных частей обратно в один массив
    mergeKernel<<<1, left_count + right_count>>>(d_data, d_temp_left, d_temp_right, left_count, right_count);
    cudaCheck(cudaGetLastError(), "mergeKernel");  // Проверяем ошибки
    cudaDeviceSynchronize();  // Ждем завершения
    
    // Создаем временный вектор на CPU для хранения разделенного массива
    std::vector<int> partitioned(n);
    // Копируем результат из памяти GPU обратно в память CPU
    cudaCheck(cudaMemcpy(partitioned.data(), d_data, n * sizeof(int), cudaMemcpyDeviceToHost), "Копирование результата");
    
    // Освобождаем память GPU (очень важно делать это для предотвращения утечек памяти)
    cudaCheck(cudaFree(d_data), "cudaFree d_data");
    cudaCheck(cudaFree(d_temp_left), "cudaFree d_temp_left");
    cudaCheck(cudaFree(d_temp_right), "cudaFree d_temp_right");
    cudaCheck(cudaFree(d_left_count), "cudaFree d_left_count");
    cudaCheck(cudaFree(d_right_count), "cudaFree d_right_count");
    
    // Создаем подмассивы для левой и правой частей из разделенного массива
    // Левая часть: элементы с индексами от 0 до left_count-1
    std::vector<int> left_part(partitioned.begin(), partitioned.begin() + left_count);
    // Правая часть: элементы с индексами от left_count до конца
    std::vector<int> right_part(partitioned.begin() + left_count, partitioned.end());
    
    // Рекурсивно сортируем левую часть, если в ней больше одного элемента
    if (left_count > 1) parallelQuickSort(left_part);
    // Рекурсивно сортируем правую часть, если в ней больше одного элемента
    if (right_count > 1) parallelQuickSort(right_part);
    
    // Объединяем отсортированные части обратно в исходный массив
    // Копируем отсортированную левую часть
    for (int i = 0; i < left_count; i++) arr[i] = left_part[i];
    // Копируем отсортированную правую часть после левой
    for (int i = 0; i < right_count; i++) arr[left_count + i] = right_part[i];
}

// 6. Функция для проверки правильности сортировки массива
bool checkSorted(const std::vector<int>& arr) {
    // Проходим по массиву и проверяем, что каждый следующий элемент не меньше предыдущего
    for (size_t i = 1; i < arr.size(); i++) {
        if (arr[i] < arr[i - 1]) {  // Если найден элемент меньше предыдущего
            return false;  // Массив не отсортирован
        }
    }
    return true;  // Все элементы в правильном порядке
}

// 7. Главная функция программы - точка входа
int main() {
    // Выводим заголовок программы
    std::cout << "Параллельная быстрая сортировка на CUDA" << std::endl;
    
    // Создаем тестовый массив из 16 элементов
    std::vector<int> arr = {38, 27, 43, 3, 9, 82, 10, 15, 7, 22, 56, 41, 18, 33, 29, 5};
    
    // Выводим исходный массив
    std::cout << "\nИсходный массив (16 элементов):" << std::endl;
    for (int x : arr) std::cout << x << " ";  // Цикл for-each для вывода всех элементов
    std::cout << std::endl;  // Переход на новую строку
    
    // Вызываем функцию параллельной быстрой сортировки
    parallelQuickSort(arr);
    
    // Выводим отсортированный массив
    std::cout << "\nРезультат сортировки:" << std::endl;
    for (int x : arr) std::cout << x << " ";
    std::cout << std::endl;
    
    // Проверяем корректность сортировки
    std::cout << "\nПроверка: ";
    if (checkSorted(arr)) {  // Если массив отсортирован правильно
        std::cout << " Массив отсортирован корректно" << std::endl;
    } else {  // Если есть ошибки в сортировке
        std::cout << " Ошибка сортировки" << std::endl;
    }
    
    // Сообщение об успешном завершении программы
    std::cout << "Программа выполнена успешно!" << std::endl;
    
    return 0;  // Возвращаем 0 - признак успешного завершения программы
}
