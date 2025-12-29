// Aruzhan Bissimbayeva ADA-2403M
// task2.cpp
// Работа с массивами и параллелизация OpenMP
//
// Практическое задание:
// 1) Создать массив из 10 000 случайных чисел
// 2) Найти минимальное и максимальное значения:
//    - последовательным алгоритмом
//    - параллельным алгоритмом с использованием OpenMP
// 3) Сравнить время выполнения обеих реализаций и сделать вывод
//
// Компиляция:
// g++ -fopenmp -O2 -std=c++17 task2.cpp -o task2.exe
//
// Запуск:
// .\task2.exe
//
// (Опционально) Задать число потоков OpenMP:
// $env:OMP_NUM_THREADS = '4'

#include <iostream>     // Подключение библиотеки для ввода и вывода данных
#include <vector>       // Подключение контейнера vector для работы с массивами
#include <random>       // Подключение современных генераторов случайных чисел
#include <chrono>       // Подключение библиотеки для измерения времени
#ifdef _OPENMP
#include <omp.h>        // Подключение OpenMP (если поддерживается компилятором)
#endif

int main() {             // Точка входа в программу
    using namespace std; // Использование стандартного пространства имён

    // Параметры задачи
    const size_t N = 10000;         // Размер массива по условию задания
    const int RAND_MIN_VAL = 1;     // Минимальное значение случайных чисел
    const int RAND_MAX_VAL = 100000;// Максимальное значение случайных чисел

    vector<int> arr(N);             // Создание массива (vector) из N элементов

    // Заполнение массива случайными числами
    random_device rd;               // Источник энтропии для инициализации генератора
    mt19937 gen(rd());              // Генератор случайных чисел Mersenne Twister
    uniform_int_distribution<int>  // Равномерное распределение в заданном диапазоне
        dist(RAND_MIN_VAL, RAND_MAX_VAL);

    for (size_t i = 0; i < N; ++i)  // Последовательный проход по массиву
        arr[i] = dist(gen);         // Заполнение массива случайными значениями

    // Последовательный поиск минимального и максимального значений
     
    auto t1 = chrono::high_resolution_clock::now(); // Начало замера времени

    int min_seq = arr[0];           // Инициализация минимума первым элементом
    int max_seq = arr[0];           // Инициализация максимума первым элементом

    for (size_t i = 1; i < N; ++i) {// Последовательный обход массива
        if (arr[i] < min_seq)       // Проверка на новый минимум
            min_seq = arr[i];       // Обновление минимума
        if (arr[i] > max_seq)       // Проверка на новый максимум
            max_seq = arr[i];       // Обновление максимума
    }

    auto t2 = chrono::high_resolution_clock::now(); // Конец замера времени
    chrono::duration<double, milli>                 // Время в миллисекундах
        time_seq = t2 - t1;

    // Параллельный поиск минимального и максимального значений (OpenMP)
    int min_par = arr[0];           // Начальное значение минимума
    int max_par = arr[0];           // Начальное значение максимума

    auto t3 = chrono::high_resolution_clock::now(); // Начало замера времени

#ifdef _OPENMP
    #pragma omp parallel for reduction(min:min_par) reduction(max:max_par)
    // Директива OpenMP:
    // - parallel for — параллельное выполнение цикла
    // - reduction — корректное объединение локальных min/max из потоков
    for (int i = 1; i < static_cast<int>(N); ++i) {
        if (arr[i] < min_par)       // Сравнение внутри потока
            min_par = arr[i];       // Обновление локального минимума
        if (arr[i] > max_par)       // Сравнение внутри потока
            max_par = arr[i];       // Обновление локального максимума
    }
#else
    // Если программа собрана без OpenMP, выполняем обычный последовательный код
    for (size_t i = 1; i < N; ++i) {
        if (arr[i] < min_par) min_par = arr[i];
        if (arr[i] > max_par) max_par = arr[i];
    }
#endif

    auto t4 = chrono::high_resolution_clock::now(); // Конец замера времени
    chrono::duration<double, milli>                 // Время в миллисекундах
        time_par = t4 - t3;

    // Вывод результатов
    cout << "Sequential version:\n";                 // Заголовок
    cout << "Min = " << min_seq                     // Минимум (последовательно)
         << ", Max = " << max_seq << '\n';           // Максимум
    cout << "Time = " << time_seq.count()           // Время выполнения
         << " ms\n\n";

    cout << "Parallel version (OpenMP):\n";          // Заголовок
    cout << "Min = " << min_par                     // Минимум (параллельно)
         << ", Max = " << max_par << '\n';           // Максимум
    cout << "Time = " << time_par.count()           // Время выполнения
         << " ms\n";

#ifdef _OPENMP
    cout << "OpenMP threads (max): "                // Вывод числа потоков
         << omp_get_max_threads() << '\n';
#else
    cout << "OpenMP not available\n";               // Если OpenMP не поддерживается
#endif

    return 0;                                       // Успешное завершение программы
}
