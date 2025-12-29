// // Aruzhan Bissimbayeva ADA-2403M
// task3.cpp
// Параллельная сортировка выбором с использованием OpenMP
//
// Практическое задание:
// 1) Реализовать алгоритм сортировки выбором (selection sort) последовательно
// 2) Добавить параллелизм с помощью директив OpenMP
// 3) Сравнить производительность для массивов размером 1 000 и 10 000 элементов
//
// Компиляция (PowerShell):
// g++ -fopenmp -O2 -std=c++17 task3.cpp -o task3.exe
//
// Запуск:
// .\task3.exe
//
// (Опционально) Задать число потоков OpenMP:
// $env:OMP_NUM_THREADS = '4'

#include <iostream>     // Ввод и вывод данных
#include <vector>       // Контейнер vector
#include <random>       // Генерация случайных чисел
#include <chrono>       // Измерение времени
#ifdef _OPENMP
#include <omp.h>        // OpenMP
#endif

using namespace std;

// Последовательная сортировка выбором
// на каждом шаге ищем минимальный элемент и ставим его на своё место
// Сложность: O(N^2)

void selection_sort_sequential(vector<int>& a) {
    int n = static_cast<int>(a.size());     // Размер массива

    for (int i = 0; i < n - 1; ++i) {        // Внешний цикл по позициям
        int min_idx = i;                     // Индекс минимального элемента

        for (int j = i + 1; j < n; ++j) {    // Поиск минимума в неотсортированной части
            if (a[j] < a[min_idx])
                min_idx = j;                 // Обновление индекса минимума
        }

        swap(a[i], a[min_idx]);              // Обмен элементов
    }
}


// Параллельная сортировка выбором (OpenMP)
// Параллелизуется поиск минимума на каждом шаге
// Внешний цикл остаётся последовательным (зависимость по данным)

void selection_sort_parallel(vector<int>& a) {
    int n = static_cast<int>(a.size());     // Размер массива

    for (int i = 0; i < n - 1; ++i) {        // Внешний цикл (последовательный)
        int min_idx = i;                     // Глобальный индекс минимума

#ifdef _OPENMP
        #pragma omp parallel
        {
            int local_min_idx = min_idx;    // Локальный минимум для каждого потока

            #pragma omp for nowait
            for (int j = i + 1; j < n; ++j) {
                if (a[j] < a[local_min_idx])
                    local_min_idx = j;      // Поиск локального минимума
            }

            #pragma omp critical
            {
                if (a[local_min_idx] < a[min_idx])
                    min_idx = local_min_idx;// Обновление глобального минимума
            }
        }
#else
        // Если OpenMP недоступен - обычный последовательный поиск
        for (int j = i + 1; j < n; ++j) {
            if (a[j] < a[min_idx])
                min_idx = j;
        }
#endif

        swap(a[i], a[min_idx]);              // Обмен элементов
    }
}


// Вспомогательная функция для тестирования сортировки

void test_sort(size_t N) {
    const int RAND_MIN_VAL = 1;              // Минимальное случайное значение
    const int RAND_MAX_VAL = 100000;         // Максимальное случайное значение

    vector<int> a(N);                        // Исходный массив
    vector<int> b;                           // Копия для параллельной сортировки

    random_device rd;                        // Источник энтропии
    mt19937 gen(rd());                       // Генератор случайных чисел
    uniform_int_distribution<int> dist(RAND_MIN_VAL, RAND_MAX_VAL);

    for (size_t i = 0; i < N; ++i)            // Заполнение массива
        a[i] = dist(gen);

    b = a;                                   // Создание точной копии массива для параллельной сортировки

 
    // Последовательная сортировка
   
    auto t1 = chrono::high_resolution_clock::now(); // Замер времени начала последовательной сортировки
    selection_sort_sequential(a);            // Вызов функции последовательной сортировки выбором
    auto t2 = chrono::high_resolution_clock::now(); // Замер времени окончания последовательной сортировки
    chrono::duration<double, milli> time_seq = t2 - t1; // Вычисление времени выполнения последовательной сортировки

   
    // Параллельная сортировка
  
    auto t3 = chrono::high_resolution_clock::now(); // Замер времени начала параллельной сортировки
    selection_sort_parallel(b);              // Вызов функции параллельной сортировки выбором с OpenMP
    auto t4 = chrono::high_resolution_clock::now(); // Замер времени окончания параллельной сортировки
    chrono::duration<double, milli> time_par = t4 - t3; // Вычисление времени выполнения параллельной сортировки


    // Вывод результатов

    cout << "Размер массива: " << N << '\n'; // Вывод размера массива
    cout << "Время выполнения последовательной сортировки выбором: : "
        << time_seq.count() << " ms\n";     // Вывод времени выполнения последовательной сортировки
    cout << "Время выполнения параллельной сортировки выбором::   "
         << time_par.count() << " ms\n\n";   // Вывод времени выполнения параллельной сортировки
}

// Главная функция
// 
int main() {

    cout << "\nСортировка выбором с OpenMP\n\n";

    test_sort(1000);     // Тест для массива из 1 000 элементов
    test_sort(10000);    // Тест для массива из 10 000 элементов

#ifdef _OPENMP
    cout << "OpenMP threads (max): "
         << omp_get_max_threads() << '\n';
#else
    cout << "OpenMP not available\n";
#endif

    return 0;            // Завершение программы
}
