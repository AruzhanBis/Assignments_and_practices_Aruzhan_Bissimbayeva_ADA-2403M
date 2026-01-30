#include <iostream>            // Подключаем стандартную библиотеку ввода-вывода
#include <vector>              // Подключаем библиотеку для работы с динамическими массивами (векторами)
#include <omp.h>               // Подключаем библиотеку OpenMP для параллельных вычислений
#include <cmath>               // Подключаем математическую библиотеку для функции pow (степень)
#include <iomanip>             // Подключаем библиотеку для форматированного вывода (установка точности)

int main() {
    const int N = 20000000;    // Определяем размер массива (20 миллионов элементов)
    std::vector<double> arr(N, 1.5); // Создаем массив и заполняем его значением 1.5
    
    double t1 = 0;             // Переменная для хранения времени выполнения на 1 потоке
    double tn = 0;             // Переменная для хранения времени выполнения на n потоках
    int max_threads = 4;       // Установим количество потоков для параллельного теста

    std::cout << "Задание 1. Анализ производительности CPU (OpenMP)" << std::endl;

    // --- ЭТАП 1: ПОСЛЕДОВАТЕЛЬНОЕ ВЫПОЛНЕНИЕ (1 поток) ---
    std::cout << "1. Выполнение базовой последовательной версии..." << std::endl;
    omp_set_num_threads(1);    // Принудительно устанавливаем 1 поток
    double start = omp_get_wtime(); // Запоминаем время начала

    double sum = 0;
    #pragma omp parallel for reduction(+:sum) // Параллельная директива (здесь с 1 потоком)
    for (int i = 0; i < N; i++) sum += arr[i]; // Считаем сумму элементов

    double mean = sum / N;     // Считаем среднее значение
    double var_sum = 0;
    #pragma omp parallel for reduction(+:var_sum) // Считаем сумму квадратов разностей
    for (int i = 0; i < N; i++) var_sum += std::pow(arr[i] - mean, 2);
    
    t1 = omp_get_wtime() - start; // Вычисляем общее время T1
    std::cout << "    Время на 1 потоке (T1): " << t1 << " сек." << std::endl;

    // --- ЭТАП 2: ПАРАЛЛЕЛЬНОЕ ВЫПОЛНЕНИЕ (n потоков) ---
    std::cout << "2. Выполнение параллельной версии (" << max_threads << " потока)..." << std::endl;
    omp_set_num_threads(max_threads); // Устанавливаем количество потоков (например, 4)
    start = omp_get_wtime();   // Запоминаем время начала

    sum = 0;
    #pragma omp parallel for reduction(+:sum) // Распараллеливаем суммирование
    for (int i = 0; i < N; i++) sum += arr[i];

    mean = sum / N;
    var_sum = 0;
    #pragma omp parallel for reduction(+:var_sum) // Распараллеливаем расчет дисперсии
    for (int i = 0; i < N; i++) var_sum += std::pow(arr[i] - mean, 2);

    tn = omp_get_wtime() - start; // Вычисляем общее время Tn
    std::cout << "    Время на " << max_threads << " потоках (Tn): " << tn << " сек." << std::endl;

    // --- ЭТАП 3: АНАЛИЗ РЕЗУЛЬТАТОВ (ЗАКОН АМДАЛА) ---
    double S = t1 / tn;        // Вычисляем реальное ускорение (Speedup)
    // Формула оценки доли параллельного кода P на основе ускорения и числа потоков:
    double P = (max_threads / (double)(max_threads - 1)) * (1.0 - (1.0 / S));
    double S_max = 1.0 / (1.0 - P); // Теоретический предел ускорения по Амдалу

    std::cout << "3. Влияние числа потоков на ускорение:" << std::endl;
    std::cout << "    Реальное ускорение (S): " << std::fixed << std::setprecision(2) << S << "x" << std::endl;
    
    std::cout << "4. Анализ в контексте закона Амдала:" << std::endl;
    std::cout << "    Доля параллельной части (P): " << P * 100 << "%" << std::endl;
    std::cout << "    Доля последовательной части (1-P): " << (1.0 - P) * 100 << "%" << std::endl;
    std::cout << "    Максимально возможное ускорение при n -> inf: " << S_max << "x" << std::endl;

    return 0; // Завершение программы
}
