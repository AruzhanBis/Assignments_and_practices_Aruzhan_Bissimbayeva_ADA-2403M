#include <mpi.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <ctime>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv); // Инициализация среды MPI

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Получение номера текущего процесса
    MPI_Comm_size(MPI_COMM_WORLD, &size); // Получение общего количества процессов

    int N = 1000000; // Общий размер массива
    std::vector<double> data;
    std::vector<int> sendcounts(size); // Массив количеств элементов для каждого процесса
    std::vector<int> displs(size);     // Массив смещений для Scatterv

    // Расчет порций данных для каждого процесса (учитываем остаток)
    int sum = 0;
    for (int i = 0; i < size; i++) {
        sendcounts[i] = N / size + (i < (N % size) ? 1 : 0); // Равномерное распределение остатка
        displs[i] = sum; // Смещение относительно начала массива
        sum += sendcounts[i];
    }

    if (rank == 0) {
        std::cout << " Задание 1: Среднее значение и Стандартное отклонение." << std::endl;
        std::cout << "1. Генерация массива из " << N << " элементов на rank 0..." << std::endl;
        data.resize(N);
        srand(time(0));
        for (int i = 0; i < N; i++) data[i] = rand() % 100; // Заполнение случайными числами
    }

    // Локальный буфер для каждого процесса
    std::vector<double> local_data(sendcounts[rank]);

    // [2] Распределение массива между процессами с учетом остатка
    MPI_Scatterv(data.data(), sendcounts.data(), displs.data(), MPI_DOUBLE,
                 local_data.data(), sendcounts[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double start_time = MPI_Wtime(); // Начало замера времени

    // [3] Вычисление локальных сумм
    double local_sum = 0, local_sq_sum = 0;
    for (double val : local_data) {
        local_sum += val;             // Сумма элементов
        local_sq_sum += val * val;    // Сумма квадратов
    }

    // [4] Сбор данных на процессе 0
    double total_sum, total_sq_sum;
    MPI_Reduce(&local_sum, &total_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_sq_sum, &total_sq_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    double end_time = MPI_Wtime(); // Конец замера времени

    if (rank == 0) {
        // [5] Итоговые вычисления
        double mean = total_sum / N;
        double variance = (total_sq_sum / N) - (mean * mean);
        double std_dev = sqrt(variance);

        std::cout << "2. Данные успешно собраны через MPI_Reduce" << std::endl;
        std::cout << "3. Результат: Среднее = " << mean << std::endl;
        std::cout << "4. Результат: Ст. отклонение = " << std_dev << std::endl;
        std::cout << "5. Время выполнения: " << end_time - start_time << " сек." << std::endl;
    }

    MPI_Finalize(); // Завершение работы MPI
    return 0;
}
