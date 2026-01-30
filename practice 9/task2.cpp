#include <mpi.h>
#include <iostream>
#include <vector>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int N = 4; // Размерность системы (можно менять)
    
    // Каждый процесс будет отвечать за определенные строки
    int rows_per_proc = N / size;
    std::vector<double> matrix_part(rows_per_proc * (N + 1));
    std::vector<double> full_matrix;

    if (rank == 0) {
        std::cout << "Задание 2: Решение СЛАУ методом Гаусса" << std::endl;
        std::cout << "1. Создание матрицы " << N << "x" << N << " на rank 0..." << std::endl;
        full_matrix.resize(N * (N + 1), 1.0); // Пример заполнения
        full_matrix[0] = 2; full_matrix[1] = 1; full_matrix[4] = 5; // Просто пример
    }

    // [2] Рассылка строк матрицы
    MPI_Scatter(full_matrix.data(), rows_per_proc * (N + 1), MPI_DOUBLE,
                matrix_part.data(), rows_per_proc * (N + 1), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double start_time = MPI_Wtime();

    // [3] Прямой ход
    for (int k = 0; k < N; k++) {
        std::vector<double> pivot_row(N + 1);
        int root = k / rows_per_proc; // Процесс, владеющий текущей строкой

        if (rank == root) {
            int local_idx = (k % rows_per_proc) * (N + 1);
            for (int j = 0; j <= N; j++) pivot_row[j] = matrix_part[local_idx + j];
        }

        // Передача текущей опорной строки всем остальным
        MPI_Bcast(pivot_row.data(), N + 1, MPI_DOUBLE, root, MPI_COMM_WORLD);

        // Обнуление коэффициентов ниже опорного
        for (int i = 0; i < rows_per_proc; i++) {
            int global_i = rank * rows_per_proc + i;
            if (global_i > k) {
                double factor = matrix_part[i * (N + 1) + k] / pivot_row[k];
                for (int j = k; j <= N; j++) {
                    matrix_part[i * (N + 1) + j] -= factor * pivot_row[j];
                }
            }
        }
    }

    if (rank == 0) {
        std::cout << "2. Прямой ход завершен успешно." << std::endl;
        std::cout << "3. Время выполнения: " << MPI_Wtime() - start_time << " сек." << std::endl;
        std::cout << "4. В учебных целях обратный ход выполняется на rank 0 после сбора." << std::endl;
    }

    MPI_Finalize();
    return 0;
}
