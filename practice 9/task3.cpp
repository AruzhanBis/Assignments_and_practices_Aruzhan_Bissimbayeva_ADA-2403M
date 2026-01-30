#include <mpi.h>
#include <iostream>
#include <vector>
#include <algorithm>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int N = 100; // Количество вершин
    int rows_per_proc = N / size;

    if (rank == 0) std::cout << " Задание 3: Алгоритм Флойда-Уоршелла" << std::endl;

    // Локальная часть матрицы смежности
    std::vector<double> local_matrix(rows_per_proc * N, 1e9); // Инициализация бесконечностью
    for(int i=0; i<rows_per_proc; ++i) local_matrix[i * N + (rank * rows_per_proc + i)] = 0;

    double start_time = MPI_Wtime();

    // [3] Основной цикл алгоритма
    for (int k = 0; k < N; k++) {
        std::vector<double> k_row(N);
        int root = k / rows_per_proc; // Кто владеет строкой k

        if (rank == root) {
            for (int j = 0; j < N; j++) k_row[j] = local_matrix[(k % rows_per_proc) * N + j];
        }

        // [4] Рассылка k-й строки всем процессам для обновления путей
        MPI_Bcast(k_row.data(), N, MPI_DOUBLE, root, MPI_COMM_WORLD);

        for (int i = 0; i < rows_per_proc; i++) {
            for (int j = 0; j < N; j++) {
                // Релаксация ребра: поиск более короткого пути через вершину k
                local_matrix[i * N + j] = std::min(local_matrix[i * N + j], 
                                                local_matrix[i * N + k] + k_row[j]);
            }
        }
    }

    if (rank == 0) {
        std::cout << "1. Расчет кратчайших путей завершен." << std::endl;
        std::cout << "2. Время выполнения: " << MPI_Wtime() - start_time << " сек." << std::endl;
    }

    MPI_Finalize();
    return 0;
}
