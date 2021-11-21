#include <mpi.h>
#include <iostream>
#include <cstdlib>

void construct_matrices(int *n_ptr, int *m_ptr, int *l_ptr, int **a_mat_ptr, int **b_mat_ptr) {
    // Get rank
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    if (world_rank == 0) {
        // Read n, m, and l
        std::cin >> *n_ptr >> *m_ptr >> *l_ptr;

        // Construct matrix a
        *a_mat_ptr = (int *) malloc((*n_ptr) * (*m_ptr) * sizeof(int));
        for (int idx = 0; idx < (*n_ptr) * (*m_ptr); idx++)
            std::cin >> (*a_mat_ptr)[idx];

        // Construct matrix b
        *b_mat_ptr = (int *) malloc((*m_ptr) * (*l_ptr) * sizeof(int));
        for (int idx = 0; idx < (*m_ptr) * (*l_ptr); idx++)
            std::cin >> (*b_mat_ptr)[idx];
    }

    MPI_Barrier(MPI_COMM_WORLD);
}

void matrix_multiply(const int n, const int m, const int l, const int *a_mat, const int *b_mat) {
    // Get rank and size
    int world_rank, world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
}

void destruct_matrices(int *a_mat, int *b_mat) {
    // Get rank
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    if (world_rank == 0) {
        delete[] a_mat;
        delete[] b_mat;
    }
}