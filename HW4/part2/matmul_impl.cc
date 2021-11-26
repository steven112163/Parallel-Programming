#include <mpi.h>
#include <iostream>
#include <cstdlib>
#include <cmath>

typedef struct {
    int n, m, l;
} Arguments;

void construct_matrices(int *n_ptr, int *m_ptr, int *l_ptr, int **a_mat_ptr, int **b_mat_ptr) {
    // Get rank and size
    int world_rank, world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // Setup new data type
    MPI_Datatype arguments_type, old_types[1] = {MPI_INT};
    int block_counts[1] = {3};
    MPI_Aint offsets[1] = {0};
    MPI_Type_create_struct(1, block_counts, offsets, old_types, &arguments_type);
    MPI_Type_commit(&arguments_type);

    if (world_rank == 0) {
        // Read n, m, and l
        std::cin >> *n_ptr >> *m_ptr >> *l_ptr;

        // Send n, m, and l to other workers
        Arguments arguments;
        arguments.n = *n_ptr;
        arguments.m = *m_ptr;
        arguments.l = *l_ptr;
        MPI_Request req;
        for (int idx = 1; idx < world_size; idx++)
            MPI_Isend(&arguments, 1, arguments_type, idx, 0, MPI_COMM_WORLD, &req);

        // Construct matrix a
        *a_mat_ptr = (int *) malloc((*n_ptr) * (*m_ptr) * sizeof(int));
        for (int idx = 0; idx < (*n_ptr) * (*m_ptr); idx++)
            std::cin >> (*a_mat_ptr)[idx];

        // Construct matrix b
        *b_mat_ptr = (int *) malloc((*m_ptr) * (*l_ptr) * sizeof(int));
        for (int idx = 0; idx < (*m_ptr) * (*l_ptr); idx++)
            std::cin >> (*b_mat_ptr)[idx];
    } else {
        // Receive n, m, and l from rank 0
        Arguments arguments;
        MPI_Status status;
        MPI_Recv(&arguments, 1, arguments_type, 0, 0, MPI_COMM_WORLD, &status);
        *n_ptr = arguments.n;
        *m_ptr = arguments.m;
        *l_ptr = arguments.l;

        // Construct matrix a
        *a_mat_ptr = (int *) malloc((*n_ptr) * (*m_ptr) * sizeof(int));

        // Construct matrix b
        *b_mat_ptr = (int *) malloc((*m_ptr) * (*l_ptr) * sizeof(int));
    }

    // Free new data type
    MPI_Type_free(&arguments_type);

    // Wait until all processes have constructed matrices
    MPI_Barrier(MPI_COMM_WORLD);

    if (world_rank == 0) {
        MPI_Request requests[world_size - 1];
        MPI_Status statuses[world_size - 1];

        // Send matrix a
        for (int idx = 1; idx < world_size; idx++)
            MPI_Isend(*a_mat_ptr, (*n_ptr) * (*m_ptr), MPI_INT, idx, 0, MPI_COMM_WORLD, &requests[idx - 1]);
        MPI_Waitall(world_size - 1, requests, statuses);

        // Send matrix b
        for (int idx = 1; idx < world_size; idx++)
            MPI_Isend(*b_mat_ptr, (*m_ptr) * (*l_ptr), MPI_INT, idx, 0, MPI_COMM_WORLD, &requests[idx - 1]);
    } else {
        MPI_Status status;

        // Receive matrix a from rank 0
        MPI_Recv(*a_mat_ptr, (*n_ptr) * (*m_ptr), MPI_INT, 0, 0, MPI_COMM_WORLD, &status);

        // Receive matrix b from rank 0
        MPI_Recv(*b_mat_ptr, (*m_ptr) * (*l_ptr), MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
    }

    // Wait until every process has matrix a and matrix b
    MPI_Barrier(MPI_COMM_WORLD);
}

void matrix_multiply(const int n, const int m, const int l, const int *a_mat, const int *b_mat) {
    // Get rank and size
    int world_rank, world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // Declare the window and the result
    MPI_Win win;
    int *result;

    // Create the window and allocate the result
    if (world_rank == 0) {
        MPI_Alloc_mem(n * l * sizeof(int), MPI_INFO_NULL, &result);
        MPI_Win_create(result,
                       n * l * sizeof(int),
                       sizeof(int),
                       MPI_INFO_NULL,
                       MPI_COMM_WORLD,
                       &win);
    } else {
        MPI_Win_create(NULL, 0, 1, MPI_INFO_NULL, MPI_COMM_WORLD, &win);
    }

    // Multiply matrices
    MPI_Win_lock(MPI_LOCK_SHARED, 0, 0, win);
    for (int idx = world_rank; idx < n * l; idx += world_size) {
        int row_idx = std::floor(idx / l);
        int col_idx = idx % l;
        int summation = 0;
        for (int middle_idx = 0; middle_idx < m; middle_idx++)
            summation += a_mat[row_idx * m + middle_idx] * b_mat[middle_idx * l + col_idx];
        MPI_Put(&summation, 1, MPI_INT, 0, idx, 1, MPI_INT, win);
    }
    MPI_Win_unlock(0, win);

    // Wait for all processes to finished multiplying matrices
    MPI_Barrier(MPI_COMM_WORLD);

    // Print the result and destruct it
    if (world_rank == 0) {
        for (int idx = 0; idx < n * l; idx++) {
            std::cout << result[idx] << " ";
            if (idx % l == l - 1)
                std::cout << std::endl;
        }

        MPI_Free_mem(result);
    }
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