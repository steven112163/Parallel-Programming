#include <mpi.h>
#include <cstdio>
#include <cstdlib>

#define MASTER 0

typedef struct {
    int offset_row, num_rows;
} Region;

// Regions of matrix a
Region *regions;

void construct_matrices(int *n_ptr, int *m_ptr, int *l_ptr, int **a_mat_ptr, int **b_mat_ptr) {
    // Get rank and size
    int world_rank, world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // Read data
    if (world_rank == MASTER) {
        // Read n, m, and l
        scanf("%d %d %d", n_ptr, m_ptr, l_ptr);

        // Send n, m, and l to other workers
        MPI_Request req;
        for (int idx = 1; idx < world_size; idx++) {
            MPI_Isend(n_ptr, 1, MPI_INT, idx, 0, MPI_COMM_WORLD, &req);
            MPI_Isend(m_ptr, 1, MPI_INT, idx, 0, MPI_COMM_WORLD, &req);
            MPI_Isend(l_ptr, 1, MPI_INT, idx, 0, MPI_COMM_WORLD, &req);
        }

        // Construct matrix a
        *a_mat_ptr = (int *) malloc((*n_ptr) * (*m_ptr) * sizeof(int));
        for (int idx = 0; idx < (*n_ptr) * (*m_ptr); idx++)
            scanf("%d", &((*a_mat_ptr)[idx]));

        // Construct matrix b
        *b_mat_ptr = (int *) malloc((*m_ptr) * (*l_ptr) * sizeof(int));
        for (int idx = 0; idx < (*m_ptr) * (*l_ptr); idx++)
            scanf("%d", &((*b_mat_ptr)[idx]));
    } else {
        // Receive n, m, and l from rank 0
        MPI_Status status;
        MPI_Recv(n_ptr, 1, MPI_INT, MASTER, 0, MPI_COMM_WORLD, &status);
        MPI_Recv(m_ptr, 1, MPI_INT, MASTER, 0, MPI_COMM_WORLD, &status);
        MPI_Recv(l_ptr, 1, MPI_INT, MASTER, 0, MPI_COMM_WORLD, &status);

        // Construct matrix a
        *a_mat_ptr = (int *) malloc((*n_ptr) * (*m_ptr) * sizeof(int));

        // Construct matrix b
        *b_mat_ptr = (int *) malloc((*m_ptr) * (*l_ptr) * sizeof(int));
    }

    // Wait until the matrices are constructed
    MPI_Barrier(MPI_COMM_WORLD);

    // Split matrix a into regions
    regions = (Region *) malloc(sizeof(Region) * world_size);
    int average_rows = (int) (*n_ptr) / world_size;
    int remaining_rows = (*n_ptr) % world_size;
    int offset_row = 0;
    for (int idx = 0; idx < world_size; idx++) {
        regions[idx].num_rows = (idx < remaining_rows) ? average_rows + 1 : average_rows;
        offset_row += (idx > 0) ? regions[idx - 1].num_rows : 0;
        regions[idx].offset_row = offset_row;
    }

    // Send matrices
    if (world_rank == MASTER) {
        MPI_Request requests[world_size - 1];
        MPI_Status statuses[world_size - 1];

        // Send region of matrix a
        for (int idx = 1; idx < world_size; idx++)
            MPI_Isend(&((*a_mat_ptr)[regions[idx].offset_row * (*m_ptr)]),
                      regions[idx].num_rows * (*m_ptr),
                      MPI_INT, idx, 0, MPI_COMM_WORLD, &requests[idx - 1]);
        MPI_Waitall(world_size - 1, requests, statuses);


        // Send matrix b
        for (int idx = 1; idx < world_size; idx++)
            MPI_Isend(*b_mat_ptr, (*m_ptr) * (*l_ptr), MPI_INT, idx, 0, MPI_COMM_WORLD, &requests[idx - 1]);
    } else {
        MPI_Status status;

        // Receive region of matrix a from rank 0
        MPI_Recv(&((*a_mat_ptr)[regions[world_rank].offset_row * (*m_ptr)]),
                 regions[world_rank].num_rows * (*m_ptr),
                 MPI_INT, MASTER, 0, MPI_COMM_WORLD, &status);

        // Receive matrix b from rank 0
        MPI_Recv(*b_mat_ptr, (*m_ptr) * (*l_ptr), MPI_INT, MASTER, 0, MPI_COMM_WORLD, &status);
    }
}

void matrix_multiply(const int n, const int m, const int l, const int *a_mat, const int *b_mat) {
    // Get rank
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // Declare the window and the result
    MPI_Win win;
    int *result;

    // Create the window and allocate the result
    if (world_rank == MASTER) {
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
    int summation;
    for (int row_idx = regions[world_rank].offset_row;
         row_idx < regions[world_rank].offset_row + regions[world_rank].num_rows;
         row_idx++) {
        for (int col_idx = 0; col_idx < l; col_idx++) {
            summation = 0;
            for (int middle_idx = 0; middle_idx < m; middle_idx++)
                summation += a_mat[row_idx * m + middle_idx] * b_mat[middle_idx * l + col_idx];
            MPI_Put(&summation, 1, MPI_INT, MASTER, row_idx * l + col_idx, 1, MPI_INT, win);
        }
    }
    MPI_Win_unlock(0, win);

    // Wait for all processes to finish multiplying matrices
    MPI_Barrier(MPI_COMM_WORLD);

    // Print the result and destruct it
    if (world_rank == MASTER) {
        for (int idx = 0; idx < n * l; idx++) {
            printf("%d ", result[idx]);
            if (idx % l == l - 1)
                printf("\n");
        }

        MPI_Free_mem(result);
    }
}

void destruct_matrices(int *a_mat, int *b_mat) {
    delete[] a_mat;
    delete[] b_mat;
    delete[] regions;
}