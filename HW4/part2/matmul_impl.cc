#include <mpi.h>
#include <cstdio>
#include <cstdlib>

#define MASTER 0
#define BLOCK_SIZE 64
#define min(a, b) ((a < b) ? a : b)

typedef struct {
    int offset_row, num_rows;
} Region;

// Regions of matrix a
Region *regions;

void get_input(int *input) {
    *input = 0;
    char ch = getchar_unlocked();
    if ('0' <= ch && ch <= '9')
        *input = *input * 10 + (ch - '0');
    while (true) {
        ch = getchar_unlocked();
        if ('0' <= ch && ch <= '9')
            *input = *input * 10 + (ch - '0');
        else
            break;
    }
}

void construct_matrices(int *n_ptr, int *m_ptr, int *l_ptr, int **a_mat_ptr, int **b_mat_ptr) {
    // Get rank and size
    int world_rank, world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // Read and send n, m, and l
    if (world_rank == MASTER) {
        // Read n, m, and l
        get_input(n_ptr);
        get_input(m_ptr);
        get_input(l_ptr);

        // Send n, m, and l to other workers
        MPI_Request req;
        for (int idx = 1; idx < world_size; idx++) {
            MPI_Isend(n_ptr, 1, MPI_INT, idx, 0, MPI_COMM_WORLD, &req);
            MPI_Isend(m_ptr, 1, MPI_INT, idx, 0, MPI_COMM_WORLD, &req);
            MPI_Isend(l_ptr, 1, MPI_INT, idx, 0, MPI_COMM_WORLD, &req);
        }
    } else {
        // Receive n, m, and l from rank 0
        MPI_Status status;
        MPI_Recv(n_ptr, 1, MPI_INT, MASTER, 0, MPI_COMM_WORLD, &status);
        MPI_Recv(m_ptr, 1, MPI_INT, MASTER, 0, MPI_COMM_WORLD, &status);
        MPI_Recv(l_ptr, 1, MPI_INT, MASTER, 0, MPI_COMM_WORLD, &status);
    }

    // Wait until all processes have n, m, and l
    MPI_Barrier(MPI_COMM_WORLD);

    // Construct matrix a
    *a_mat_ptr = (int *) malloc((*n_ptr) * (*m_ptr) * sizeof(int));

    // Construct matrix b
    *b_mat_ptr = (int *) malloc((*m_ptr) * (*l_ptr) * sizeof(int));

    // Split matrix a into regions
    int num_workers = world_size - 1;
    regions = (Region *) malloc(sizeof(Region) * num_workers);
    int average_rows = (int) (*n_ptr) / num_workers;
    int remaining_rows = (*n_ptr) % num_workers;
    int offset_row = 0;
    for (int worker_idx = 0; worker_idx < num_workers; worker_idx++) {
        regions[worker_idx].num_rows = (worker_idx < remaining_rows) ? average_rows + 1 : average_rows;
        offset_row += (worker_idx > 0) ? regions[worker_idx - 1].num_rows : 0;
        regions[worker_idx].offset_row = offset_row;
    }

    // Read and send matrices
    if (world_rank == MASTER) {
        // Read matrix a
        for (int idx = 0; idx < (*n_ptr) * (*m_ptr); idx++)
            get_input(&((*a_mat_ptr)[idx]));

        // Read matrix b
        for (int idx = 0; idx < (*m_ptr) * (*l_ptr); idx++)
            get_input(&((*b_mat_ptr)[idx]));

        MPI_Request req;

        // Send region of matrix a
        for (int worker_idx = 0; worker_idx < num_workers; worker_idx++)
            MPI_Isend(&((*a_mat_ptr)[regions[worker_idx].offset_row * (*m_ptr)]),
                      regions[worker_idx].num_rows * (*m_ptr),
                      MPI_INT, worker_idx + 1, 0, MPI_COMM_WORLD, &req);

        // Send matrix b
        for (int idx = 1; idx < world_size; idx++)
            MPI_Isend(*b_mat_ptr, (*m_ptr) * (*l_ptr), MPI_INT, idx, 0, MPI_COMM_WORLD, &req);
    } else {
        MPI_Status status;

        // Receive region of matrix a from rank 0
        MPI_Recv(&((*a_mat_ptr)[regions[world_rank - 1].offset_row * (*m_ptr)]),
                 regions[world_rank - 1].num_rows * (*m_ptr),
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
        result = (int *) malloc(n * l * sizeof(int));

        // Multiply matrices
        int start_row = regions[world_rank - 1].offset_row;
        int end_row = regions[world_rank - 1].offset_row + regions[world_rank - 1].num_rows;
        int row_block, col_block, middle_block, row_idx, col_idx, middle_idx;
        for (row_block = start_row; row_block < end_row; row_block += BLOCK_SIZE)
            for (col_block = 0; col_block < l; col_block += BLOCK_SIZE)
                for (middle_block = 0; middle_block < m; middle_block += BLOCK_SIZE)
                    for (row_idx = row_block; row_idx < min(end_row, row_block + BLOCK_SIZE); row_idx++)
                        for (col_idx = col_block; col_idx < min(l, col_block + BLOCK_SIZE); col_idx++) {
                            result[row_idx * l + col_idx] = 0;
                            for (middle_idx = middle_block;
                                 middle_idx < min(m, middle_block + BLOCK_SIZE); middle_idx++)
                                result[row_idx * l + col_idx] +=
                                        a_mat[row_idx * m + middle_idx] * b_mat[middle_idx * l + col_idx];
                        }

        // Send the result to MASTER
        MPI_Win_lock(MPI_LOCK_SHARED, 0, 0, win);
        MPI_Put(&result[regions[world_rank - 1].offset_row * l],
                regions[world_rank - 1].num_rows * l,
                MPI_INT, MASTER,
                regions[world_rank - 1].offset_row * l,
                regions[world_rank - 1].num_rows * l,
                MPI_INT, win);
        MPI_Win_unlock(0, win);
        delete[] result;
    }

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