/*
  AMGX_matrix_upload_all_global上传矩阵测试成功。
*/


// par_global_upload_fixed.cpp
#include <iostream>
#include <vector>
#include <cstdlib>
#include <mpi.h>
#include <amgx_c.h>
#include <cuda_runtime.h>

#define AMGX_CHECK(ans) { amgx_assert((ans)); }
inline void amgx_assert(AMGX_RC code) {
    if (code != AMGX_RC_OK) {
        char msg[512];
        AMGX_get_error_string(code, msg, 512);
        std::cerr << "AMGX error: " << msg << std::endl;
        std::exit(1);
    }
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if (size != 2) {
        if (rank == 0)
            std::cerr << "此示例需要 2 个 MPI 进程，当前进程数为 " << size << std::endl;
        MPI_Finalize();
        return 1;
    }

    cudaSetDevice(0);
    cudaFree(0);
    cudaDeviceSynchronize();
    MPI_Barrier(comm);

    AMGX_CHECK(AMGX_initialize());
    const char *config_string = R"(
        config_version=2,
        communicator=MPI,
        solver(main)=FGMRES,
        main:use_scalar_norm=1,
        main:tolerance=1e-6,
        main:max_iters=100,
        main:gmres_n_restart=10,
        main:norm=L2,
        main:convergence=RELATIVE_INI,
        main:monitor_residual=1,
        main:store_res_history=1,
        main:preconditioner(amg)=AMG,
        amg:algorithm=AGGREGATION,
        amg:max_iters=1,
        amg:selector=SIZE_8,
        amg:merge_singletons=1,
        amg:cycle=V,
        amg:smoother=BLOCK_JACOBI,
        amg:presweeps=0,
        amg:postsweeps=3,
        amg:error_scaling=0,
        amg:max_levels=100,
        amg:coarseAgenerator=LOW_DEG,
        amg:matrix_coloring_scheme=PARALLEL_GREEDY,
        amg:max_uncolored_percentage=0.05,
        amg:max_unassigned_percentage=0.05,
        amg:relaxation_factor=0.75,
        amg:coarse_solver(coarse)=FGMRES,
        amg:communicator=MPI,
        amg:min_coarse_rows=24,
        coarse:max_iters=10,
        coarse:gmres_n_restart=100,
        coarse:monitor_residual=1,
        coarse:preconditioner(coarse_pre)=MULTICOLOR_DILU,
        coarse_pre:max_iters=3,
        coarse_pre:monitor_residual=0,
        print_solve_stats=1,
        obtain_timings=1,
        monitor_residual=1
    )";
    AMGX_config_handle cfg;
    AMGX_CHECK(AMGX_config_create(&cfg, config_string));
    AMGX_resources_handle rsrc;
    int device_ids[1] = {0};
    AMGX_CHECK(AMGX_resources_create(&rsrc, cfg, &comm, 1, device_ids));

    int global_n = 4, global_nnz = 12;
    int partition_vector[4] = {0, 0, 1, 1};  // 行 0-1 在 rank 0，行 2-3 在 rank 1
    
    AMGX_matrix_handle A;
    AMGX_CHECK(AMGX_matrix_create(&A, rsrc, AMGX_mode_dDDI));
    
    // 每个进程上传自己的行（本地 CSR 格式）
    if (rank == 0) {
        int local_row_ptr[3] = {0, 4, 6};  // 行 0 有 4 个非零元，行 1 有 2 个
        int64_t local_col_idx[6] = {0, 1, 2, 3, 1, 3};  // 全局索引
        double local_vals[6] = {1, 2, 5, 7, 4, 9};      // 对应数据

        double d[2]={1,4};
    
        AMGX_CHECK(AMGX_matrix_upload_all_global(
            A,
            global_n, 2, 6,  // 本地行数=2，本地非零元=6
            1, 1,
            local_row_ptr,
            local_col_idx,
            local_vals,
            nullptr,
            1, 
            1, 
            partition_vector
        ));
    } else {  // rank == 1
        int local_row_ptr[3] = {0, 3, 6};  // 行 2 有 3 个非零元，行 3 有 3 个
        int64_t local_col_idx[6] = {0, 1, 2, 0, 2, 3};  // 全局索引
        double local_vals[6] = {10, 11, 13, 12, 15, 14}; // 对应数据
        double d[2]={13,14};
        AMGX_CHECK(AMGX_matrix_upload_all_global(
            A,
            global_n, 2, 6,  // 本地行数=2，本地非零元=6
            1, 1,
            local_row_ptr,
            local_col_idx,
            local_vals,
            nullptr,
            1, 
            1, 
            partition_vector
        ));
    }

    AMGX_vector_handle x, b;
    std::vector<double> rhs_local(2, 1.0);
    std::vector<double> x_init_local(2, 0.0);
    if(rank == 0)
    {
        rhs_local= {1.0, 1.0};
        x_init_local = {0.0, 0.0};
    }
    else
    {
        rhs_local= {1.0, 1.0};
        x_init_local = {0.0, 0.0};
    }

    AMGX_CHECK(AMGX_vector_create(&x, rsrc, AMGX_mode_dDDI));
    AMGX_CHECK(AMGX_vector_create(&b, rsrc, AMGX_mode_dDDI));
    AMGX_CHECK(AMGX_vector_bind(b, A));
    AMGX_CHECK(AMGX_vector_bind(x, A));

    AMGX_CHECK(AMGX_vector_upload(b, 2, 1, rhs_local.data()));
    AMGX_CHECK(AMGX_vector_upload(x, 2, 1, x_init_local.data()));


    AMGX_solver_handle solver;
    AMGX_CHECK(AMGX_solver_create(&solver, rsrc, AMGX_mode_dDDI, cfg));
    AMGX_CHECK(AMGX_solver_setup(solver, A));
    AMGX_CHECK(AMGX_solver_solve(solver, b, x));

    double x_local[2];
    AMGX_CHECK(AMGX_vector_download(x, x_local));

    MPI_Barrier(comm);
    for (int r = 0; r < size; ++r) {
        if (r == rank) {
            std::cout << "Rank " << rank << " solution: ";
            for (int i = 0; i < 2; ++i)
                std::cout << x_local[i] << " ";
            std::cout << std::endl;
        }
        MPI_Barrier(comm);
    }

    AMGX_solver_destroy(solver);
    AMGX_vector_destroy(x);
    AMGX_vector_destroy(b);
    AMGX_matrix_destroy(A);
    AMGX_resources_destroy(rsrc);
    AMGX_config_destroy(cfg);
    AMGX_finalize();

    MPI_Finalize();
    return 0;
}
