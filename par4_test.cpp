// par_test3.cpp
/*
   *指定两进程下，运行API成功的案例
*/

#include <iostream>
#include <vector>
#include <cstdlib>
#include <mpi.h>
#include <amgx_c.h>
#include <cuda_runtime.h>
#include <random>
#include <direct.h>
#include <cstring>

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

    if (size != 3) {
        if (rank == 0)
            std::cerr << "示例需要正好 2 个 MPI 进程，当前为 " << size << std::endl;
        MPI_Finalize();
        std::exit(1);
    }

    // 强制使用 GPU 0
    cudaSetDevice(0);
    cudaFree(0);
    cudaDeviceSynchronize();
    MPI_Barrier(comm);

    // 初始化 AMGX
    AMGX_CHECK(AMGX_initialize());

    // 创建配置
const char *config_string = R"(
        config_version=2,
        communicator=MPI,
        solver(main)=FGMRES,
        main:use_scalar_norm= 1, 
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

    // 创建 MPI 并行资源（单 GPU）
    AMGX_resources_handle rsrc;
    int device_ids[1] = { 0 };
    AMGX_CHECK(AMGX_resources_create(&rsrc, cfg, &comm, 1, device_ids));

    // 创建矩阵、向量、求解器句柄
    AMGX_matrix_handle A;
    AMGX_vector_handle b, x;
    AMGX_solver_handle solver;
    AMGX_CHECK(AMGX_matrix_create(&A, rsrc, AMGX_mode_dDDI));
    AMGX_CHECK(AMGX_vector_create(&b, rsrc, AMGX_mode_dDDI));
    AMGX_CHECK(AMGX_vector_create(&x, rsrc, AMGX_mode_dDDI));
    AMGX_CHECK(AMGX_solver_create(&solver, rsrc, AMGX_mode_dDDI, cfg));

    int partition_vector_size = 12;
    int *partition_vector = (int *)malloc(partition_vector_size * sizeof(int));
    for (int i = 0; i < 4; i++)
        partition_vector[i] = 0;
    for (int i = 4; i < 8; i++)
        partition_vector[i] = 1;
    for (int i = 8; i < 12; i++)
        partition_vector[i] = 2;

    // 定义矩阵数据
    int n, nnz, num_neighbors;
    int *row_ptrs = NULL, *col_indices = NULL, *neighbors = NULL;
    void *values = NULL, *diag = NULL, *h_x = NULL, *h_b = NULL;
    int *send_sizes = NULL, **send_maps = NULL;
    int *recv_sizes = NULL, **recv_maps = NULL;
    int *partition_sizes = NULL;

    // 本地子矩阵存储
    const int global_n = 4, local_n = 2;
    std::vector<int> row_ptr;
    std::vector<int64_t> col_idx;
    std::vector<double> vals;
    std::vector<double> b_data(local_n, 1.0);
    std::vector<double> x_init(local_n, 0.0);

        const char *matrix_file = "D:\\work\\amgx\\AMGX\\examples\\matrix.mtx";
    if (argc > 1)
    {
        matrix_file = argv[1]; // 允许通过命令行参数指定矩阵文件
    }
    
  

    // 设置分布
    int64_t partition_offsets[3] = {0, 2, 4};
    AMGX_distribution_handle dist;
    AMGX_CHECK(AMGX_distribution_create(&dist, cfg));
    AMGX_CHECK(AMGX_distribution_set_partition_data(dist, AMGX_DIST_PARTITION_OFFSETS, partition_offsets));

    if (rank == 0) {
        // 行 0 和 1
        row_ptr = {0, 4, 6};
        col_idx = {0,1,2,3,  1,3};
        vals    = {1,2,5,7,  4,9};
        b_data = {1.0,1.0};
    } else {
        // 行 2 和 3
        row_ptr = {0, 3, 6};
        col_idx = {0,1,2,  0,2,3};
        vals    = {10,11,13, 12,15,14};
        b_data = {1.0,1.0};
    }
    


    AMGX_CHECK(AMGX_matrix_upload_distributed(
        A, global_n, local_n, vals.size(), 1, 1, 
        row_ptr.data(), col_idx.data(), vals.data(), 
        nullptr, dist
    ));
    // 上传向量 b, x
    AMGX_CHECK(AMGX_vector_bind(b, A));
    AMGX_CHECK(AMGX_vector_bind(x, A));
    AMGX_CHECK(AMGX_vector_upload(b, local_n, 1, b_data.data()));
    AMGX_CHECK(AMGX_vector_upload(x, local_n, 1, x_init.data()));
    AMGX_CHECK(AMGX_solver_setup(solver, A));
    AMGX_CHECK(AMGX_solver_solve(solver, b, x));
    std::vector<double> local_x(2);
    AMGX_CHECK(AMGX_vector_download(x, local_x.data()));
    
    for (double val : local_x) {
        std::cout << val << std::endl;
    }

    // 清理
    AMGX_CHECK(AMGX_distribution_destroy(dist));
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
