#include <mpi.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <amgx_c.h>
#include <cuda_runtime.h>
#include <cassert>

#define  WMPI_Init MPI_Init
#define AMGX_CHECK(ans) { amgx_assert((ans)); }

inline void amgx_assert(AMGX_RC code) {
    if (code != AMGX_RC_OK) {
        char msg[512];
        AMGX_get_error_string(code, msg, 512);
        std::cerr << "AMGX error: " << msg << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

void print_callback(const char *msg, int length) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) std::cout << std::string(msg, length);
}

int main(int argc, char **argv) {
    system("chcp 65001");
    std::setlocale(LC_ALL, "chs");
    WMPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if (size != 2) {
        if (rank == 0) std::cerr << "need 2 rank" << std::endl;
        MPI_Finalize();
        return 1;
    }

    // 单GPU设置
    cudaSetDevice(0);
    cudaFree(0);
    cudaDeviceSynchronize();
    MPI_Barrier(comm);

    AMGX_CHECK(AMGX_initialize());
    AMGX_CHECK(AMGX_initialize_plugins());
    AMGX_register_print_callback(&print_callback);

    // 加载配置
    AMGX_config_handle cfg;
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
    // AMGX_CHECK(AMGX_config_create(&cfg, "config_version=2, solver(main)=FGMRES, main:max_iters=100, main:convergence=RELATIVE_MAX, main:monitor_residual=1, main:relaxation_factor=1.0, main:reorder_cols_by_nonzeros=0"));
     AMGX_CHECK(AMGX_config_create(&cfg, config_string));


    // 创建资源
    AMGX_resources_handle rsrc;
    int device_ids[1] = {0};
    AMGX_CHECK(AMGX_resources_create(&rsrc, cfg, &comm, 1, device_ids));
    // AMGX_RC rc = AMGX_resources_check_initialization(rsrc);
    // if (rc != AMGX_RC_OK)
    // {
    //     std::cerr << "资源初始化失败" << std::endl;
    //     MPI_Abort(comm, 1);
    // }

    // 设置分布
    int64_t partition_offsets[3] = {0, 2, 4};
    AMGX_distribution_handle dist;
    AMGX_CHECK(AMGX_distribution_create(&dist, cfg));
    AMGX_CHECK(AMGX_distribution_set_partition_data(dist, AMGX_DIST_PARTITION_OFFSETS, partition_offsets));

    // 创建矩阵/向量
    AMGX_matrix_handle A;
    AMGX_vector_handle b, x;
    AMGX_solver_handle solver;
    AMGX_CHECK(AMGX_matrix_create(&A, rsrc, AMGX_mode_dDDI));
    AMGX_CHECK(AMGX_vector_create(&b, rsrc, AMGX_mode_dDDI));
    AMGX_CHECK(AMGX_vector_create(&x, rsrc, AMGX_mode_dDDI));
    AMGX_CHECK(AMGX_solver_create(&solver, rsrc, AMGX_mode_dDDI, cfg));

    // 上传矩阵数据
    if (rank == 0) {
        std::vector<int> row_ptr = {0, 2, 5};
        std::vector<int> col_idx = {0, 1, 0, 1, 2};
        std::vector<double> vals = {10.0, -1.0, -1.0, 10.0, -2.0};
        std::vector<double> b_data = {1.0, 2.0};
        std::vector<double> x_init = {0.0, 0.0};

        AMGX_CHECK(AMGX_matrix_upload_distributed(
            A, 4, 2, 5, 1, 1, 
            row_ptr.data(), col_idx.data(), vals.data(), 
            nullptr, dist
        ));
        AMGX_CHECK(AMGX_vector_upload(b, 2, 1, b_data.data()));
        AMGX_CHECK(AMGX_vector_upload(x, 2, 1, x_init.data()));
    } else {
        std::vector<int> row_ptr = {0, 3, 5};
        std::vector<int> col_idx = {1, 2, 3, 2, 3};
        std::vector<double> vals = {-2.0, 10.0, -3.0, -3.0, 10.0};
        std::vector<double> b_data = {3.0, 4.0};
        std::vector<double> x_init = {0.0, 0.0};

        AMGX_CHECK(AMGX_matrix_upload_distributed(
            A, 4, 2, 5, 1, 1,
            row_ptr.data(), col_idx.data(), vals.data(),
            nullptr, dist
        ));
        AMGX_CHECK(AMGX_vector_upload(b, 2, 1, b_data.data()));
        AMGX_CHECK(AMGX_vector_upload(x, 2, 1, x_init.data()));
    }
    AMGX_CHECK(AMGX_vector_bind(b, A));
    AMGX_CHECK(AMGX_vector_bind(x, A));


    // 同步检查
    MPI_Barrier(comm);
    if (rank == 0) std::cout << "upload of data and now compute ..." << std::endl;

    // 求解
    AMGX_CHECK(AMGX_solver_setup(solver, A));
    AMGX_CHECK(AMGX_solver_solve(solver, b, x));
    // 输出结果
    std::vector<double> local_x(2);
    AMGX_CHECK(AMGX_vector_download(x, local_x.data()));
    for (int r = 0; r < size; ++r) {
        MPI_Barrier(comm);
        if (rank == r) {
            std::cout << "rank " << rank << " of solve : ";
            for (double v : local_x) std::cout << v << " ";
            std::cout << std::endl;
        }
    }

    // 清理资源
    AMGX_solver_destroy(solver);
    AMGX_vector_destroy(x);
    AMGX_vector_destroy(b);
    AMGX_matrix_destroy(A);
    AMGX_distribution_destroy(dist);
    AMGX_resources_destroy(rsrc);
    AMGX_config_destroy(cfg);
    AMGX_finalize_plugins();
    AMGX_finalize();

    MPI_Finalize();
    return 0;
}