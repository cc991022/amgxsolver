#include <mpi.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <amgx_c.h>
#include <cuda_runtime.h>

#define AMGX_CHECK(ans) { amgx_assert((ans)); }

inline void amgx_assert(AMGX_RC code) {
    if (code != AMGX_RC_OK) {
        char msg[512];
        AMGX_get_error_string(code, msg, 512);
        std::cerr << "AMGX error: " << msg << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
        exit(1);
    }
}

void print_callback(const char *msg, int length) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) std::cout << std::string(msg, length);
}

int main(int argc, char **argv) {
    std::cout<<"hang of 1"<<std::endl;
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    std::cout<<"hang of 2"<<std::endl;
    if (size != 2) {
        if (rank == 0)
            std::cerr << "此测试程序要求 2 个 MPI 进程" << std::endl;
        MPI_Finalize();
        return 1;
    }
    std::cout<<"hang of 3"<<std::endl;

    cudaSetDevice(0); // 所有进程共用 GPU 0
    AMGX_CHECK(AMGX_initialize());
    AMGX_CHECK(AMGX_initialize_plugins());
    AMGX_register_print_callback(&print_callback);

    // 加载 AMGX 配置
    AMGX_config_handle cfg;
    std::cout<<"hang of 4"<<std::endl;
    AMGX_CHECK(AMGX_config_create_from_file(&cfg, "D:/work/amgx/AMGX/install/lib/configs/FGMRES_AGGREGATION_DILU.json"));

    // 创建资源
    AMGX_resources_handle rsrc;
    int device_id = 0;
    AMGX_CHECK(AMGX_resources_create(&rsrc, cfg, &comm, 1, &device_id));

    // 创建矩阵、向量、求解器
    AMGX_matrix_handle A;
    AMGX_vector_handle b, x;
    AMGX_solver_handle solver;
    std::cout<<"hang of 5"<<std::endl;
    AMGX_CHECK(AMGX_matrix_create(&A, rsrc, AMGX_mode_dDDI));
    AMGX_CHECK(AMGX_vector_create(&b, rsrc, AMGX_mode_dDDI));
    AMGX_CHECK(AMGX_vector_create(&x, rsrc, AMGX_mode_dDDI));
    AMGX_CHECK(AMGX_solver_create(&solver, rsrc, AMGX_mode_dDDI, cfg));
    std::cout<<"hang of 6"<<std::endl;
    const int global_n = 4;

    // 设置 partition offsets
    int64_t partition_offsets[3] = {0, 2, 4};  // rank 0: 0-1, rank 1: 2-3

    // 创建 AMGX distribution 结构
    AMGX_distribution_handle dist;
    AMGX_CHECK(AMGX_distribution_create(&dist, cfg));
    AMGX_CHECK(AMGX_distribution_set_partition_data(dist, AMGX_DIST_PARTITION_OFFSETS, partition_offsets));
    std::cout<<"hang of 7"<<std::endl;
    if (rank == 0) {
        // 行 0 和 1
        std::vector<int> row_ptr = {0, 2, 5};
        std::vector<int> col_idx = {0, 1, 0, 1, 2};
        std::vector<double> vals   = {10.0, -1.0, -1.0, 10.0, -2.0};
        std::vector<double> b_vals = {1.0, 2.0};
        std::vector<double> x_init = {0.0, 0.0};

        //AMGX_CHECK(AMGX_matrix_upload_distributed(A, global_n, 2, 5, 1, 1, row_ptr.data(), col_idx.data(), vals.data(), nullptr, dist));
        AMGX_RC rc = AMGX_matrix_upload_distributed(A, global_n, 2, 5, 1, 1, row_ptr.data(), col_idx.data(), vals.data(), nullptr, dist);
        if (rc != AMGX_RC_OK)
        {
            std::cerr << "Matrix upload failed" << std::endl;
            exit(1);
        }

        // AMGX_CHECK(AMGX_vector_upload(b, 2, 1, b_vals.data()));
        rc = AMGX_vector_upload(b, 2, 1, b_vals.data());
        if (rc != AMGX_RC_OK)
        {
            std::cerr << "Vector upload failed" << std::endl;
            exit(1);
        }

        AMGX_CHECK(AMGX_vector_upload(x, 2, 1, x_init.data()));
    } else {
        // 行 2 和 3
        std::vector<int> row_ptr = {0, 3, 5};
        std::vector<int> col_idx = {1, 2, 3, 2, 3};
        std::vector<double> vals   = {-2.0, 10.0, -3.0, -3.0, 10.0};
        std::vector<double> b_vals = {3.0, 4.0};
        std::vector<double> x_init = {0.0, 0.0};

        //AMGX_CHECK(AMGX_matrix_upload_distributed(A, global_n, 2, 5, 1, 1, row_ptr.data(), col_idx.data(), vals.data(), nullptr, dist));
        AMGX_RC rc = AMGX_matrix_upload_distributed(A, global_n, 2, 5, 1, 1, row_ptr.data(), col_idx.data(), vals.data(), nullptr, dist);
        if (rc != AMGX_RC_OK)
        {
            std::cerr << "Matrix upload failed" << std::endl;
            exit(1);
        }
        // AMGX_CHECK(AMGX_vector_upload(b, 2, 1, b_vals.data()));
        rc = AMGX_vector_upload(b, 2, 1, b_vals.data());
        if (rc != AMGX_RC_OK)
        {
            std::cerr << "Vector upload failed" << std::endl;
            exit(1);
        }
        AMGX_CHECK(AMGX_vector_upload(x, 2, 1, x_init.data()));
    }
    std::cout<<"hang of 8"<<std::endl;
    // 求解器 setup 和 solve
    AMGX_CHECK(AMGX_solver_setup(solver, A));
    AMGX_CHECK(AMGX_solver_solve(solver, b, x));
    std::cout<<"hang of 9"<<std::endl;
    // 下载解并输出
    if (rank == 0) {
        std::vector<double> x_host(2);
        AMGX_CHECK(AMGX_vector_download(x, x_host.data()));
        std::cout << "[rank 0] 解: ";
        for (double v : x_host) std::cout << v << " ";
        std::cout << std::endl;
    } else {
        std::vector<double> x_host(2);
        AMGX_CHECK(AMGX_vector_download(x, x_host.data()));
        std::cout << "[rank 1] 解: ";
        for (double v : x_host) std::cout << v << " ";
        std::cout << std::endl;
    }
    std::cout<<"hang of 10"<<std::endl;
    // 清理资源
    AMGX_solver_destroy(solver);
    AMGX_vector_destroy(x);
    AMGX_vector_destroy(b);
    AMGX_matrix_destroy(A);
    AMGX_resources_destroy(rsrc);
    AMGX_config_destroy(cfg);
    AMGX_distribution_destroy(dist);
    AMGX_finalize_plugins();
    AMGX_finalize();
    MPI_Finalize();

    return 0;
}
