#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include "cuda_runtime.h"
#include "amgx_c.h"
#include <cstring>
#include <iostream>
#include <random>
#include <direct.h> // 添加这个头文件用于 _getcwd

// 添加打印回调函数
void print_callback(const char *msg, int length)
{
    printf("%s", msg);
}

#define AMGX_CHECK(ans)                                                                               \
    {                                                                                                 \
        AMGX_RC code = (ans);                                                                         \
        if (code != AMGX_RC_OK)                                                                       \
        {                                                                                             \
            char msg[512];                                                                            \
            AMGX_get_error_string(code, msg, 512);                                                    \
            std::cerr << "AMGX error in " << __FILE__ << ":" << __LINE__ << ": " << msg << std::endl; \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                             \
        }                                                                                             \
    }

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if (size != 3)
    {
        if (rank == 0)
            std::cerr << "示例需要正好 3 个 MPI 进程，当前为 " << size << std::endl;
        MPI_Finalize();
        std::exit(1);
    }

    // 强制使用 GPU 0
    int dev_count = 0;
    cudaGetDeviceCount(&dev_count);
    std::cout << "Rank " << rank << " sees " << dev_count << " GPUs" << std::endl;
    std::cout << std::flush;

    // 所有进程都使用 GPU 0
    cudaSetDevice(0);
    cudaDeviceSynchronize();
    MPI_Barrier(comm);
    std::cout << "Process " << rank << " using GPU 0" << std::endl
              << std::flush;

    MPI_Barrier(comm);

    // 重置设备
    cudaDeviceReset();
    std::cout << "1" << std::endl;

    // 初始化 AMGX
    AMGX_CHECK(AMGX_initialize());
    AMGX_CHECK(AMGX_register_print_callback(&print_callback));
    AMGX_CHECK(AMGX_install_signal_handler());

    // 创建配置
    const char *config_string = R"(
config_version=2,
communicator=MPI,
solver(main)=FGMRES
)";

    // 创建配置
    AMGX_config_handle cfg;
    char msg[512];
    AMGX_RC err_code = AMGX_config_create(&cfg, config_string);

    std::cout << "1.1" << std::endl;

    AMGX_Mode mode = AMGX_mode_dDDI;
    int block_dimx = 1, block_dimy = 1;
    int block_size = block_dimx * block_dimy;

    // 创建 MPI 并行资源（单 GPU）
    AMGX_resources_handle rsrc;
    int device_ids[1] = {0};
    MPI_Barrier(comm);
    err_code = AMGX_resources_create(&rsrc, cfg, &comm, 1, device_ids);
    if (err_code != AMGX_RC_OK)
    {
        AMGX_get_error_string(err_code, msg, 512);
        std::cerr << "Resource creation error: " << msg << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Barrier(comm);
    std::cout << "1.2" << std::endl;
    // 创建 AMGX matrix、vector 和 solver
    AMGX_matrix_handle A;
    AMGX_vector_handle b, x;
    AMGX_solver_handle solver;
    // 确保所有进程同步创建矩阵
    MPI_Barrier(comm);
    std::cout << "Process " << rank << " creating matrix..." << std::endl
              << std::flush;
    AMGX_CHECK(AMGX_matrix_create(&A, rsrc, mode));

    // 同步创建向量
    MPI_Barrier(comm);
    std::cout << "Process " << rank << " creating vectors..." << std::endl
              << std::flush;
    AMGX_CHECK(AMGX_vector_create(&x, rsrc, mode));
    AMGX_CHECK(AMGX_vector_create(&b, rsrc, mode));

    // 同步创建求解器
    MPI_Barrier(comm);
    std::cout << "Process " << rank << " creating solver..." << std::endl
              << std::flush;
    err_code = AMGX_solver_create(&solver, rsrc, mode, cfg);
    if (err_code != AMGX_RC_OK)
    {
        AMGX_get_error_string(err_code, msg, 512);
        std::cerr << "Process " << rank << " solver creation error: " << msg << std::endl
                  << std::flush;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Barrier(comm);
    std::cout << "1.3" << std::endl;
    // 定义分区向量 [0 0 0 0 1 1 1 1 2 2 2 2]
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
    std::cout << "2" << std::endl;

    // 设置矩阵文件路径
    const char *matrix_file = "D:\\work\\amgx\\AMGX\\examples\\matrix.mtx";
    if (argc > 1)
    {
        matrix_file = argv[1]; // 允许通过命令行参数指定矩阵文件
    }

    // 检查文件是否存在
    FILE *f = fopen(matrix_file, "r");
    if (f == NULL)
    {
        std::cerr << "Error: Cannot open matrix file: " << matrix_file << std::endl;
        std::cerr << "Current working directory: ";
        char cwd[1024];
        if (_getcwd(cwd, sizeof(cwd)) != NULL)
        { // 使用 _getcwd 替代 getcwd
            std::cerr << cwd << std::endl;
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    fclose(f);

    if (rank == 0)
    {
        std::cout << "Using matrix file: " << matrix_file << std::endl;
    }

    // 使用 AMGX_read_system_maps_one_ring 读取和分区矩阵
    AMGX_read_system_maps_one_ring(
        &n, &nnz, &block_dimx, &block_dimy,
        &row_ptrs, &col_indices, &values, &diag, &h_b, &h_x,
        &num_neighbors, &neighbors, &send_sizes, &send_maps, &recv_sizes, &recv_maps,
        rsrc, mode, matrix_file, 1, size,
        partition_sizes, partition_vector_size, partition_vector);
    std::cout << "1" << std::endl;

    // 设置通信映射信息
    MPI_Barrier(comm);
    err_code = AMGX_matrix_comm_from_maps_one_ring(A, 1, num_neighbors, neighbors,
                                                   send_sizes, (const int **)send_maps, recv_sizes, (const int **)recv_maps);
    if (err_code != AMGX_RC_OK)
    {
        AMGX_get_error_string(err_code, msg, 512);
        std::cerr << "Communication mapping error: " << msg << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Barrier(comm);

    // 绑定向量到矩阵
    AMGX_CHECK(AMGX_vector_bind(x, A));
    AMGX_CHECK(AMGX_vector_bind(b, A));

    // 上传矩阵和向量数据
    MPI_Barrier(comm);
    err_code = AMGX_matrix_upload_all(A, n, nnz, block_dimx, block_dimy, row_ptrs, col_indices, values, diag);
    if (err_code != AMGX_RC_OK)
    {
        AMGX_get_error_string(err_code, msg, 512);
        std::cerr << "Matrix upload error: " << msg << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Barrier(comm);
    AMGX_CHECK(AMGX_vector_upload(x, n, block_dimx, h_x));
    AMGX_CHECK(AMGX_vector_upload(b, n, block_dimx, h_b));

    // 进行求解
    MPI_Barrier(comm);
    AMGX_CHECK(AMGX_solver_setup(solver, A));
    MPI_Barrier(comm);
    AMGX_CHECK(AMGX_solver_solve(solver, b, x));
    MPI_Barrier(comm);

    // 下载并打印解
    double *sol = (double *)malloc(n * sizeof(double));
    AMGX_CHECK(AMGX_vector_download(x, sol));
    printf("进程 %d 求解后 x = ", rank);
    for (int i = 0; i < n; i++)
    {
        printf("%f ", sol[i]);
    }
    printf("\n");

    // 清理内存及 AMGX 资源
    free(sol);
    free(partition_vector);
    AMGX_free_system_maps_one_ring(row_ptrs, col_indices, values, diag, h_b, h_x,
                                   num_neighbors, neighbors, send_sizes, send_maps, recv_sizes, recv_maps);

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