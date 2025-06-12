#include <iostream>
#include <vector>
#include <cstdlib>
#include <amgx_c.h>
#include "amgx_solver.h"

#define AMGX_CHECK(ans)     \
    {                       \
        amgx_assert((ans)); \
    }

inline void amgx_assert(AMGX_RC code, AMGX_RC expected = AMGX_RC_OK)
{
    if (code != expected)
    {
        const int error_buf_size = 512;
        char error_buf[error_buf_size];
        AMGX_get_error_string(code, error_buf, error_buf_size);
        std::cerr << "AMGX error [" << code << "]: " << error_buf << std::endl;
        exit(1);
    }
}

int run_amgx_solver()
{
    // 初始化AMGX
    AMGX_CHECK(AMGX_initialize());

    AMGX_config_handle cfg;
    const char *config_string = R"(
        config_version=2,
        solver(main)=FGMRES,
        main:use_scalar_norm= 1, 
        main:tolerance=1e-6,
        main:max_iters=100,
        main:gmres_n_restart=10,
        main:norm=L2,
        main:convergence=RELATIVE_INI;
        main:monitor_residual=1,
        main:store_res_history=1,
        main:preconditioner(amg)=AMG,
        amg:algorithm=AGGREGATION,
        amg:max_iters=1,
        amg:selector=SIZE_8,
        amg:merge_singletons=1,
        amg:cycle=V,
        amg:smoother=MULTICOLOR_DILU,
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

    AMGX_CHECK(AMGX_config_create(&cfg, config_string));

    // 创建 AMGX 资源
    AMGX_resources_handle rsrc;
    AMGX_CHECK(AMGX_resources_create_simple(&rsrc, cfg));

    // 创建矩阵和向量句柄
    AMGX_matrix_handle A;
    AMGX_vector_handle b, x;
    AMGX_CHECK(AMGX_matrix_create(&A, rsrc, AMGX_mode_dDDI));
    AMGX_CHECK(AMGX_vector_create(&b, rsrc, AMGX_mode_dDDI));
    AMGX_CHECK(AMGX_vector_create(&x, rsrc, AMGX_mode_dDDI));

    // 构造一个 5 阶矩阵（CSR 格式）和对应的向量
    const int n = 5;
    const int nnz = 15;
    std::vector<int> row_ptrs = {0, 3, 6, 9, 12, 15};
    std::vector<int> col_indices = {0, 1, 2, 1, 2, 3, 1, 2, 3, 2, 3, 4, 2, 3, 4};
    std::vector<double> values = {1, 6, 7, 2, 8, 9, 10, 3, 11, 12, 4, 13, 14, 15, 5};

    AMGX_CHECK(AMGX_matrix_upload_all(
        A, n, nnz, 1, 1,
        row_ptrs.data(), col_indices.data(), values.data(), nullptr));

    // 上传右侧向量 b（全 1）
    std::vector<double> b_host(n, 1.0);
    AMGX_CHECK(AMGX_vector_upload(b, n, 1, b_host.data()));

    // 上传初始解向量 x（初始化为 0）
    std::vector<double> x_host(n, 0.0);
    AMGX_CHECK(AMGX_vector_upload(x, n, 1, x_host.data()));

    // 创建求解器，设置求解器并求解线性系统
    AMGX_solver_handle solver;
    AMGX_CHECK(AMGX_solver_create(&solver, rsrc, AMGX_mode_dDDI, cfg));
    AMGX_CHECK(AMGX_solver_setup(solver, A));
    AMGX_CHECK(AMGX_solver_solve(solver, b, x));
    AMGX_CHECK(AMGX_vector_download(x, x_host.data()));

    // 输出解向量
    std::cout << "Solution vector:" << std::endl;
    for (double val : x_host)
    {
        std::cout << val << std::endl;
    }

    // 清理各类对象和资源
    AMGX_matrix_destroy(A);
    AMGX_vector_destroy(b);
    AMGX_vector_destroy(x);
    AMGX_solver_destroy(solver);
    AMGX_resources_destroy(rsrc);
    AMGX_config_destroy(cfg);
    AMGX_finalize();

    return 0;
}