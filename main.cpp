// #include <iostream>
// #include <vector>
// #include <cstdlib>
// #include <amgx_c.h>

// #define AMGX_CHECK(ans) { amgx_assert((ans)); }

// inline void amgx_assert(AMGX_RC code, AMGX_RC expected = AMGX_RC_OK)
// {
//     if(code != expected)
//     {
//         const int error_buf_size = 512;
//         char error_buf[error_buf_size];
//         AMGX_get_error_string(code, error_buf, error_buf_size);
//         std::cerr << "AMGX error [" << code << "]: " << error_buf << std::endl;
//         exit(1);
//     }
// }

// int main()
// {
//     // 初始化 AMGX 库
//     AMGX_CHECK(AMGX_initialize());

//     AMGX_config_handle cfg;
//     const char *config_string = R"(
//         config_version=2,
//         solver(main)=FGMRES,
//         main:use_scalar_norm= 1, 
//         main:tolerance=1e-6,
//         main:max_iters=100,
//         main:gmres_n_restart=10,
//         main:norm=L2,
//         main:convergence=RELATIVE_INI;
//         main:monitor_residual=1,
//         main:store_res_history=1,
//         main:preconditioner(amg)=AMG,
//         amg:algorithm=AGGREGATION,
//         amg:max_iters=1,
//         amg:selector=SIZE_8,
//         amg:merge_singletons=1,
//         amg:cycle=V,
//         amg:smoother=MULTICOLOR_DILU,
//         amg:presweeps=0,
//         amg:postsweeps=3,
//         amg:error_scaling=0,
//         amg:max_levels=100,
//         amg:coarseAgenerator=LOW_DEG,
//         amg:matrix_coloring_scheme=PARALLEL_GREEDY,
//         amg:max_uncolored_percentage=0.05,
//         amg:max_unassigned_percentage=0.05,
//         amg:relaxation_factor=0.75,
//         amg:coarse_solver(coarse)=FGMRES,
//         amg:min_coarse_rows=24,
//         coarse:max_iters=10,
//         coarse:gmres_n_restart=100,
//         coarse:monitor_residual=1,
//         coarse:preconditioner(coarse_pre)=MULTICOLOR_DILU,
//         coarse_pre:max_iters=3,
//         coarse_pre:monitor_residual=0,
//         print_solve_stats=1,
//         obtain_timings=1,
//         monitor_residual=1
//     )";

//     AMGX_CHECK(AMGX_config_create(&cfg, config_string));

//     // 创建 AMGX 资源
//     AMGX_resources_handle rsrc;
//     AMGX_CHECK(AMGX_resources_create_simple(&rsrc, cfg));

//     // 创建矩阵和向量句柄
//     AMGX_matrix_handle A;
//     AMGX_vector_handle b, x;
//     AMGX_CHECK(AMGX_matrix_create(&A, rsrc, AMGX_mode_dFFI));
//     AMGX_CHECK(AMGX_vector_create(&b, rsrc, AMGX_mode_dFFI));
//     AMGX_CHECK(AMGX_vector_create(&x, rsrc, AMGX_mode_dFFI));

//     //const int n = 4;
//     //const int nnz = 12;
//     //std::vector<int> row_ptrs = {0, 4, 6, 9, 12};
//     //std::vector<int> col_indices = {0, 1, 2,3, 1, 3, 0,1, 2, 0,2, 3};
//     //std::vector<double> values = {1, 2, 5, 7, 4, 9, 10, 11, 13, 12,  15, 14};

//     const int n =4;
//     const int nnz = 2;
//     std::vector<int> row_ptrs = {0, 2,4};
//     std::vector<int> col_indices = {0, 1,0, 1};
//     std::vector<double> values ={
//         1,2,0,4,
//         5,7,0,9,
//         10,11,12,0,
//         13,0,15,14
//     };

//     AMGX_CHECK(AMGX_matrix_upload_all(
//         A, 2, nnz, 2, 2,
//         row_ptrs.data(), col_indices.data(), values.data(), nullptr
//     ));

//     // 上传右侧向量 b（全 1）
//     std::vector<double> b_host(n, 1.0);
//     AMGX_CHECK(AMGX_vector_upload(b, 2, 2, b_host.data()));

//     // 上传初始解向量 x（初始化为 0）
//     std::vector<double> x_host(n, 0.0);
//     AMGX_CHECK(AMGX_vector_upload(x, 2, 2, x_host.data()));

//     // 创建求解器，设置求解器并求解线性系统
//     AMGX_solver_handle solver;
//     AMGX_CHECK(AMGX_solver_create(&solver, rsrc, AMGX_mode_dFFI, cfg));
//     AMGX_CHECK(AMGX_solver_setup(solver, A));
//     AMGX_CHECK(AMGX_solver_solve(solver, b, x));
//     AMGX_CHECK(AMGX_vector_download(x, x_host.data()));

//     // 输出解向量
//     std::cout << "Solution vector:" << std::endl;
//     for (double val : x_host)
//     {
//         std::cout << val << std::endl;
//     }

//     // 清理各类对象和资源
//     AMGX_matrix_destroy(A);
//     AMGX_vector_destroy(b);
//     AMGX_vector_destroy(x);
//     AMGX_solver_destroy(solver);
//     AMGX_resources_destroy(rsrc);
//     AMGX_config_destroy(cfg);
//     AMGX_finalize();

//     return 0;
// }

#include <iostream>
#include <vector>
#include <cstdlib>
#include <amgx_c.h>

#define AMGX_CHECK(ans) { amgx_assert((ans)); }

inline void amgx_assert(AMGX_RC code, AMGX_RC expected = AMGX_RC_OK)
{
    if(code != expected)
    {
        const int error_buf_size = 512;
        char error_buf[error_buf_size];
        AMGX_get_error_string(code, error_buf, error_buf_size);
        std::cerr << "AMGX error [" << code << "]: " << error_buf << std::endl;
        exit(1);
    }
}

int main()
{
    // 初始化 AMGX 库
    AMGX_CHECK(AMGX_initialize());

    AMGX_config_handle cfg;
    const char *config_string = R"(
        config_version=2,
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
        amg:block_size=2,          // 明确指定块大小
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
    AMGX_CHECK(AMGX_matrix_create(&A, rsrc, AMGX_mode_dFFI));
    AMGX_CHECK(AMGX_vector_create(&b, rsrc, AMGX_mode_dFFI));
    AMGX_CHECK(AMGX_vector_create(&x, rsrc, AMGX_mode_dFFI));

    // 块矩阵参数
    const int block_rows = 2;       // 块行数
    const int block_dimx = 2;       // 每个块的行数
    const int block_dimy = 2;       // 每个块的列数
    const int nnz = 2;              // 非零块数量
    std::vector<int> row_ptrs = {0, 1, 2}; // 每个块行的非零块起始位置
    std::vector<int> col_indices = {0, 1};  // 非零块的列索引
    std::vector<double> values = {
        1.0, -2.0, -3.0, 1.0,      // 第一个2x2块（行优先）
        1.0, -4.0, -5.0, 1.0        // 第二个2x2块（行优先）
    };

    // 上传矩阵（注意参数顺序）
    AMGX_CHECK(AMGX_matrix_upload_all(
        A, 
        block_rows,  // 块行数
        nnz,         // 非零块数量
        block_dimx,  // 块的行数
        block_dimy,  // 块的列数
        row_ptrs.data(), 
        col_indices.data(), 
        values.data(), 
        nullptr
    ));

    // 上传右侧向量b（每个块2个元素，共2个块）
    const int vec_blocks = 2;       // 向量块数
    std::vector<double> b_host(vec_blocks * block_dimx, 1.0); // 每个块元素为1.0
    AMGX_CHECK(AMGX_vector_upload(b, vec_blocks, block_dimx, b_host.data()));

    // 上传初始解向量x（初始化为0）
    std::vector<double> x_host(vec_blocks * block_dimx, 0.0);
    AMGX_CHECK(AMGX_vector_upload(x, vec_blocks, block_dimx, x_host.data()));

    // 创建求解器并求解
    AMGX_solver_handle solver;
    AMGX_CHECK(AMGX_solver_create(&solver, rsrc, AMGX_mode_dFFI, cfg));
    AMGX_CHECK(AMGX_solver_setup(solver, A));
    AMGX_CHECK(AMGX_solver_solve(solver, b, x));
    AMGX_CHECK(AMGX_vector_download(x, x_host.data()));

    // 输出解向量
    std::cout << "Solution vector:" << std::endl;
    for (double val : x_host)
    {
        std::cout << val << std::endl;
    }

    // 清理资源
    AMGX_matrix_destroy(A);
    AMGX_vector_destroy(b);
    AMGX_vector_destroy(x);
    AMGX_solver_destroy(solver);
    AMGX_resources_destroy(rsrc);
    AMGX_config_destroy(cfg);
    AMGX_finalize();

    return 0;
}