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

    // 从 JSON 配置文件中加载配置（请确保文件路径正确）
    AMGX_config_handle cfg;
    AMGX_CHECK(AMGX_config_create_from_file(&cfg, "D:\\work\\amgx\\AMGX\\install\\lib\\configs\\FGMRES_AGGREGATION_DILU.json"));

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
    // 原始 CSR 数据：
    //   row_ptrs: {0, 3, 6, 9, 12, 15}
    //   col_indices: {0, 1, 2,  1, 2, 3,  1, 2, 3,  2, 3, 4,  2, 3, 4}
    //   values:      {1, 6, 7,  2, 8, 9,  10, 3, 11,  12, 4, 13,  14, 15, 5}
    // 其中，对于每一行，主对角线元素为：
    //   row0: 对角线在列0，值 = 1
    //   row1: 对角线在列1，值 = 2
    //   row2: 对角线在列2，值 = 3
    //   row3: 对角线在列3，值 = 4
    //   row4: 对角线在列4，值 = 5
    // 故单独构造 diag_data 数组：
    const int n = 5;
    const int nnz = 10;
    std::vector<int> row_ptrs = {0, 2, 4, 6, 8, 10};
    std::vector<int> col_indices = {1, 2, 2, 3, 1, 3, 2, 4, 2, 3};
    std::vector<double> values = {6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    std::vector<double> diag_data = {1, 2, 3, 4, 5};

    AMGX_CHECK(AMGX_matrix_upload_all(
        A, n, nnz, 1, 1,
        row_ptrs.data(), col_indices.data(), values.data(), diag_data.data()
    ));

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