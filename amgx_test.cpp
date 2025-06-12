#include <iostream>
#include <vector>
#include <cstdlib>
#include <amgx_c.h>

#define AMGX_CHECK(ans) { amgx_assert((ans)); }

inline void amgx_assert(AMGX_RC code, AMGX_RC expected = AMGX_RC_OK) {
    if (code != expected) {
        const int error_buf_size = 512;
        char error_buf[error_buf_size];
        AMGX_get_error_string(code, error_buf, error_buf_size);
        std::cerr << "AMGX error [" << code << "]: " << error_buf << std::endl;
        exit(1);
    }
}

int main() {
    AMGX_CHECK(AMGX_initialize());

    // 使用 GMRES 求解器的配置，并禁用预处理器（即不使用 AMG）
    AMGX_config_handle cfg;
    const char *config_string =
        "config_version=2, "
        "solver=GMRES, "
        "GMRES:monitor_residual=1, " // 启用残差监控
        "GMRES:tolerance=1e-1, "
        "GMRES:max_iters=100000, "
        "print_solve_stats=1";

    AMGX_CHECK(AMGX_config_create(&cfg, config_string));

    AMGX_resources_handle rsrc;
    AMGX_CHECK(AMGX_resources_create_simple(&rsrc, cfg));

    AMGX_matrix_handle A;
    AMGX_vector_handle b, x;
    AMGX_CHECK(AMGX_matrix_create(&A, rsrc, AMGX_mode_dDDI));
    AMGX_CHECK(AMGX_vector_create(&b, rsrc, AMGX_mode_dDDI));
    AMGX_CHECK(AMGX_vector_create(&x, rsrc, AMGX_mode_dDDI));

    const int n = 5;
    const int nnz = 15;
    std::vector<int> row_ptrs = {0, 3, 6, 9, 12, 15};
    std::vector<int> col_indices = {0, 1, 2, 1, 2, 3, 1, 2, 3, 2, 3, 4, 2, 3, 4};
    std::vector<double> values = {1, 6, 7, 2, 8, 9, 10, 3, 11, 12, 4, 13, 14, 15, 5};

    AMGX_CHECK(AMGX_matrix_upload_all(
        A, n, nnz, 1, 1,
        row_ptrs.data(), col_indices.data(), values.data(), nullptr
    ));

    std::vector<double> b_host(n, 1.0);
    AMGX_CHECK(AMGX_vector_upload(b, n, 1, b_host.data()));

    std::vector<double> x_host(n, 0.0);
    AMGX_CHECK(AMGX_vector_upload(x, n, 1, x_host.data()));

    AMGX_solver_handle solver;
    AMGX_CHECK(AMGX_solver_create(&solver, rsrc, AMGX_mode_dDDI, cfg));
    AMGX_CHECK(AMGX_solver_setup(solver, A));

    AMGX_CHECK(AMGX_solver_solve(solver, b, x));

    AMGX_CHECK(AMGX_vector_download(x, x_host.data()));
    std::cout << "Solution vector:\n";
    for (double val : x_host) {
        std::cout << val << std::endl;
    }

    AMGX_matrix_destroy(A);
    AMGX_vector_destroy(b);
    AMGX_vector_destroy(x);
    AMGX_solver_destroy(solver);
    AMGX_resources_destroy(rsrc);
    AMGX_config_destroy(cfg);
    AMGX_finalize();

    return 0;
}