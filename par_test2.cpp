/*
    * 这是一个简单的AMGX求解器示例，使用FGMRES求解线性方程组Ax=b。
    * 该示例使用AMGX C API进行实现。
    * 注意：在运行之前，请确保已正确安装AMGX并配置CUDA环境。
    * 单进程下测试成功的案例
*/


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
        exit(1);
    }
}

int main(int argc, char **argv) {
    // 不使用MPI，单进程运行
    // 初始化CUDA设备
    cudaSetDevice(0);
    cudaFree(0);

    // 初始化AMGX库
    AMGX_CHECK(AMGX_initialize());
    AMGX_CHECK(AMGX_initialize_plugins());

    // 使用配置文件（按需修改路径）
    AMGX_config_handle cfg;
    const char *config_string = R"(
        config_version=2,
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

    // 创建单进程资源（不用传MPI通信子）
    AMGX_resources_handle rsrc;
    // 此处以单GPU资源创建接口为例
    AMGX_CHECK(AMGX_resources_create_simple(&rsrc, cfg));

    // 创建矩阵、向量和求解器
    AMGX_matrix_handle A;
    AMGX_vector_handle b, x;
    AMGX_solver_handle solver;
    // 如果上传的是标量数据，则使用模式 AMGX_mode_dDDI
    AMGX_CHECK(AMGX_matrix_create(&A, rsrc, AMGX_mode_dDDI));
    AMGX_CHECK(AMGX_vector_create(&b, rsrc, AMGX_mode_dDDI));
    AMGX_CHECK(AMGX_vector_create(&x, rsrc, AMGX_mode_dDDI));
    AMGX_CHECK(AMGX_solver_create(&solver, rsrc, AMGX_mode_dDDI, cfg));

    // 单进程上传数据（用 AMGX_matrix_upload_all 而非 distributed 接口）
    // 例如：4行的矩阵，CSR格式如下：
    // 假设原来的矩阵:
    //  n = 4, nnz=12, row_ptrs = {0, 4, 6, 9, 12} 
    //  col_idx, vals等数据需要按实际情况修改。
    // 这里提供一个简单示例（注意数据要与你的配置匹配）
    const int n = 4;
    const int nnz = 12;
    std::vector<int> row_ptr = {0, 4, 6, 9, 12};
    std::vector<int> col_idx = {0,1,2,3, 1,3, 0,1,2, 0,2,3};
    std::vector<double> vals = {1,2,5,7, 4,9, 10,11,13, 12,15,14};

    AMGX_CHECK(AMGX_matrix_upload_all(
        A, n, nnz, 1, 1,
        row_ptr.data(), col_idx.data(), vals.data(), nullptr
    ));

    // 上传向量b和初始解向量x
    std::vector<double> b_data = {1.0, 1.0, 1.0, 1.0};
    std::vector<double> x_init(n, 0.0);
    AMGX_CHECK(AMGX_vector_upload(b, n, 1, b_data.data()));
    AMGX_CHECK(AMGX_vector_upload(x, n, 1, x_init.data()));

    std::cout << "数据上传完成，开始求解..." << std::endl;

    // 执行求解
    AMGX_CHECK(AMGX_solver_setup(solver, A));
    AMGX_CHECK(AMGX_solver_solve(solver, b, x));

    // 下载结果
    std::vector<double> sol(n);
    AMGX_CHECK(AMGX_vector_download(x, sol.data()));

    std::cout << "求解结果:" << std::endl;
    for (double v : sol)
        std::cout << v << std::endl;

    // 清理资源
    AMGX_solver_destroy(solver);
    AMGX_vector_destroy(x);
    AMGX_vector_destroy(b);
    AMGX_matrix_destroy(A);
    AMGX_resources_destroy(rsrc);
    AMGX_config_destroy(cfg);
    AMGX_finalize_plugins();
    AMGX_finalize();

    return 0;
}