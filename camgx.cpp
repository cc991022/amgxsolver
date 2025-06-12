#include <iostream>
#include <vector>
#include <cstdlib>
#include <cstdio>
#include <windows.h> // 新增Windows API头文件

// 删除原AMGX头文件引用（避免隐式链接）
// #include <amgx_c.h>
// --- 手动声明AMGX类型和枚举（需根据amgx_c.h精简） ---
typedef enum
{
    AMGX_RC_OK = 0,
    AMGX_RC_BAD_PARAMETERS = 1,
    AMGX_RC_UNKNOWN = 2,
    AMGX_RC_NOT_SUPPORTED_TARGET = 3,
    AMGX_RC_NOT_SUPPORTED_BLOCKSIZE = 4,
    AMGX_RC_CUDA_FAILURE = 5,
    AMGX_RC_THRUST_FAILURE = 6,
    AMGX_RC_NO_MEMORY = 7,
    AMGX_RC_IO_ERROR = 8,
    AMGX_RC_BAD_MODE = 9,
    AMGX_RC_CORE = 10,
    AMGX_RC_PLUGIN = 11,
    AMGX_RC_BAD_CONFIGURATION = 12,
    AMGX_RC_NOT_IMPLEMENTED = 13,
    AMGX_RC_LICENSE_NOT_FOUND = 14,
    AMGX_RC_INTERNAL = 15
} AMGX_RC ;

typedef enum
{
    AMGX_unset = -1,
    AMGX_mode_hDDI = 8192,
    AMGX_mode_hDFI = 8448,
    AMGX_mode_hFFI = 8464,
    AMGX_mode_dDDI = 8193,
    AMGX_mode_dDFI = 8449,
    AMGX_mode_dFFI = 8465,
    AMGX_mode_hZZI = 8192,
    AMGX_mode_hZCI =  8448,
    AMGX_mode_hCCI = 8464,
    AMGX_mode_dZZI = 8193,
    AMGX_mode_dZCI = 8449,
    AMGX_mode_dCCI = 8465,
    AMGX_modeNum = 10
} AMGX_Mode;

// 定义句柄类型（通常为void*的别名）
typedef void* AMGX_config_handle;
typedef void* AMGX_resources_handle;
typedef void* AMGX_matrix_handle;
typedef void* AMGX_vector_handle;
typedef void* AMGX_solver_handle;
// 定义所有需要的AMGX函数原型（需根据AMGX文档手动声明）
typedef AMGX_RC (*AMGX_initialize_t)();
typedef AMGX_RC (*AMGX_finalize_t)();
typedef AMGX_RC (*AMGX_config_create_t)(AMGX_config_handle *, const char *);
typedef AMGX_RC (*AMGX_resources_create_simple_t)(AMGX_resources_handle *, AMGX_config_handle);
typedef AMGX_RC (*AMGX_matrix_create_t)(AMGX_matrix_handle *, AMGX_resources_handle, AMGX_Mode);
typedef AMGX_RC (*AMGX_vector_create_t)(AMGX_vector_handle *, AMGX_resources_handle, AMGX_Mode);
typedef AMGX_RC (*AMGX_matrix_upload_all_t)(AMGX_matrix_handle, int, int, int, int, const int *, const int *, const double *, const double *);
typedef AMGX_RC (*AMGX_vector_upload_t)(AMGX_vector_handle, int, int, const double *);
typedef AMGX_RC (*AMGX_solver_create_t)(AMGX_solver_handle *, AMGX_resources_handle, AMGX_Mode, AMGX_config_handle);
typedef AMGX_RC (*AMGX_solver_setup_t)(AMGX_solver_handle, AMGX_matrix_handle);
typedef AMGX_RC (*AMGX_solver_solve_t)(AMGX_solver_handle, AMGX_vector_handle, AMGX_vector_handle);
typedef AMGX_RC (*AMGX_vector_download_t)(AMGX_vector_handle, double *);
typedef AMGX_RC (*AMGX_matrix_destroy_t)(AMGX_matrix_handle);
typedef AMGX_RC (*AMGX_vector_destroy_t)(AMGX_vector_handle);
typedef AMGX_RC (*AMGX_solver_destroy_t)(AMGX_solver_handle);
typedef AMGX_RC (*AMGX_resources_destroy_t)(AMGX_resources_handle);
typedef AMGX_RC (*AMGX_config_destroy_t)(AMGX_config_handle);
typedef AMGX_RC (*AMGX_get_error_string_t)(AMGX_RC, char *, int);

// 定义全局函数指针
AMGX_initialize_t pAMGX_initialize = nullptr;
AMGX_finalize_t pAMGX_finalize = nullptr;
AMGX_config_create_t pAMGX_config_create = nullptr;
AMGX_resources_create_simple_t pAMGX_resources_create_simple = nullptr;
AMGX_matrix_create_t pAMGX_matrix_create = nullptr;
AMGX_vector_create_t pAMGX_vector_create = nullptr;
AMGX_matrix_upload_all_t pAMGX_matrix_upload_all = nullptr;
AMGX_vector_upload_t pAMGX_vector_upload = nullptr;
AMGX_solver_create_t pAMGX_solver_create = nullptr;
AMGX_solver_setup_t pAMGX_solver_setup = nullptr;
AMGX_solver_solve_t pAMGX_solver_solve = nullptr;
AMGX_vector_download_t pAMGX_vector_download = nullptr;
AMGX_matrix_destroy_t pAMGX_matrix_destroy = nullptr;
AMGX_vector_destroy_t pAMGX_vector_destroy = nullptr;
AMGX_solver_destroy_t pAMGX_solver_destroy = nullptr;
AMGX_resources_destroy_t pAMGX_resources_destroy = nullptr;
AMGX_config_destroy_t pAMGX_config_destroy = nullptr;
AMGX_get_error_string_t pAMGX_get_error_string = nullptr;

// 修改错误检查宏以使用函数指针
// #define AMGX_CHECK(ans)     \
//     {                       \
//         amgx_assert((ans)); \
//     }

// inline void amgx_assert(AMGX_RC code, AMGX_RC expected = AMGX_RC_OK)
// {
//     if (code != expected)
//     {
//         const int error_buf_size = 512;
//         char error_buf[error_buf_size];
//         pAMGX_get_error_string(code, error_buf, error_buf_size); // 使用函数指针
//         std::cerr << "AMGX error [" << code << "]: " << error_buf << std::endl;
//         exit(1);
//     }
// }

#define AMGX_CHECK(rc)   \
{                   \
    AMGX_RC err;\
    char msg[4096];\
    switch (err = (rc))\
    {\
    case AMGX_RC_OK:\
        break;\
    default:\
        fprintf(stderr, "AMGX error: file %s line %d/n", __FILE__, __LINE__);\
        pAMGX_get_error_string(err, msg, 4096);\
        fprintf(stderr, "AMGX error: %s\n", msg);\
        exit(1);\
        break;\
    }\
}
inline void amgx_assert(AMGX_RC code, const char *file, int line, AMGX_RC expected = AMGX_RC_OK)
{
    if (code != expected)
    {
        const int error_buf_size = 512;
        char error_buf[error_buf_size];
        pAMGX_get_error_string(code, error_buf, error_buf_size);
        std::cerr << "AMGX error [" << code << "]: " << error_buf
                  << "  (" << file << ":" << line << ")" << std::endl;
        exit(1);
    }
}
 
#define AMGX_CHECK(ans) amgx_assert((ans), __FILE__, __LINE__)
// 加载AMGX函数指针
bool LoadAMGXFunctions(const char *dllPath)
{
    HMODULE hDLL = LoadLibraryA(dllPath);
    if (!hDLL)
    {
        std::cerr << "Failed to load amgxsh.dll. Error: " << GetLastError() << std::endl;
        return false;
    }

    // 获取所有函数地址
    pAMGX_initialize = (AMGX_initialize_t)GetProcAddress(hDLL, "AMGX_initialize");
    pAMGX_finalize = (AMGX_finalize_t)GetProcAddress(hDLL, "AMGX_finalize");
    pAMGX_config_create = (AMGX_config_create_t)GetProcAddress(hDLL, "AMGX_config_create");
    pAMGX_resources_create_simple = (AMGX_resources_create_simple_t)GetProcAddress(hDLL, "AMGX_resources_create_simple");
    pAMGX_matrix_create = (AMGX_matrix_create_t)GetProcAddress(hDLL, "AMGX_matrix_create");
    pAMGX_vector_create = (AMGX_vector_create_t)GetProcAddress(hDLL, "AMGX_vector_create");
    pAMGX_matrix_upload_all = (AMGX_matrix_upload_all_t)GetProcAddress(hDLL, "AMGX_matrix_upload_all");
    pAMGX_vector_upload = (AMGX_vector_upload_t)GetProcAddress(hDLL, "AMGX_vector_upload");
    pAMGX_solver_create = (AMGX_solver_create_t)GetProcAddress(hDLL, "AMGX_solver_create");
    pAMGX_solver_setup = (AMGX_solver_setup_t)GetProcAddress(hDLL, "AMGX_solver_setup");
    pAMGX_solver_solve = (AMGX_solver_solve_t)GetProcAddress(hDLL, "AMGX_solver_solve");
    pAMGX_vector_download = (AMGX_vector_download_t)GetProcAddress(hDLL, "AMGX_vector_download");
    pAMGX_matrix_destroy = (AMGX_matrix_destroy_t)GetProcAddress(hDLL, "AMGX_matrix_destroy");
    pAMGX_vector_destroy = (AMGX_vector_destroy_t)GetProcAddress(hDLL, "AMGX_vector_destroy");
    pAMGX_solver_destroy = (AMGX_solver_destroy_t)GetProcAddress(hDLL, "AMGX_solver_destroy");
    pAMGX_resources_destroy = (AMGX_resources_destroy_t)GetProcAddress(hDLL, "AMGX_resources_destroy");
    pAMGX_config_destroy = (AMGX_config_destroy_t)GetProcAddress(hDLL, "AMGX_config_destroy");
    pAMGX_get_error_string = (AMGX_get_error_string_t)GetProcAddress(hDLL, "AMGX_get_error_string");

    // 检查所有函数是否成功加载
    if (!pAMGX_initialize || !pAMGX_finalize || !pAMGX_config_create)
    {
        std::cerr << "Failed to load one or more AMGX functions." << std::endl;
        FreeLibrary(hDLL);
        return false;
    }

    return true;
}

int main()
{
    // 指定amgxsh.dll的绝对路径
    const char *dllPath = "D:\\work\\amgx\\test\\install\\lib\\amgxsh.dll"; // 修改为实际路径
    if (!LoadAMGXFunctions(dllPath))
    {
        return 1;
    }

    // 初始化AMGX（通过函数指针调用）
    AMGX_CHECK(pAMGX_initialize());

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
    AMGX_CHECK(pAMGX_config_create(&cfg, config_string));

    // 创建 AMGX 资源
    AMGX_resources_handle rsrc;
    AMGX_CHECK(pAMGX_resources_create_simple(&rsrc, cfg));
    AMGX_matrix_handle A;
    AMGX_vector_handle b, x;
    AMGX_CHECK(pAMGX_matrix_create(&A, rsrc, AMGX_mode_dDDI));
    AMGX_CHECK(pAMGX_vector_create(&b, rsrc, AMGX_mode_dDDI));
    AMGX_CHECK(pAMGX_vector_create(&x, rsrc, AMGX_mode_dDDI));

    const int n = 5;
    const int nnz = 15;
    std::vector<int> row_ptrs = {0, 3, 6, 9, 12, 15};
    std::vector<int> col_indices = {0, 1, 2, 1, 2, 3, 1, 2, 3, 2, 3, 4, 2, 3, 4};
    std::vector<double> values = {1, 6, 7, 2, 8, 9, 10, 3, 11, 12, 4, 13, 14, 15, 5};

    AMGX_CHECK(pAMGX_matrix_upload_all(
        A, n, nnz, 1, 1,
        row_ptrs.data(), col_indices.data(), values.data(), nullptr));
    // 上传右侧向量 b（全 1）
    std::vector<double> b_host(n, 1.0);
    AMGX_CHECK(pAMGX_vector_upload(b, n, 1, b_host.data()));

    // 上传初始解向量 x（初始化为 0）
    std::vector<double> x_host(n, 0.0);
    AMGX_CHECK(pAMGX_vector_upload(x, n, 1, x_host.data()));
    // 创建求解器，设置求解器并求解线性系统
    AMGX_solver_handle solver;
    AMGX_CHECK(pAMGX_solver_create(&solver, rsrc, AMGX_mode_dDDI, cfg));
    AMGX_CHECK(pAMGX_solver_setup(solver, A));
    AMGX_CHECK(pAMGX_solver_solve(solver, b, x));
    AMGX_CHECK(pAMGX_vector_download(x, x_host.data()));
    // 输出解向量
    std::cout << "Solution vector:" << std::endl;
    for (double val : x_host)
    {
        std::cout << val << std::endl;
    }
    // 清理资源（通过函数指针调用）
    pAMGX_matrix_destroy(A);
    pAMGX_vector_destroy(b);
    pAMGX_vector_destroy(x);
    pAMGX_solver_destroy(solver);
    pAMGX_resources_destroy(rsrc);
    pAMGX_config_destroy(cfg);
    pAMGX_finalize();

    return 0;
}