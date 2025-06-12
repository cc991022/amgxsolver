#include <mpi.h>
#include <amgx_c.h>
#include <vector>
#include <set>
#include <algorithm>
#include<iostream>

#define AMGX_CHECK(ans) { amgx_assert((ans)); }
inline void amgx_assert(AMGX_RC code) {
    if (code != AMGX_RC_OK) {
        char msg[512];
        AMGX_get_error_string(code, msg, 512);
        std::cerr << "AMGX error: " << msg << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // 示例：4x4 矩阵（全局 CSR 格式）
    int global_rows = 4;
    int global_row_ptr[5] = {0, 4, 6, 9, 12};
    int global_col_idx[12] = {0, 1, 2, 3, 1, 3, 0, 1, 2, 0, 2, 3};
    double global_vals[12] = {1, 2, 5, 7, 4, 9, 10, 11, 13, 12, 15, 14};

    // (1) 自动划分矩阵
    int local_rows = global_rows / size;
    int remainder = global_rows % size;
    if (rank < remainder) local_rows++;

    int start_row = (rank < remainder) 
        ? rank * local_rows 
        : remainder * (local_rows + 1) + (rank - remainder) * local_rows;
    int end_row = start_row + local_rows;

    std::vector<int> local_row_ptr(local_rows + 1);
    std::vector<int> local_col_idx;
    std::vector<double> local_vals;

    local_row_ptr[0] = 0;
    for (int i = 0; i < local_rows; i++) {
        int global_i = start_row + i;
        int row_nnz = global_row_ptr[global_i + 1] - global_row_ptr[global_i];
        local_row_ptr[i + 1] = local_row_ptr[i] + row_nnz;

        for (int j = global_row_ptr[global_i]; j < global_row_ptr[global_i + 1]; j++) {
            local_col_idx.push_back(global_col_idx[j]);
            local_vals.push_back(global_vals[j]);
        }
    }

    // (2) 构建通信映射
    std::set<int> neighbors;
    for (int col : local_col_idx) {
        if (col < start_row || col >= end_row) {
            int owner_rank = (col < remainder) 
                ? col / (local_rows + 1) 
                : remainder + (col - remainder) / local_rows;
            neighbors.insert(owner_rank);
        }
    }
    int num_neighbors = neighbors.size();
    std::vector<int> neighbor_ranks(neighbors.begin(), neighbors.end());

    // (2.1) 计算 send_maps
    std::vector<int> send_sizes(num_neighbors, 0);
    std::vector<std::vector<int>> send_maps(num_neighbors);
    for (int i = 0; i < local_rows; i++) {
        int global_i = start_row + i;
        for (int j = global_row_ptr[global_i]; j < global_row_ptr[global_i + 1]; j++) {
            int col = global_col_idx[j];
            if (col < start_row || col >= end_row) {
                int owner_rank = (col < remainder) 
                    ? col / (local_rows + 1) 
                    : remainder + (col - remainder) / local_rows;
                auto it = std::find(neighbor_ranks.begin(), neighbor_ranks.end(), owner_rank);
                int neighbor_idx = std::distance(neighbor_ranks.begin(), it);
                send_maps[neighbor_idx].push_back(i);
                send_sizes[neighbor_idx]++;
            }
        }
    }

    // (2.2) 计算 recv_maps
    std::vector<int> recv_sizes(num_neighbors, 0);
    std::vector<std::vector<int>> recv_maps(num_neighbors);
    std::set<int> needed_rows;
    for (int col : local_col_idx) {
        if (col < start_row || col >= end_row) {
            needed_rows.insert(col);
        }
    }
    for (int global_row : needed_rows) {
        int owner_rank = (global_row < remainder) 
            ? global_row / (local_rows + 1) 
            : remainder + (global_row - remainder) / local_rows;
        auto it = std::find(neighbor_ranks.begin(), neighbor_ranks.end(), owner_rank);
        int neighbor_idx = std::distance(neighbor_ranks.begin(), it);
        recv_maps[neighbor_idx].push_back(global_row - start_row);
        recv_sizes[neighbor_idx]++;
    }

    // (3) 初始化 AMGX
    AMGX_initialize();
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
    AMGX_resources_handle rsrc;
    int device_ids[1] = {0};
    AMGX_resources_create(&rsrc, cfg, &comm, 1, device_ids);

    // (4) 上传本地矩阵
    AMGX_matrix_handle A;
    AMGX_matrix_create(&A, rsrc, AMGX_mode_dDDI);
    AMGX_matrix_upload_all(
        A,
        local_rows,
        local_col_idx.size(),
        1, 1,
        local_row_ptr.data(),
        local_col_idx.data(),
        local_vals.data(),
        nullptr
    );

    // (5) 设置通信映射
    std::vector<const int*> send_maps_ptrs(num_neighbors);
    std::vector<const int*> recv_maps_ptrs(num_neighbors);
    for (int i = 0; i < num_neighbors; i++) {
        send_maps_ptrs[i] = send_maps[i].data();
        recv_maps_ptrs[i] = recv_maps[i].data();
    }

    std::cout<<"1"<<std::endl;

    AMGX_matrix_comm_from_maps_one_ring(
        A,
        1,
        num_neighbors,
        neighbor_ranks.data(),
        send_sizes.data(),
        send_maps_ptrs.data(),
        recv_sizes.data(),
        recv_maps_ptrs.data()
    );
    std::cout<<"2"<<std::endl;

    // (6) 上传向量并求解
    AMGX_vector_handle x, b;
    AMGX_vector_create(&x, rsrc, AMGX_mode_dDDI);
    AMGX_vector_create(&b, rsrc, AMGX_mode_dDDI);

    std::vector<double> rhs_local(local_rows, 1.0);  // 假设 RHS 全 1
    std::vector<double> x_init(local_rows, 0.0);     // 初始解全 0
    
    AMGX_vector_upload(b, local_rows, 1, rhs_local.data());
    AMGX_vector_upload(x, local_rows, 1, x_init.data());
    AMGX_vector_bind(b, A);
    AMGX_vector_bind(x, A);
    std::cout<<"3"<<std::endl;
    AMGX_solver_handle solver;
    AMGX_solver_create(&solver, rsrc, AMGX_mode_dDDI, cfg);
    AMGX_solver_setup(solver, A);
    AMGX_solver_solve(solver, b, x);
    std::cout<<"4"<<std::endl;
    // (7) 下载并打印结果
    std::vector<double> x_local(local_rows);
    AMGX_vector_download(x, x_local.data());

    for (int r = 0; r < size; r++) {
        if (r == rank) {
            std::cout << "Rank " << rank << " solution: ";
            for (double val : x_local) std::cout << val << " ";
            std::cout << std::endl;
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    // (8) 清理资源
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