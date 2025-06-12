cmake_minimum_required(VERSION 3.15)
project(AMGX_Test)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置 AMGX 安装路径
set(AMGX_ROOT "D:/work/amgx/test/install")
message(STATUS "AMGX_ROOT: ${AMGX_ROOT}")

# 查找头文件
find_path(AMGX_INCLUDE_DIR amgx_c.h
    PATHS "${AMGX_ROOT}/include"
    REQUIRED
    NO_DEFAULT_PATH
)

# 查找动态库
find_library(AMGX_LIBRARY amgxsh
    PATHS "${AMGX_ROOT}/lib"
    REQUIRED
    NO_DEFAULT_PATH
)

# 添加可执行文件
#add_executable(AMGX_Test test.cpp)
add_executable(AMGX_Test main.cpp amgx_solver.cpp)
add_executable(Json_test Json.cpp)
add_executable(Diagonal_test diagonal_test.cpp)
add_executable(test_angx camgx.cpp)
# 为 AMGX_Test 添加包含目录和链接库
target_include_directories(AMGX_Test PRIVATE ${AMGX_INCLUDE_DIR})
target_link_libraries(AMGX_Test PRIVATE ${AMGX_LIBRARY})

# 为 Json_test 添加包含目录和链接库（如果 Json_test 也需要使用 AMGX 库）
target_include_directories(Json_test PRIVATE ${AMGX_INCLUDE_DIR})
target_link_libraries(Json_test PRIVATE ${AMGX_LIBRARY})

target_include_directories(Diagonal_test PRIVATE ${AMGX_INCLUDE_DIR})
target_link_libraries(Diagonal_test PRIVATE ${AMGX_LIBRARY})

# 确保 AMGX_Test 的输出目录存在
add_custom_command(TARGET AMGX_Test PRE_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:AMGX_Test>
)

# 构建后复制 DLL 到 AMGX_Test 的输出目录
add_custom_command(TARGET AMGX_Test POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
        "${AMGX_ROOT}/lib/amgxsh.dll"
        $<TARGET_FILE_DIR:AMGX_Test>
    COMMENT "Copying amgxsh.dll to output directory for AMGX_Test"
)

# 同样，确保 Json_test 的输出目录存在
add_custom_command(TARGET Json_test PRE_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:Json_test>
)

# 构建后复制 DLL 到 Json_test 的输出目录
add_custom_command(TARGET Json_test POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
        "${AMGX_ROOT}/lib/amgxsh.dll"
        $<TARGET_FILE_DIR:Json_test>
    COMMENT "Copying amgxsh.dll to output directory for Json_test"
)

add_custom_command(TARGET Diagonal_test PRE_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:Diagonal_test>
)

# 构建后复制 DLL 到 Diagonal_test 的输出目录
add_custom_command(TARGET Diagonal_test POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
        "${AMGX_ROOT}/lib/amgxsh.dll"
        $<TARGET_FILE_DIR:Diagonal_test>
    COMMENT "Copying amgxsh.dll to output directory for Diagonal_test"
)


####################################################
# 并行程序测试

# 并行程序测试

# 查找 MPI
find_package(MPI REQUIRED)
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
add_executable(par_amgx par4_test.cpp)

target_include_directories(par_amgx PRIVATE 
    ${AMGX_INCLUDE_DIR}
    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8/include"
    ${MPI_C_INCLUDE_PATH}
)
target_link_libraries(par_amgx PRIVATE ${AMGX_LIBRARY} ${MPI_C_LIBRARIES} ${CUDA_CUDART_LIBRARY})
target_compile_options(par_amgx PRIVATE ${MPI_C_COMPILE_FLAGS})

# 确保 par_amgx 的输出目录存在
add_custom_command(TARGET par_amgx PRE_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:par_amgx>
)

# 构建后复制 DLL 到 par_amgx 的输出目录
add_custom_command(TARGET par_amgx POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
        "${AMGX_ROOT}/lib/amgxsh.dll"
        $<TARGET_FILE_DIR:par_amgx>
    COMMENT "Copying amgxsh.dll to output directory for par_amgx"
)


#############################################