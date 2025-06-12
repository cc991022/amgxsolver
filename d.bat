@echo off
REM 删除 build 目录（如果存在）
echo 正在检查 build 目录...
if exist build (
    echo 删除 build 目录...
    rmdir /s /q build
) else (
    echo build 目录不存在，跳过删除步骤.
)

REM 生成工程
echo 执行 cmake -B build ...
cmake -B build

REM 编译工程
echo 执行 cmake --build build ...
cmake --build build

REM 进入 build\Debug 目录并运行测试程序
echo 进入 build\Debug 目录...
cd build\Debug
echo 执行 par_amgx.exe 测试程序...
mpiexec -n 2 .\par_amgx.exe
pause