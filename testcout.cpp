#include <stdio.h>
#include <amgx_c.h>
#ifdef _WIN32
#include <windows.h>
#endif

int main(int argc, char **argv)
{
#ifdef _WIN32
    // 设置控制台输出为 UTF-8 编码
    SetConsoleOutputCP(CP_UTF8);
#endif

    // 初始化 AMGX 库
    if (AMGX_initialize() != AMGX_RC_OK) {
        printf("AMGX 初始化失败\n");
        return -1;
    }
    
    // 此处可以加入其他 AMGX 调用测试代码，本示例不涉及求解过程

    // 释放 AMGX 库
    if (AMGX_finalize() != AMGX_RC_OK) {
        printf("AMGX 释放失败\n");
        return -1;
    }
    
    printf("AMGX 调用测试成功\n");
    return 0;
}
