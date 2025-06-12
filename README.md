# 安装说明

在运行 AMGX 的测试文件（如生成的可执行文件 `.exe`）时，需要确保 `amgxsh.dll` 文件与可执行文件位于同一目录下，否则无法正常调用 AMGX 库。

## 关于 CMakeLists 的配置

1. 在 `CMakeLists.txt` 中，已配置在项目编译时将 `amgxsh.dll` 文件自动复制到 `build/Debug` 目录下，确保运行环境完整。
2. 需要注意的是，`CMakeLists.txt` 中将 AMGX 安装路径写死为当前项目目录下的 `install` 文件夹。因此，AMGX 必须安装在该路径下才能正常编译和运行。

## 注意事项

- 请确保 AMGX 安装路径正确，并与 `CMakeLists.txt` 中的配置一致。
- 如果需要更改 AMGX 的安装路径，请同时修改 `CMakeLists.txt` 中的相关配置。