cmake_minimum_required(VERSION 3.18)
project(cpp_mcts LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# MSVC: 强制 UTF-8 编码（否则中文注释/字符串会报 C2001/C4819）
if(MSVC)
    add_compile_options(/utf-8 /wd4267 /wd4819)
endif()

# --- Find LibTorch ---
# 手动设置路径以绕过 CUDA Toolkit 依赖（使用 pip 安装的 PyTorch）
# 若需使用 find_package(Torch)，需安装 CUDA Toolkit
if(NOT DEFINED TORCH_ROOT)
    # 自动检测：通过 Python 获取 torch 安装路径
    execute_process(
        COMMAND python -c "import torch, os; print(os.path.dirname(torch.__file__))"
        OUTPUT_VARIABLE _TORCH_ROOT
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if(_TORCH_ROOT)
        set(TORCH_ROOT "${_TORCH_ROOT}")
    else()
        message(FATAL_ERROR "Cannot detect PyTorch. Set -DTORCH_ROOT=... or install torch.")
    endif()
endif()
message(STATUS "Using Torch root: ${TORCH_ROOT}")
set(TORCH_INCLUDE_DIRS "${TORCH_ROOT}/include" "${TORCH_ROOT}/include/torch/csrc/api/include")
set(TORCH_LIB_DIR "${TORCH_ROOT}/lib")

# 链接必要的 LibTorch 库（CPU + CUDA 动态加载支持）
# 链接 torch_cuda.lib 确保 CUDA 符号在编译时解析；运行时通过 PATH 查找 DLL。
# 需要将 torch\lib 加入 PATH 才能加载 torch_cuda.dll 及其依赖（cublas, cudnn 等）。
set(TORCH_LIBRARIES
    "${TORCH_LIB_DIR}/torch.lib"
    "${TORCH_LIB_DIR}/torch_cpu.lib"
    "${TORCH_LIB_DIR}/torch_cuda.lib"
    "${TORCH_LIB_DIR}/c10.lib"
    "${TORCH_LIB_DIR}/c10_cuda.lib"
)

# 添加 DLL 搜索路径（运行时需要）
link_directories(${TORCH_LIB_DIR})

# --- Find pybind11 (optional, only needed for Python extension) ---
find_package(pybind11 QUIET)

# --- MCTS 核心静态库 ---
add_library(mcts_core STATIC
    src/board_state.cpp
    src/mcts_config.cpp
    src/mcts_node.cpp
    src/neural_net.cpp
    src/mcts_searcher.cpp
    src/thread_pool.cpp
    src/game_runner.cpp
)

target_include_directories(mcts_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${TORCH_INCLUDE_DIRS}
)

target_link_libraries(mcts_core PUBLIC
    ${TORCH_LIBRARIES}
)

# --- Python 扩展模块 (pybind11) ---
if(pybind11_FOUND)
    pybind11_add_module(cpp_mcts
        src/bindings.cpp
    )
    target_link_libraries(cpp_mcts PRIVATE mcts_core)
    set_target_properties(cpp_mcts PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
        PREFIX ""
        SUFFIX ".pyd"
    )
endif()

# --- 单元测试 ---
enable_testing()

function(add_unit_test test_name test_file)
    add_executable(${test_name}
        tests/unit/${test_file}
    )
    target_link_libraries(${test_name} PRIVATE mcts_core)
    add_test(NAME ${test_name} COMMAND ${test_name})
    # 确保测试运行时能找到 LibTorch DLLs
    set_tests_properties(${test_name} PROPERTIES
        ENVIRONMENT_MODIFICATION "PATH=path_list_prepend:${TORCH_LIB_DIR}"
    )
endfunction()

add_unit_test(test_board_state test_board_state.cpp)
add_unit_test(test_mcts_node test_mcts_node.cpp)
add_unit_test(test_neural_net test_neural_net.cpp)
add_unit_test(test_mcts_searcher test_mcts_searcher.cpp)
add_unit_test(test_thread_pool_queue test_thread_pool_queue.cpp)
add_unit_test(test_neural_net_singleton test_neural_net_singleton.cpp)
add_unit_test(test_mcts_searcher_injection test_mcts_searcher_injection.cpp)
add_unit_test(test_game_runner_workers test_game_runner_workers.cpp)
add_unit_test(test_sample_augmentor test_sample_augmentor.cpp)
add_unit_test(test_neural_net_dual_model test_neural_net_dual_model.cpp)

# --- 集成测试 ---
function(add_integration_test test_name test_file)
    add_executable(${test_name}
        tests/integration/${test_file}
    )
    target_link_libraries(${test_name} PRIVATE mcts_core)
    add_test(NAME ${test_name} COMMAND ${test_name})
    # 确保测试运行时能找到 LibTorch DLLs
    set_tests_properties(${test_name} PROPERTIES
        ENVIRONMENT_MODIFICATION "PATH=path_list_prepend:${TORCH_LIB_DIR}"
    )
endfunction()

add_integration_test(test_mcts_integration test_mcts_integration.cpp)
