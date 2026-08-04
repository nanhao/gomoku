cmake_minimum_required(VERSION 3.18)
project(cpp_mcts LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- Find LibTorch ---
# 用户需设置 Torch_DIR 环境变量或通过 -DCMAKE_PREFIX_PATH 传入
find_package(Torch REQUIRED)

# --- Find pybind11 (optional, only needed for Python extension) ---
find_package(pybind11 QUIET)

# --- MCTS 核心静态库 ---
add_library(mcts_core STATIC
    src/board_state.cpp
    src/mcts_node.cpp
    src/neural_net.cpp
    src/mcts_searcher.cpp
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
endfunction()

add_unit_test(test_board_state test_board_state.cpp)
add_unit_test(test_mcts_node test_mcts_node.cpp)
add_unit_test(test_neural_net test_neural_net.cpp)
add_unit_test(test_mcts_searcher test_mcts_searcher.cpp)

# --- 集成测试 ---
function(add_integration_test test_name test_file)
    add_executable(${test_name}
        tests/integration/${test_file}
    )
    target_link_libraries(${test_name} PRIVATE mcts_core)
    add_test(NAME ${test_name} COMMAND ${test_name})
endfunction()

add_integration_test(test_mcts_integration test_mcts_integration.cpp)
