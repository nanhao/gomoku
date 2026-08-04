#pragma once

/** C++ MCTS 模块运行时配置
 *
 * 所有可调参数从 mcts_config.json 动态加载，修改配置无需重新编译。
 *
 * 生命周期：
 *   1. bindings.cpp 在模块初始化时调用 Python json 模块读取 JSON
 *   2. 填充全局实例 g_mcts_config
 *   3. 所有 C++ 代码通过 g_mcts_config 访问配置
 *
 * Python 端对应配置：cpp/python/mcts_config.py（与 JSON 同步读取）
 * JSON 配置文件：   cpp/python/mcts_config.json
 */

struct MctsConfig {
    int num_simulations = 800;
    int num_threads = 32;
    double c_puct = 2.0;
    double dirichlet_epsilon = 0.25;
    double dirichlet_alpha = 0.03;
    int batch_size = 16;
    double virtual_loss = 3.0;
    bool use_gpu = true;
};

/// 全局运行时配置实例，由 bindings.cpp 在模块导入时填充
extern MctsConfig g_mcts_config;
