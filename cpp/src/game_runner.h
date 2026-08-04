#pragma once

#include <atomic>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

/** 游戏配置参数

所有 MCTS 搜索参数 + 温度采样参数集中管理。
Python 端通过 pybind11 设置字段值后传入 C++。
*/
struct GameConfig {
    int num_simulations = 800;
    double c_puct = 2.0;
    double dirichlet_epsilon = 0.25;
    double dirichlet_alpha = 0.03;
    int batch_size = 16;
    double virtual_loss = 3.0;
    double temperature = 1.0;
    int temp_steps = 8;
    bool use_gpu = true;
};

/** 单局对战结果

数据结构由 C++ 游戏线程填充，Python 端通过 pybind11 读取。\nmodel1_won / model2_won 由 Python 端根据 winner 和执方计算。
*/
struct GameResult {
    int game_index;       // 局编号 0..N-1
    int winner;           // 0=平局, 1=黑胜, 2=白胜
    int model1_color;     // model1 执方 1=黑, 2=白
    int model2_color;     // model2 执方
    int move_count;       // 本局总步数
};

/** 并行对局管理器

职责：
- 接收两个模型路径、对局数量、配置参数
- 初始化全局资源（线程池、NeuralNet）
- 启动 N 个游戏线程，每个线程运行一局完整对局
- 收集结果，提供进度查询

线程安全：get_completed_count / is_finished / get_results 均可多线程调用。
*/
class GameRunner {
public:
    /** 创建对局管理器

    @param model1_path 模型1 的 TorchScript 文件路径
    @param model2_path 模型2 的 TorchScript 文件路径
    @param num_games 对战局数（≥1）
    @param config 游戏配置参数
    @throws std::runtime_error 若模型文件不存在或加载失败
    */
    GameRunner(const std::string& model1_path,
               const std::string& model2_path,
               int num_games,
               const GameConfig& config);

    ~GameRunner();

    // 禁止拷贝和移动
    GameRunner(const GameRunner&) = delete;
    GameRunner& operator=(const GameRunner&) = delete;

    /** 启动所有游戏线程（非阻塞，立即返回） */
    void start();

    /** 进度查询（线程安全） */
    int get_completed_count() const;
    int get_total_count() const;

    /** 是否全部完成 */
    bool is_finished() const;

    /** 阻塞直到所有游戏线程完成 */
    void wait_completion();

    /** 获取所有结果（仅在 is_finished() 为 true 后调用）

    @return 按 game_index 排序的结果列表
    */
    std::vector<GameResult> get_results() const;

private:
    std::string model1_path_;
    std::string model2_path_;
    int num_games_;
    GameConfig config_;

    // 游戏线程
    std::vector<std::thread> game_threads_;

    // 已完成计数器（线程安全）
    std::atomic<int> completed_count_{0};

    // 结果存储（线程安全：start 前 resize，各线程写入不同索引）
    std::vector<GameResult> results_;
    mutable std::mutex results_mutex_;

    /** 单局对局主循环（由游戏线程执行）

    随机分配 model1/2 的执方，运行完整对局并写入结果。
    @param game_index 局编号 0..N-1
    */
    void run_single_game(int game_index);

    /** 温度采样：从 visit_counts 中按概率选择一个 action

    前 temp_steps 步：按 N^(1/τ) 权重采样
    temp_steps 之后：纯贪婪选择（N 最大者）

    @param visit_counts 225 维访问计数
    @param step 当前步数（从 0 开始）
    @param config 游戏配置（含 temperature 和 temp_steps）
    @return 选中的 action 索引
    */
    static int sample_action(const std::vector<int>& visit_counts,
                             int step,
                             const GameConfig& config);
};
