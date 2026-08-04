#include "game_runner.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>

#include "board_state.h"
#include "mcts_config.h"
#include "mcts_searcher.h"
#include "neural_net.h"
#include "thread_pool.h"

// =====================================================================
// 构造 & 析构
// =====================================================================

GameRunner::GameRunner(const std::string& model1_path,
                       const std::string& model2_path,
                       int num_games,
                       const GameConfig& config,
                       int num_workers)
    : model1_path_(model1_path)
    , model2_path_(model2_path)
    , num_games_(num_games)
    , num_workers_(num_workers > 0 ? num_workers : num_games)
    , config_(config)
{
    if (num_games_ < 1) {
        throw std::runtime_error(
            "GameRunner: num_games must be >= 1, got "
            + std::to_string(num_games_));
    }

    // 1. 初始化全局 MCTS 线程池（使用 JSON 配置中的 num_threads，
    //    而非对局数；call_once 保护，若已初始化则忽略）
    MctsThreadPool::initialize(g_mcts_config.num_threads);
    std::cout << "[GameRunner] 线程池已初始化, size="
              << MctsThreadPool::instance().size() << std::endl;

    // 2. 初始化全局 NeuralNet 单例（先用 model1 加载，攒批参数从 JSON 读取）
    NeuralNet::initialize(model1_path_, config_.use_gpu);
    std::cout << "[GameRunner] NeuralNet 已初始化, model="
              << model1_path_ << std::endl;

    std::cout << "[GameRunner] 游戏线程数=" << num_workers_
              << " (对局数=" << num_games_ << ")" << std::endl;

    // 3. 预分配结果数组（按 game_index 索引，线程安全写入）
    results_.resize(num_games_);
}

GameRunner::~GameRunner() {
    // 确保所有游戏线程已退出
    wait_completion();
    // 关闭全局资源
    NeuralNet::shutdown();
    MctsThreadPool::shutdown();
}

// =====================================================================
// 公开 API
// =====================================================================

void GameRunner::start() {
    game_threads_.reserve(num_workers_);
    for (int i = 0; i < num_workers_; ++i) {
        game_threads_.emplace_back(&GameRunner::run_worker_games, this, i);
    }
}

int GameRunner::get_completed_count() const {
    return completed_count_.load();
}

int GameRunner::get_total_count() const {
    return num_games_;
}

bool GameRunner::is_finished() const {
    return completed_count_.load() >= num_games_;
}

void GameRunner::wait_completion() {
    for (auto& t : game_threads_) {
        if (t.joinable()) {
            t.join();
        }
    }
}

std::vector<GameResult> GameRunner::get_results() const {
    std::lock_guard<std::mutex> lock(results_mutex_);
    return results_;
}

// =====================================================================
// 工作线程入口：轮询分发对局
// =====================================================================

void GameRunner::run_worker_games(int worker_index) {
    // 轮询分发：worker 0 负责 game 0, K, 2K, ...；worker 1 负责 game 1, K+1, ...
    for (int g = worker_index; g < num_games_; g += num_workers_) {
        run_single_game(g);
    }
}

// =====================================================================
// 单局对局主循环
// =====================================================================

void GameRunner::run_single_game(int game_index) {
    // 1. 随机确定 model1 执方（1=黑/2=白），model2 执相反方
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> color_dist(1, 2);
    int model1_color = color_dist(gen);
    int model2_color = 3 - model1_color;  // 1↔2

    // 2. 获取共享资源
    MctsThreadPool& pool = MctsThreadPool::instance();
    NeuralNet& nn = NeuralNet::instance();

    // 3. 创建两个 MctsSearcher（共享线程池和 GPU）
    // 注意：两个 searcher 使用同一 NeuralNet，搜索前需调用 set_model
    MctsSearcher searcher1(
        model1_path_, pool, nn,
        config_.num_simulations, config_.c_puct,
        config_.dirichlet_epsilon, config_.dirichlet_alpha,
        config_.virtual_loss);

    MctsSearcher searcher2(
        model2_path_, pool, nn,
        config_.num_simulations, config_.c_puct,
        config_.dirichlet_epsilon, config_.dirichlet_alpha,
        config_.virtual_loss);

    // 4. 初始化空棋盘，黑方先手
    std::vector<int> board(225, 0);
    int current_player = 1;  // 1=黑先

    // 准备 226 维 state（225 board + 1 player）
    std::vector<int> state = board;
    state.push_back(current_player);

    int move_count = 0;
    int winner = 0;  // 0=未决出

    // 5. 对局主循环
    while (move_count < 225) {
        // 确定当前落子方使用哪个模型
        MctsSearcher* current_searcher = nullptr;
        std::string current_model_path;

        if (current_player == model1_color) {
            current_searcher = &searcher1;
            current_model_path = model1_path_;
        } else {
            current_searcher = &searcher2;
            current_model_path = model2_path_;
        }

        // 切换 NeuralNet 到当前模型（线程安全）
        nn.set_model(current_model_path);

        // MCTS 搜索
        MctsResult mcts_result = current_searcher->search(state);

        // 温度采样选动作
        int action = sample_action(mcts_result.visit_counts, move_count, config_);

        // 落子
        board[action] = current_player;
        move_count++;

        // 终局判定
        BoardState bs(board, current_player);
        if (bs.has_winner()) {
            winner = current_player;
            break;
        }
        if (bs.is_full()) {
            winner = 0;  // 平局
            break;
        }

        // 子树复用：推进根节点
        current_searcher->advance_root(action);

        // 切换玩家
        current_player = 3 - current_player;
        state = board;
        state.push_back(current_player);
    }

    // 6. 记录结果
    {
        GameResult result;
        result.game_index = game_index;
        result.winner = winner;
        result.model1_color = model1_color;
        result.model2_color = model2_color;
        result.move_count = move_count;

        std::lock_guard<std::mutex> lock(results_mutex_);
        results_[game_index] = result;
    }

    // 7. 更新完成计数
    completed_count_.fetch_add(1);
}

// =====================================================================
// 温度采样
// =====================================================================

int GameRunner::sample_action(const std::vector<int>& visit_counts,
                              int step,
                              const GameConfig& config)
{
    // 收集合法动作（visit_counts > 0 的位置）
    std::vector<int> valid_actions;
    std::vector<int> valid_counts;

    for (int i = 0; i < 225; ++i) {
        if (visit_counts[i] > 0) {
            valid_actions.push_back(i);
            valid_counts.push_back(visit_counts[i]);
        }
    }

    if (valid_actions.empty()) {
        // 退化：无合法动作（不应发生，返回 -1）
        return -1;
    }

    bool use_temperature = (step < config.temp_steps);

    if (use_temperature && config.temperature > 0.0) {
        // 温度采样：权重 = N^(1/τ)
        double tau = config.temperature;
        std::vector<double> weights;
        weights.reserve(valid_counts.size());

        for (int c : valid_counts) {
            double weight = std::pow(static_cast<double>(c), 1.0 / tau);
            weights.push_back(weight);
        }

        // 离散分布采样
        std::random_device rd;
        std::mt19937 gen(rd());
        std::discrete_distribution<int> dist(weights.begin(), weights.end());
        int idx = dist(gen);
        return valid_actions[idx];
    } else {
        // 贪婪选择：visit_counts 最大者
        int best_idx = 0;
        int best_count = valid_counts[0];
        for (size_t i = 1; i < valid_counts.size(); ++i) {
            if (valid_counts[i] > best_count) {
                best_count = valid_counts[i];
                best_idx = static_cast<int>(i);
            }
        }
        return valid_actions[best_idx];
    }
}
