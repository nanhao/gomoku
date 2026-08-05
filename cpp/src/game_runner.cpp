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


GameRunner::GameRunner(const std::string& model_path,
                       int num_games,
                       const GameConfig& config,
                       int num_workers,
                       bool collect_samples)
    : model1_path_(model_path)
    , model2_path_(model_path)  // 自对弈：双方用同一模型
    , num_games_(num_games)
    , num_workers_(num_workers > 0 ? num_workers : num_games)
    , config_(config)
    , is_self_play_(true)
    , collect_samples_(collect_samples)
{
    if (num_games_ < 1) {
        throw std::runtime_error(
            "GameRunner: num_games must be >= 1, got "
            + std::to_string(num_games_));
    }

    // 初始化全局资源（与对战模式相同）
    MctsThreadPool::initialize(g_mcts_config.num_threads);
    std::cout << "[GameRunner] 线程池已初始化, size="
              << MctsThreadPool::instance().size() << std::endl;

    NeuralNet::initialize(model_path, config_.use_gpu);
    std::cout << "[GameRunner] NeuralNet 已初始化, model="
              << model_path << std::endl;

    std::cout << "[GameRunner] 自对弈模式, 游戏线程数=" << num_workers_
              << " (对局数=" << num_games_
              << ", 采集样本=" << (collect_samples ? "是" : "否") << ")"
              << std::endl;

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
    // 1. 确定执方
    //    对战模式：随机分配；自对弈模式：交替分配
    int model1_color;
    int model2_color;

    if (is_self_play_) {
        // 自对弈交替模式：game_index 偶数 → model1 执黑
        if (game_index % 2 == 0) {
            model1_color = 1;  // 黑
            model2_color = 2;  // 白
        } else {
            model1_color = 2;  // 白
            model2_color = 1;  // 黑
        }
    } else {
        // 随机确定 model1 执方（1=黑/2=白），model2 执相反方
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<int> color_dist(1, 2);
        model1_color = color_dist(gen);
        model2_color = 3 - model1_color;  // 1↔2
    }

    // 2. 获取共享资源
    MctsThreadPool& pool = MctsThreadPool::instance();
    NeuralNet& nn = NeuralNet::instance();

    // 3. 创建两个 MctsSearcher（共享线程池和 GPU）
    //    自对弈模式下 model1_path == model2_path，两个 searcher 使用同一模型
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

    // 样本采集：存储原始样本三元组 (state_float, visit_counts, current_player)
    struct RawSampleRecord {
        std::vector<float> state_float;   // 226 维
        std::vector<int> visit_counts;    // 225 维
        int player;                       // 落子方 1=黑 2=白
    };
    std::vector<RawSampleRecord> raw_samples;

    // 5. 对局主循环
    std::string last_model_path;  // 追踪上次加载的模型路径，避免重复切换
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

        // 仅在模型路径变化时切换（幂等优化，避免不必要的加载）
        if (current_model_path != last_model_path) {
            nn.set_model(current_model_path);
            last_model_path = current_model_path;
        }

        // 若开启样本采集，记录落子前的状态
        if (collect_samples_) {
            RawSampleRecord rec;
            // 将 state 转为 float（226 维）
            for (int v : state) {
                rec.state_float.push_back(static_cast<float>(v));
            }
            rec.player = current_player;
            rec.visit_counts.resize(225, 0);  // 稍后填充
            raw_samples.push_back(std::move(rec));
        }

        // MCTS 搜索
        MctsResult mcts_result = current_searcher->search(state);

        // 若开启样本采集，保存 visit_counts
        if (collect_samples_ && !raw_samples.empty()) {
            raw_samples.back().visit_counts = mcts_result.visit_counts;
        }

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

    // 6. 样本生成（仅在 collect_samples 且非平局时）
    std::vector<GameSample> game_samples;
    if (collect_samples_ && winner != 0) {
        for (auto& rec : raw_samples) {
            // 计算 z：若 winner == rec.player 则 +1，否则 -1
            float z = (winner == rec.player) ? 1.0f : -1.0f;

            // 归一化 pi：pi[i] = N[i] / ΣN
            int total_visits = 0;
            for (int v : rec.visit_counts) {
                total_visits += v;
            }

            GameSample raw_sample;
            raw_sample.state = std::move(rec.state_float);
            raw_sample.pi.resize(225);
            if (total_visits > 0) {
                for (int i = 0; i < 225; ++i) {
                    raw_sample.pi[i] = static_cast<float>(rec.visit_counts[i])
                                     / static_cast<float>(total_visits);
                }
            } else {
                // 退化：所有合法位置均分
                float uniform = 1.0f / 225.0f;
                for (int i = 0; i < 225; ++i) {
                    raw_sample.pi[i] = uniform;
                }
            }
            raw_sample.z = z;

            // D4 增强：每条原始样本产生 8 条变体
            std::vector<GameSample> augmented =
                SampleAugmentor::augment(raw_sample);
            for (auto& aug : augmented) {
                game_samples.push_back(std::move(aug));
            }
        }
    }

    // 7. 记录结果
    {
        GameResult result;
        result.game_index = game_index;
        result.winner = winner;
        result.model1_color = model1_color;
        result.model2_color = model2_color;
        result.move_count = move_count;
        result.samples = std::move(game_samples);

        std::lock_guard<std::mutex> lock(results_mutex_);
        results_[game_index] = std::move(result);
    }

    // 8. 更新完成计数
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
