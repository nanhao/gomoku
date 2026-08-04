#include "mcts_searcher.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <future>
#include <random>
#include <stdexcept>

#include "thread_pool.h"

// =====================================================================
// 构造 & 析构
// =====================================================================

MctsSearcher::MctsSearcher(std::string model_path, int num_simulations,
                           double c_puct, double dirichlet_epsilon,
                           double dirichlet_alpha, int num_threads,
                           int batch_size, double virtual_loss, bool use_gpu)
    : model_path_(std::move(model_path))
    , num_simulations_(num_simulations)
    , c_puct_(c_puct)
    , dirichlet_epsilon_(dirichlet_epsilon)
    , dirichlet_alpha_(dirichlet_alpha)
    , num_threads_(num_threads)
    , virtual_loss_(virtual_loss)
    , use_gpu_(use_gpu)
    , root_(nullptr)
    , root_edge_n_snapshot_(225, 0)
    , root_policy_(225, 1.0f / 225.0f)
{
    // 创建共享的神经网络推理引擎
    neural_net_ = std::make_unique<NeuralNet>(
        model_path_, batch_size, use_gpu_);
}

MctsSearcher::~MctsSearcher() {
    std::lock_guard<std::mutex> lock(root_mutex_);
    delete root_;
    root_ = nullptr;
}

// =====================================================================
// 公开 API: search
// =====================================================================

MctsResult MctsSearcher::search(const std::vector<int>& state) {
    // 校验输入
    if (state.size() != 226) {
        throw std::runtime_error(
            "MctsSearcher::search: state 长度必须为 226, 实际 "
            + std::to_string(state.size()));
    }

    // 保存初始状态供 worker 线程使用
    initial_board_ = std::vector<int>(state.begin(), state.begin() + 225);
    initial_player_ = state[225];

    // 初始化搜索（创建/复用根节点，施加狄利克雷噪声）
    init_search(state);

    // 重置模拟计数器
    sim_counter_ = 0;

    // 启动全局线程池的 MCTS 工作线程（进程生命周期内只创建一次）
    MctsThreadPool::instance().run_parallel([this] {
        this->run_single_simulation();
    });

    // 提取结果
    return get_result();
}

// =====================================================================
// 公开 API: advance_root
// =====================================================================

bool MctsSearcher::advance_root(int action) {
    std::lock_guard<std::mutex> lock(root_mutex_);

    if (root_ == nullptr) return false;
    if (action < 0 || action >= 225) return false;

    // 查找目标子节点
    MctsNode* child = root_->get_child(action);
    if (child == nullptr) return false;

    // 1. 从旧根中分离子节点（使其成为独立树）
    child->detach_from_parent();

    // 2. 删除旧根（兄弟子树也会被递归删除）
    delete root_;

    // 3. 将子节点设为新根
    root_ = child;

    return true;
}

// =====================================================================
// 内部: init_search
// =====================================================================

void MctsSearcher::init_search(const std::vector<int>& state) {
    std::vector<int> board(state.begin(), state.begin() + 225);
    int current_player = state[225];

    std::lock_guard<std::mutex> lock(root_mutex_);

    // 检查是否子树复用：根节点已存在
    if (root_ != nullptr) {
        // 子树复用场景：记录当前 N 快照
        root_edge_n_snapshot_ = root_->get_visit_counts();

        // 重新施加狄利克雷噪声
        // 获取合法动作列表
        std::vector<int> legal_actions;
        for (int i = 0; i < 225; ++i) {
            if (board[i] == 0) legal_actions.push_back(i);
        }
        add_dirichlet_noise(root_, legal_actions);
        return;
    }

    // 全新搜索：创建根节点
    BoardState root_state(board, current_player);

    if (root_state.is_terminal()) {
        // 终局状态：无合法动作，使用均匀分布策略
        root_policy_ = std::vector<float>(225, 1.0f / 225.0f);
        root_ = new MctsNode(nullptr, 1.0);
        root_->set_root_policy(root_policy_);
        root_edge_n_snapshot_ = std::vector<int>(225, 0);
        return;
    }

    // 神经网络评估根状态
    std::vector<float> encoded = root_state.encode_for_network();
    NeuralNetResult nn_result = neural_net_->predict(encoded).get();

    // 保存原始策略（噪声施加前）
    root_policy_ = nn_result.policy;

    // 创建根节点
    root_ = new MctsNode(nullptr, 1.0);
    root_->set_root_policy(root_policy_);

    // 为所有合法动作创建边（子节点）
    std::vector<int> legal_actions = root_state.get_legal_actions();
    std::vector<float> priors;
    priors.reserve(legal_actions.size());
    for (int action : legal_actions) {
        priors.push_back(root_policy_[action]);
    }
    root_->expand(priors, legal_actions);

    // 记录 N 快照（全新树，全为 0）
    root_edge_n_snapshot_ = std::vector<int>(225, 0);

    // 施加狄利克雷噪声
    add_dirichlet_noise(root_, legal_actions);
}

// =====================================================================
// 内部: add_dirichlet_noise
// =====================================================================

void MctsSearcher::add_dirichlet_noise(
    MctsNode* root, const std::vector<int>& legal_actions)
{
    if (legal_actions.empty()) return;

    // 生成 Dirichlet 分布
    size_t n = legal_actions.size();
    std::vector<double> noise(n);

    // 使用 Gamma 分布生成 Dirichlet 样本
    // Dirichlet(alpha, alpha, ..., alpha) ~ Gamma(alpha, 1) normalized
    std::random_device rd;
    std::mt19937 gen(rd());
    std::gamma_distribution<double> gamma(dirichlet_alpha_, 1.0);

    double sum = 0.0;
    for (size_t i = 0; i < n; ++i) {
        noise[i] = gamma(gen);
        sum += noise[i];
    }
    for (size_t i = 0; i < n; ++i) {
        noise[i] /= sum;
    }

    // 混合噪声：P' = (1-eps) * root_policy + eps * noise
    for (size_t i = 0; i < n; ++i) {
        int action = legal_actions[i];
        float base_p = root_policy_[action];
        float mixed_p = static_cast<float>(
            (1.0 - dirichlet_epsilon_) * base_p
            + dirichlet_epsilon_ * noise[i]);

        // 更新子节点的 prior_prob（噪声混合后）
        MctsNode* child = root->get_child(action);
        if (child != nullptr) {
            child->set_prior(static_cast<double>(mixed_p));
        }
    }
}

// =====================================================================
// 内部: run_single_simulation
// =====================================================================

void MctsSearcher::run_single_simulation() {
    while (true) {
        int my_sim = sim_counter_.fetch_add(1);
        if (my_sim >= num_simulations_) break;

        // 获取根节点和初始状态
        MctsNode* root;
        BoardState root_state(initial_board_, initial_player_);
        {
            std::lock_guard<std::mutex> lock(root_mutex_);
            root = root_;
        }
        if (root == nullptr) continue;

        // === 阶段 1: Selection（沿树下降至叶节点）===
        // 追踪路径上的状态变化
        MctsNode* current = root;
        BoardState current_state = root_state;

        while (true) {
            if (current_state.is_terminal()) {
                // 当前节点已终局 → 直接回传终局价值
                // 注意：类似 NN value，terminal_val 需取反
                float terminal_val = current_state.terminal_value();
                current->backup(-terminal_val);
                break;
            }

            int best_action = current->select_child(c_puct_, virtual_loss_);
            if (best_action < 0) {
                // 无合法动作（终局但 is_terminal 未检测到的边界情况）
                break;
            }

            MctsNode* child = current->get_child(best_action);

            if (child->is_leaf()) {
                // 到达叶节点（未展开）
                // 构造子棋盘状态
                BoardState child_state =
                    current_state.apply_action(best_action);

                if (child_state.is_terminal()) {
                    // 子棋盘已终局 → 不展开，直接回传
                    // 注意：terminal_val 是从子状态视角的价值，需取反
                    float terminal_val = child_state.terminal_value();
                    child->backup(-terminal_val);
                } else {
                    // === 阶段 2: Evaluation（神经网络评估）===
                    std::vector<float> encoded =
                        child_state.encode_for_network();
                    auto future = neural_net_->predict(encoded);
                    NeuralNetResult nn_result = future.get();

                    // === 阶段 3: Expansion（展开子节点）===
                    std::vector<int> legal_actions =
                        child_state.get_legal_actions();
                    std::vector<float> priors;
                    priors.reserve(legal_actions.size());
                    for (int a : legal_actions) {
                        priors.push_back(nn_result.policy[a]);
                    }
                    child->expand(priors, legal_actions);

                    // === 阶段 4: Backup（回传更新）===
                    // 负值传入：NN 价值来自子节点视角，需取反
                    child->backup(-nn_result.value);
                }
                break;
            }

            // 继续下降
            current = child;
            current_state = current_state.apply_action(best_action);
        }
    }
}

// =====================================================================
// 内部: get_result
// =====================================================================

MctsResult MctsSearcher::get_result() {
    std::lock_guard<std::mutex> lock(root_mutex_);

    MctsResult result;
    result.root_policy = root_policy_;

    if (root_ == nullptr) {
        result.visit_counts = std::vector<int>(225, 0);
        result.root_value = 0.0f;
        return result;
    }

    // 访问计数：仅报告本搜索新增（N 增量）
    std::vector<int> current_counts = root_->get_visit_counts();
    result.visit_counts.resize(225, 0);
    for (int i = 0; i < 225; ++i) {
        int snapshot = (i < (int)root_edge_n_snapshot_.size())
                       ? root_edge_n_snapshot_[i] : 0;
        result.visit_counts[i] = current_counts[i] - snapshot;
        if (result.visit_counts[i] < 0) result.visit_counts[i] = 0;
    }

    // 根节点平均价值
    result.root_value = root_->get_root_value();

    return result;
}

// =====================================================================
// 辅助
// =====================================================================

std::vector<int> MctsSearcher::make_state(
    const std::vector<int>& board, int current_player)
{
    std::vector<int> state = board;
    state.push_back(current_player);
    return state;
}
