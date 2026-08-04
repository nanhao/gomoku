#pragma once

#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "board_state.h"
#include "mcts_node.h"
#include "neural_net.h"

/** MCTS 搜索结果

对应 Python MCTSSearcher.search() 返回的三元组：
(visit_counts[225], root_value, root_policy[225])
*/
struct MctsResult {
    std::vector<int> visit_counts;     // 225 维访问计数（仅本搜索新增）
    float root_value;                  // 根节点平均价值 Q
    std::vector<float> root_policy;    // 225 维原始策略（噪声施加前）
};

/** 多线程 MCTS 搜索器

封装搜索树管理、Virtual Loss 协调、狄利克雷噪声、子树复用。
通过 pybind11 暴露为 Python cpp_mcts.MctsSearcher。

线程安全：搜索树通过 MctsNode 的 atomic + mutex 保证并发安全。
MCTS 工作线程数量由 num_threads 控制。
*/
class MctsSearcher {
public:
    /** 创建搜索器

    @param model_path TorchScript 模型文件路径 (.pt)
    @param num_simulations 每次搜索的总模拟次数，默认 200
    @param c_puct PUCT 探索系数，默认 2.0
    @param dirichlet_epsilon 狄利克雷噪声混合比例，默认 0.25
    @param dirichlet_alpha 狄利克雷分布浓度参数，默认 0.03
    @param num_threads MCTS 工作线程数，默认 4
    @param batch_size GPU 推理攒批大小，默认 16
    @param virtual_loss 虚拟损失值，默认 3.0
    @param use_gpu 是否使用 GPU 推理，默认 true
    */
    MctsSearcher(std::string model_path, int num_simulations,
                 double c_puct, double dirichlet_epsilon,
                 double dirichlet_alpha, int num_threads,
                 int batch_size, double virtual_loss, bool use_gpu);

    ~MctsSearcher();

    // 禁止拷贝和移动
    MctsSearcher(const MctsSearcher&) = delete;
    MctsSearcher& operator=(const MctsSearcher&) = delete;

    /** 对给定状态执行 MCTS 搜索

    @param state 226 维向量：前 225=棋盘（0=空,1=黑,2=白），末位=当前玩家
    @return 搜索结果
    */
    MctsResult search(const std::vector<int>& state);

    /** 将 action 对应子节点提升为新根，丢弃兄弟子树

    @param action 要推进的动作索引 0-224
    @return true 如果子节点存在并成功提升
    */
    bool advance_root(int action);

private:
    std::string model_path_;
    int num_simulations_;
    double c_puct_;
    double dirichlet_epsilon_;
    double dirichlet_alpha_;
    int num_threads_;
    double virtual_loss_;
    bool use_gpu_;

    // 神经网络推理引擎（共享）
    std::unique_ptr<NeuralNet> neural_net_;

    // 搜索树根节点
    MctsNode* root_;

    // 根边 N 快照（用于 get_result 中计算增量）
    std::vector<int> root_edge_n_snapshot_;

    // 线程安全：保护 root_ 指针的替换（advance_root 时）
    std::mutex root_mutex_;

    // 模拟计数器（跨线程共享）
    std::atomic<int> sim_counter_{0};

    // 原始策略（仅根节点存储，噪声施加前）
    std::vector<float> root_policy_;

    // 当前搜索的初始状态（供 worker 线程使用）
    std::vector<int> initial_board_;
    int initial_player_{1};

    // 当前搜索是否首次（用于 init_search 判断）
    bool first_search_{true};

    // --- 内部方法 ---

    /** 初始化搜索：创建根节点 + 狄利克雷噪声

    若根状态已终局，跳过模型预测，使用均匀分布策略。
    子树复用时保留已有子树 N 值，记录快照。
    */
    void init_search(const std::vector<int>& state);

    /** 对根节点施加狄利克雷噪声 */
    void add_dirichlet_noise(MctsNode* root,
                             const std::vector<int>& legal_actions);

    /** 单次 MCTS simulation（在 worker 线程中执行） */
    void run_single_simulation();

    /** 从搜索结果提取最终输出 */
    MctsResult get_result();

    /** 将棋盘 + 玩家转为 226 维 state 向量（内部辅助） */
    static std::vector<int> make_state(const std::vector<int>& board,
                                       int current_player);
};
