#include "mcts_node.h"

#include <algorithm>
#include <cmath>
#include <cfloat>
#include <limits>

MctsNode::MctsNode(MctsNode* parent, double prior_prob)
    : parent_(parent), p_sa_(prior_prob) {}

MctsNode::~MctsNode() {
    // 递归删除所有子节点
    for (auto& [action, child] : children_) {
        delete child;
    }
}

int MctsNode::select_child(double c_puct, double c_virtual_loss) {
    // 加锁：防止 select_child 期间 children_ 被 expand 修改
    std::lock_guard<std::mutex> lock(children_lock_);

    if (children_.empty()) return -1;

    double best_value = -std::numeric_limits<double>::max();
    int best_action = children_.begin()->first;
    MctsNode* best_child = children_.begin()->second;

    // sum_n_visited = n_visited + n_unobserved（包含进行中的模拟）
    unsigned int sum_visited = n_visited_ + n_unobserved_;

    for (auto& [action, child] : children_) {
        double cur_value = child->get_value(c_puct, c_virtual_loss, sum_visited);
        if (cur_value > best_value) {
            best_value = cur_value;
            best_action = action;
            best_child = child;
        }
    }

    // 施加 Virtual Loss：递增选中子节点和当前节点的 n_unobserved
    // Python 等价操作：edge.N += VL, node.N += VL
    ++best_child->n_unobserved_;
    ++best_child->virtual_loss_;
    ++n_unobserved_;

    return best_action;
}

void MctsNode::expand(const std::vector<float>& action_priors,
                      const std::vector<int>& legal_actions) {
    std::lock_guard<std::mutex> lock(expand_lock_);

    if (!is_leaf_) return;  // 幂等：已展开则跳过

    do_expand_locked(action_priors, legal_actions);
    is_leaf_ = false;
}

bool MctsNode::expand_with_policy(const std::vector<float>& action_priors,
                                  const std::vector<int>& legal_actions,
                                  const std::vector<float>& full_policy) {
    // 展开与策略存储必须在同一临界区原子完成（P1-1）：
    // 并发 worker 可能同时判定本节点为叶双双进入展开流程，
    // 锁外双写 root_policy_ 属数据竞争（UB）；幂等返回决定唯一写入者。
    std::lock_guard<std::mutex> lock(expand_lock_);

    if (!is_leaf_) return false;  // 幂等：其他 worker 已完成展开

    do_expand_locked(action_priors, legal_actions);
    is_leaf_ = false;

    // 存储自身局面原始策略（噪声前 225 维），使本节点被提升为
    // 新根后 get_root_policy() 可返回正确策略（design/59 修复点 3）
    root_policy_ = full_policy;
    return true;
}

void MctsNode::do_expand_locked(const std::vector<float>& action_priors,
                                const std::vector<int>& legal_actions) {
    // 调用方须已持有 expand_lock_（expand / expand_with_policy 共用）
    for (size_t i = 0; i < legal_actions.size(); ++i) {
        int action = legal_actions[i];
        float prior = i < action_priors.size() ? action_priors[i] : 0.0f;

        // 跳过概率极小的动作
        if (std::fabs(prior) < FLT_EPSILON) continue;

        auto* child = new MctsNode(this, prior);
        children_[action] = child;
    }
}

void MctsNode::flip_value() {
    // 价值视角翻转：非根节点 q_sa_ 存父视角（边价值），被提升为根后
    // 需翻转为自身视角（根节点价值口径），与后续根特判取反的新累积混同
    // 前完成翻转，避免 root_value 向 0 漂移（design/59 修复点 4）。
    // 调用时机无并发（搜索模拟结束后/开始前），无需加锁。
    q_sa_ = -q_sa_;
}

void MctsNode::backup(float leaf_value) {
    // leaf_value 是从调用方视角传入的价值（即父节点视角的边价值）。
    //
    // 对于非根节点：q_sa_ 存储边价值（来自父节点视角），直接使用 leaf_value 更新。
    // 对于根节点：q_sa_ 存储根节点状态价值（来自根节点自身视角），
    //             需取反后更新（因为 leaf_value 来自子节点视角）。
    //
    // 传播规则：
    //   非根节点向父节点传播 -my_value，其中 my_value = leaf_value。
    //   父节点收到后，若为根节点则取反后更新，若非根节点则直接更新。
    //
    // 等价 Python 逻辑：_backup 中每层 v = -v 后更新边和节点。

    float my_value = (parent_ == nullptr) ? -leaf_value : leaf_value;

    // 撤销 Virtual Loss（所有节点，包括根节点）
    --virtual_loss_;
    --n_unobserved_;

    // 更新本节点的运行均值 Q = (N * Q + v) / (N + 1)
    int n = n_visited_.fetch_add(1);
    q_sa_ = (static_cast<double>(n) * q_sa_ + static_cast<double>(my_value))
            / static_cast<double>(n + 1);

    // 向父节点传播：父节点需要取反后的值
    if (parent_ != nullptr) {
        parent_->backup(-my_value);
    }
}

bool MctsNode::is_leaf() const {
    return is_leaf_;
}

std::vector<int> MctsNode::get_visit_counts() const {
    std::vector<int> counts(225, 0);
    std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(children_lock_));
    for (auto& [action, child] : children_) {
        counts[action] = child->n_visited_.load();
    }
    return counts;
}

float MctsNode::get_root_value() const {
    if (n_visited_ > 0) {
        return static_cast<float>(q_sa_);
    }
    return 0.0f;
}

std::vector<float> MctsNode::get_root_policy() const {
    if (root_policy_.empty()) {
        // 回退：从子节点的先验概率构建
        std::vector<float> policy(225, 0.0f);
        std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(children_lock_));
        for (auto& [action, child] : children_) {
            policy[action] = static_cast<float>(child->p_sa_);
        }
        return policy;
    }
    return root_policy_;
}

void MctsNode::set_root_policy(const std::vector<float>& policy) {
    root_policy_ = policy;
}

MctsNode* MctsNode::get_child(int action) const {
    std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(children_lock_));
    auto it = children_.find(action);
    if (it != children_.end()) {
        return it->second;
    }
    return nullptr;
}

double MctsNode::get_value(double c_puct, double c_virtual_loss,
                           unsigned int sum_n_visited) const {
    // PUCT 公式：
    //   U = c_puct * P * sqrt(sum_N) / (1 + N + N_unobserved)
    double u = c_puct * p_sa_ * std::sqrt(static_cast<double>(sum_n_visited))
               / (1.0 + n_visited_ + n_unobserved_);

    // Virtual Loss 惩罚
    double vir_loss = c_virtual_loss * virtual_loss_;

    if (n_visited_ == 0) {
        return u;
    }

    // Q值计算: (N*Q - VL) / N + U
    return (q_sa_ * n_visited_ - vir_loss) / n_visited_ + u;
}

void MctsNode::detach_from_parent() {
    if (parent_ == nullptr) return;

    // 从父节点的 children_ 中移除
    std::lock_guard<std::mutex> parent_lock(parent_->children_lock_);
    for (auto it = parent_->children_.begin();
         it != parent_->children_.end(); ++it) {
        if (it->second == this) {
            parent_->children_.erase(it);
            break;
        }
    }

    // 清除父节点关系
    parent_ = nullptr;
}
