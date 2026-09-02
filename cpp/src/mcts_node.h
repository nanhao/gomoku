#pragma once

#include <atomic>
#include <mutex>
#include <unordered_map>
#include <vector>

/** MCTS 搜索树节点
 *
 * 并发安全的树节点。每个节点维护：
 * - 子节点映射（action → MctsNode*）
 * - 访问统计（N, W, Q）
 * - Virtual Loss 计数器
 *
 * 线程安全：使用 std::atomic 保护计数器，std::mutex 保护结构修改。
 * parent 为原始指针，生命周期由 MctsSearcher 的 unique_ptr 树管理。
 */
class MctsNode {
public:
    /** 创建节点
     *
     * @param parent 父节点指针，根节点传 nullptr
     * @param prior_prob 先验概率 P(s,a)，根节点传 1.0
     */
    MctsNode(MctsNode* parent, double prior_prob);

    /** 析构：递归删除所有子节点 */
    ~MctsNode();

    // 禁止拷贝和移动（树结构用指针管理）
    MctsNode(const MctsNode&) = delete;
    MctsNode& operator=(const MctsNode&) = delete;

    /** PUCT 选择最优子节点
     *
     * 对选中子节点施加 Virtual Loss（N_unobserved++, virtual_loss++）。
     *
     * @param c_puct PUCT 探索系数
     * @param c_virtual_loss Virtual Loss 系数
     * @return 最优 action 索引，无合法子节点时返回 -1
     */
    int select_child(double c_puct, double c_virtual_loss);

    /** 展开节点，为所有合法动作创建子节点
     *
     * 仅叶节点可展开。重复调用幂等（不创建重复子节点）。
     *
     * @param action_priors 各动作的先验概率（需与 legal_actions 对应）
     * @param legal_actions 合法动作索引列表
     */
    void expand(const std::vector<float>& action_priors,
                const std::vector<int>& legal_actions);

    /** 展开节点并存储自身局面原始策略（锁内原子完成，design/59 修复点 3）
     *
     * 在 expand_lock_ 临界区内原子完成"展开 + 存储自身局面原始策略
     * （噪声前 225 维）"。与 expand 幂等语义一致：已展开返回 false 且
     * 不写策略；返回是否由本次调用真正展开。
     *
     * 并发正确性（P1-1）：存储必须与展开在同一临界区内完成——两个 worker
     * 可同时判定同一 child 为叶（is_leaf 检查无锁）双双进入展开流程，若在
     * expand 返回后锁外 set_root_policy，双写 std::vector 属数据竞争（UB）；
     * 由幂等返回决定唯一写入者。
     *
     * @param action_priors 各动作的先验概率（需与 legal_actions 对应）
     * @param legal_actions 合法动作索引列表
     * @param full_policy 本节点局面的完整 225 维原始策略（噪声施加前）
     * @return 是否由本次调用真正展开（已展开则 false）
     */
    bool expand_with_policy(const std::vector<float>& action_priors,
                            const std::vector<int>& legal_actions,
                            const std::vector<float>& full_policy);

    /** 翻转价值视角（design/59 修复点 4）
     *
     * q_sa_ = -q_sa_（n_visited_ 不变）。backup 数值链中：非根节点 q_sa_
     * 存父视角（边价值），根节点 q_sa_ 存自身视角（根特判取反）；节点被提升
     * 为新根后历史 q 需翻转，避免与新累积口径混同导致 root_value 漂移。
     * 调用时机无并发（搜索模拟结束后/开始前）。
     */
    void flip_value();

    /** 沿父链向上回传
     *
     * 每向上一层取反 value。自动撤销 Virtual Loss。
     * 更新 N += 1, W += v, Q = W/N。
     *
     * @param leaf_value 叶节点局面价值（当前玩家视角）
     */
    void backup(float leaf_value);

    /** 是否为叶节点（未展开） */
    bool is_leaf() const;

    /** 获取所有子节点的 N 值，225 维 */
    std::vector<int> get_visit_counts() const;

    /** 获取根节点平均价值 Q */
    float get_root_value() const;

    /** 获取根节点原始策略（噪声施加前），225 维 */
    std::vector<float> get_root_policy() const;

    /** 设置根节点原始策略 */
    void set_root_policy(const std::vector<float>& policy);

    /** 获取子节点（用于 advance_root） */
    MctsNode* get_child(int action) const;

    /** 父节点指针 */
    MctsNode* parent() const { return parent_; }

    /** 获取 prior probability */
    double prior_prob() const { return p_sa_; }

    /** 设置 prior probability（用于狄利克雷噪声混合后更新） */
    void set_prior(double p) { p_sa_ = p; }

    /** 从父节点分离此节点（用于 advance_root）
     *
     * 从父节点的 children_ 映射中移除此节点，并将 parent_ 置为 nullptr。
     * 调用后此节点成为一棵独立树的根，不再随父节点一起析构。
     */
    void detach_from_parent();

private:
    MctsNode* parent_;

    /** 子节点映射：action → MctsNode*。
     *
     * children 中的节点为 unique_ptr 所有，此 map 仅持有原始指针用于快速查找。
     */
    std::unordered_map<int, MctsNode*> children_;

    // 子节点的实际所有者（仅在本节点被析构时递归释放）
    // 用 vector 存储，因为 unordered_map 的遍历顺序不确定，
    // 我们需要确保所有子节点都被删除
    // 实际上 children_ 中的指针就是从 children_owned_ 中获取的
    // 简化为：children_ 存指针，析构函数中遍历删除

    bool is_leaf_{true};

    std::mutex expand_lock_;    // 展开操作的互斥锁（防重复展开）
    std::mutex children_lock_;  // 子节点遍历时的互斥锁

    // --- 统计数据 ---
    double p_sa_;                           // 先验概率 P(s,a)
    double q_sa_{0.0};                      // 平均行动价值 Q(s,a) = W/N
    std::atomic<int> n_visited_{0};         // 已完成的访问次数
    std::atomic<int> n_unobserved_{0};      // 正在进行中的模拟次数
    std::atomic<int> virtual_loss_{0};      // Virtual Loss 计数器

    // 自身局面原始策略（噪声施加前 225 维）
    // 根节点必持有；展开节点经 expand_with_policy 在展开同时持有
    // （design/59 修复点 3），使任意节点被提升为新根后可返回正确策略。
    std::vector<float> root_policy_;

    /** 计算 PUCT 值 */
    double get_value(double c_puct, double c_virtual_loss,
                     unsigned int sum_n_visited) const;

    /** 递归删除所有子节点 */
    void delete_subtree();

    /** 在已持有 expand_lock_ 的上下文中创建全部子边（expand 族共用） */
    void do_expand_locked(const std::vector<float>& action_priors,
                          const std::vector<int>& legal_actions);
};
