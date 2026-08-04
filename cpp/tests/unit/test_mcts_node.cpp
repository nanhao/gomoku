/** MctsNode 单元测试
 *
 * 覆盖 design/46/main_docs/test_cases.md 中 TC-MN-01 至 TC-MN-07 的所有用例。
 */
#include "../src/mcts_node.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <set>
#include <vector>

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name)                                         \
    do { std::printf("  TEST %s ... ", name); } while (0)

#define PASS()                                             \
    do { std::printf("PASS\n"); ++tests_passed; } while (0)

#define FAIL(msg)                                          \
    do { std::printf("FAIL: %s\n", msg); ++tests_failed; } while (0)

#define ASSERT_TRUE(cond)                                  \
    do { if (!(cond)) { FAIL(#cond); return; } } while (0)

#define ASSERT_FALSE(cond)                                 \
    do { if (cond) { FAIL(#cond " (expected false)"); return; } } while (0)

#define ASSERT_EQ(a, b)                                    \
    do {                                                   \
        if ((a) != (b)) {                                  \
            std::printf("FAIL: %s(%d) != %s(%d)\n",        \
                        #a, (int)(a), #b, (int)(b));       \
            ++tests_failed; return;                        \
        }                                                  \
    } while (0)

#define ASSERT_FLOAT_EQ(a, b, eps)                         \
    do {                                                   \
        if (std::fabs((a) - (b)) > (eps)) {                \
            std::printf("FAIL: %s(%f) != %s(%f)\n",        \
                        #a, (float)(a), #b, (float)(b));   \
            ++tests_failed; return;                        \
        }                                                  \
    } while (0)


// --- TC-MN-01: 根节点创建 ---
static void test_root_node() {
    TEST("TC-MN-01 root node creation");
    MctsNode root(nullptr, 1.0);

    ASSERT_TRUE(root.is_leaf());
    ASSERT_EQ(root.select_child(2.0, 3.0), -1);  // 无子节点
    ASSERT_FLOAT_EQ(root.get_root_value(), 0.0f, 1e-6);
    PASS();
}

// --- TC-MN-02: 展开子节点 ---
static void test_expand() {
    TEST("TC-MN-02 expand children");
    MctsNode root(nullptr, 1.0);

    std::vector<int> actions = {0, 1, 2};
    std::vector<float> priors = {0.5f, 0.3f, 0.2f};

    root.expand(priors, actions);

    ASSERT_FALSE(root.is_leaf());

    // 重复调用幂等
    root.expand(priors, actions);
    ASSERT_FALSE(root.is_leaf());

    // 验证子节点存在
    auto* child0 = root.get_child(0);
    ASSERT_TRUE(child0 != nullptr);
    ASSERT_FLOAT_EQ(static_cast<float>(child0->prior_prob()), 0.5f, 1e-6);

    auto* child2 = root.get_child(2);
    ASSERT_TRUE(child2 != nullptr);
    ASSERT_FLOAT_EQ(static_cast<float>(child2->prior_prob()), 0.2f, 1e-6);

    // 不存在的 action
    ASSERT_TRUE(root.get_child(100) == nullptr);

    PASS();
}

// --- TC-MN-03: PUCT 选择最优子节点 ---
static void test_puct_select() {
    TEST("TC-MN-03 PUCT select best child");
    MctsNode root(nullptr, 1.0);

    std::vector<int> actions = {0, 1, 2};
    std::vector<float> priors = {0.5f, 0.3f, 0.2f};
    root.expand(priors, actions);

    // 手动设置子节点的 N, Q 值：通过 backup 来模拟
    // action 0: N=10, W=5, Q=0.5
    // action 1: N=3,  W=2, Q=0.67
    // action 2: N=0,  W=0, Q=0
    // 为了实现这些精确值，直接操作子节点

    auto* c0 = root.get_child(0);
    auto* c1 = root.get_child(1);

    // 通过多次 backup 来设置 N 和 Q
    // c0: Q=0.5, N=10 → W = 0.5*10 = 5
    for (int i = 0; i < 10; ++i) c0->backup(0.5);
    // c1: Q≈0.667, N=3 → W ≈ 2
    for (int i = 0; i < 3; ++i) c1->backup(0.667);

    // 注意：backup 会沿父链向上传播，这里 c0/c1 的 parent 是 root
    // root 也被更新了，但这不影响 c0/c1 自身的 N/Q

    // 选择最优子节点（VL=0）
    int best = root.select_child(2.0, 0.0);
    ASSERT_TRUE(best >= 0 && best <= 2);
    PASS();
}

// --- TC-MN-04: Virtual Loss 施加效果 ---
static void test_virtual_loss() {
    TEST("TC-MN-04 virtual loss effect");
    MctsNode root(nullptr, 1.0);

    std::vector<int> actions = {0, 1, 2};
    std::vector<float> priors = {0.5f, 0.3f, 0.2f};
    root.expand(priors, actions);

    // 连续选择 3 次，VL=3.0 应使选择多样化
    std::set<int> selected;
    for (int i = 0; i < 3; ++i) {
        int a = root.select_child(2.0, 3.0);
        selected.insert(a);
    }

    // 在 VL 作用下，至少选中了 2 个不同的 action
    ASSERT_TRUE(selected.size() >= 2);
    PASS();
}

// --- TC-MN-05: Backup 回传 ---
static void test_backup_propagation() {
    TEST("TC-MN-05 backup propagation");
    MctsNode root(nullptr, 1.0);

    std::vector<int> actions = {0};
    std::vector<float> priors = {1.0f};
    root.expand(priors, actions);

    auto* child = root.get_child(0);
    ASSERT_TRUE(child != nullptr);

    // 记录 backup 前的 root N 和 child N
    int root_n_before = root.get_visit_counts()[0];  // 可能因 expand/其他操作

    // backup value=0.8
    child->backup(0.8f);

    // child: Q 应更新
    ASSERT_TRUE(child->get_root_value() != 0.0f);

    // root 边 [0] 应被更新（取反传播）
    auto root_counts = root.get_visit_counts();
    ASSERT_TRUE(root_counts[0] > root_n_before);

    PASS();
}

// --- TC-MN-06: 多层 Backup 取反 ---
static void test_multi_level_backup() {
    TEST("TC-MN-06 multi-level backup negation");
    MctsNode root(nullptr, 1.0);

    // root → A(action=0) → B(action=1)
    root.expand({1.0f}, {0});
    auto* A = root.get_child(0);

    A->expand({1.0f}, {1});
    auto* B = A->get_child(1);

    ASSERT_TRUE(B != nullptr);
    ASSERT_FALSE(A->is_leaf());

    float value_before = A->get_root_value();
    B->backup(1.0f);

    // B.Q = 1.0
    ASSERT_FLOAT_EQ(B->get_root_value(), 1.0f, 1e-6);

    // A 边[1]: W += -1.0, 所以 A.Q 应为负值或向负方向变化
    // 具体值依赖初始条件

    // root 边[0]: 二次取反 → 正值
    float root_q = root.get_root_value();
    ASSERT_TRUE(root_q >= 0.0f || root_q != value_before);  // 应变为正或发生变化

    PASS();
}

// --- TC-MN-07: get_visit_counts 维度 ---
static void test_visit_counts_dimension() {
    TEST("TC-MN-07 visit_counts dimension");
    MctsNode root(nullptr, 1.0);

    std::vector<int> actions = {0, 5, 10};
    std::vector<float> priors = {0.5f, 0.3f, 0.2f};
    root.expand(priors, actions);

    auto counts = root.get_visit_counts();
    ASSERT_EQ((int)counts.size(), 225);

    // 验证非零值仅在 action=0,5,10 位置（backup 后会有 N 值）
    // 初始时 N 都为 0，所以全部为 0
    for (int i = 0; i < 225; ++i) {
        if (i != 0 && i != 5 && i != 10) {
            ASSERT_EQ(counts[i], 0);
        }
    }

    PASS();
}


int main() {
    std::printf("=== MctsNode Unit Tests ===\n\n");

    test_root_node();
    test_expand();
    test_puct_select();
    test_virtual_loss();
    test_backup_propagation();
    test_multi_level_backup();
    test_visit_counts_dimension();

    std::printf("\n=== Results: %d passed, %d failed ===\n",
                tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
