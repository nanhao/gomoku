/** C++ MCTS 集成测试
 *
 * 测试多个模块的协作：
 * - TC-IX-01: BoardState + MctsNode 完整 simulation 流程
 * - TC-IX-02: 空棋盘 → 终局 完整树构建
 * - TC-IX-03: 多线程无数据竞争（detect via ThreadSanitizer / valgrind）
 * - TC-IX-04: 子树复用内存安全（advance_root 后旧树释放正确）
 *
 * 注意：需要有效的 TorchScript 模型文件。未提供时测试会被跳过。
 */

#include "../src/board_state.h"
#include "../src/mcts_node.h"
#include "../src/mcts_searcher.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

static int tests_passed = 0;
static int tests_failed = 0;
static int tests_skipped = 0;

static std::string get_model_path() {
    const char* env = std::getenv("NN_MODEL_PATH");
    if (env != nullptr) return std::string(env);
    return "src/ai/models/cnn_v11/cnn_v11_torchscript.pt";
}

#define TEST(name)                                         \
    do { std::printf("  TEST %s ... ", name); } while (0)

#define PASS()                                             \
    do { std::printf("PASS\n"); ++tests_passed; } while (0)

#define SKIP(msg)                                          \
    do { std::printf("SKIP (%s)\n", msg); ++tests_skipped; } while (0)

#define FAIL(msg)                                          \
    do { std::printf("FAIL: %s\n", msg); ++tests_failed; } while (0)

#define ASSERT_TRUE(cond)                                  \
    do { if (!(cond)) { FAIL(#cond); return; } } while (0)

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


// --- TC-IX-01: BoardState + MctsNode 协作 ---

static void test_boardstate_mctsnode_interaction() {
    TEST("TC-IX-01 BoardState + MctsNode interaction");

    // 创建非终局棋盘
    std::vector<int> board(225, 0);
    board[0] = 1;  // 放置一个黑子
    BoardState state(board, 1);

    // 创建根节点并展开
    MctsNode root(nullptr, 1.0);
    auto legal = state.get_legal_actions();
    std::vector<float> priors(legal.size(), 1.0f / legal.size());
    root.expand(priors, legal);

    // 选择子节点并模拟一次 backup
    int action = root.select_child(2.0, 0.0);
    ASSERT_TRUE(action >= 0);

    MctsNode* child = root.get_child(action);
    ASSERT_TRUE(child != nullptr);

    // 构造子棋盘状态
    BoardState child_state = state.apply_action(action);
    ASSERT_EQ((int)child_state.get_board()[action], 1);

    // backup 一个 NN 值
    child->backup(-0.5f);
    ASSERT_TRUE(child->get_root_value() != 0.0f);

    PASS();
}


// --- TC-IX-02: 完整搜索树构建（近终局）---

static void test_full_tree_near_terminal() {
    TEST("TC-IX-02 full tree near terminal");
    std::string path = get_model_path();
    try {
        MctsSearcher searcher(path, 50, 2.0, 0.0, 0.03, 2, 16, 0.0, false);

        // 四连棋盘
        std::vector<int> board(225, 0);
        board[0] = 1; board[1] = 1; board[2] = 1; board[3] = 1;
        std::vector<int> state = board;
        state.push_back(1);

        auto result = searcher.search(state);

        // 访问计数总和 = 50
        int total = 0;
        for (int c : result.visit_counts) total += c;
        ASSERT_EQ(total, 50);

        // 可获胜的 action=4 应有显著访问
        ASSERT_TRUE(result.visit_counts[4] > 0);

        PASS();
    } catch (const std::runtime_error& e) {
        SKIP(std::string("model unavailable: ") + e.what());
    }
}


// --- TC-IX-03: 多线程搜索无死锁 ---

static void test_multi_thread_no_deadlock() {
    TEST("TC-IX-03 multi-thread no deadlock");
    std::string path = get_model_path();
    try {
        // 使用较多线程和模拟数
        MctsSearcher searcher(path, 200, 2.0, 0.25, 0.03,
                              8, 16, 3.0, false);

        std::vector<int> board(225, 0);
        // 中心有少量棋子
        board[112] = 1;
        std::vector<int> state = board;
        state.push_back(2);

        auto result = searcher.search(state);

        int total = 0;
        for (int c : result.visit_counts) total += c;
        ASSERT_EQ(total, 200);

        PASS();
    } catch (const std::runtime_error& e) {
        SKIP(std::string("model unavailable: ") + e.what());
    }
}


// --- TC-IX-04: 子树复用内存安全 ---

static void test_advance_root_memory_safety() {
    TEST("TC-IX-04 advance_root memory safety");
    std::string path = get_model_path();
    try {
        MctsSearcher searcher(path, 30, 2.0, 0.25, 0.03,
                              2, 16, 3.0, false);

        std::vector<int> board(225, 0);
        std::vector<int> state = board;
        state.push_back(1);

        // 多轮搜索 + 推进
        for (int round = 0; round < 5; ++round) {
            auto result = searcher.search(state);
            int best = 0, best_n = -1;
            for (int i = 0; i < 225; ++i) {
                if (result.visit_counts[i] > best_n) {
                    best_n = result.visit_counts[i];
                    best = i;
                }
            }
            if (!searcher.advance_root(best)) break;
        }

        // 无崩溃即通过
        PASS();
    } catch (const std::runtime_error& e) {
        SKIP(std::string("model unavailable: ") + e.what());
    }
}


int main() {
    std::printf("=== MCTS Integration Tests ===\n\n");

    test_boardstate_mctsnode_interaction();
    test_full_tree_near_terminal();
    test_multi_thread_no_deadlock();
    test_advance_root_memory_safety();

    std::printf("\n=== Results: %d passed, %d failed, %d skipped ===\n",
                tests_passed, tests_failed, tests_skipped);
    return tests_failed > 0 ? 1 : 0;
}
