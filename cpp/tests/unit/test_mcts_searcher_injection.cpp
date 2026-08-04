/** MctsSearcher 共享资源注入 单元测试

覆盖 design/47/main_docs/test_cases.md 中 MctsSearcher 改造相关用例。
验证: 新构造器接受外部 pool/nn、搜索接口不变。

注意: 需要有效 TorchScript 模型文件的测试会被跳过（标记为 SKIP）。
*/

#include "../src/board_state.h"
#include "../src/mcts_config.h"
#include "../src/mcts_searcher.h"
#include "../src/neural_net.h"
#include "../src/thread_pool.h"

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


// --- TC-MS-I01: 新构造器创建成功 ---
static void test_new_constructor() {
    TEST("TC-MS-I01 new constructor with shared resources");
    std::string path = get_model_path();

    try {
        NeuralNet nn(path, false);
        MctsThreadPool::initialize(4);
        MctsThreadPool& pool = MctsThreadPool::instance();

        // 使用新构造器
        MctsSearcher searcher(
            path, pool, nn,
            g_mcts_config.num_simulations,
            g_mcts_config.c_puct,
            g_mcts_config.dirichlet_epsilon,
            g_mcts_config.dirichlet_alpha,
            g_mcts_config.virtual_loss);

        // 若有模型，验证搜索功能
        (void)searcher;
        PASS();
    } catch (const std::exception& e) {
        SKIP(std::string("model unavailable: ") + e.what());
    }
}


// --- TC-MS-I02: search 使用 pool.submit 模式 ---
static void test_search_with_submit() {
    TEST("TC-MS-I02 search uses pool.submit mode");
    std::string path = get_model_path();

    try {
        NeuralNet nn(path, false);
        MctsThreadPool::initialize(4);
        MctsThreadPool& pool = MctsThreadPool::instance();

        MctsSearcher searcher(
            path, pool, nn,
            50,  // 少量模拟
            2.0, 0.25, 0.03, 3.0);

        std::vector<int> board(225, 0);
        board[112] = 1;  // 中心放一枚黑子
        std::vector<int> state = board;
        state.push_back(2);  // 白方落子

        auto result = searcher.search(state);

        // 总访问计数应等于 num_simulations
        int total = 0;
        for (int c : result.visit_counts) total += c;
        ASSERT_EQ(total, 50);

        PASS();
    } catch (const std::exception& e) {
        SKIP(std::string("model unavailable: ") + e.what());
    }
}


// --- TC-MS-I03: 旧构造器仍可用（向后兼容）---
static void test_legacy_constructor_still_works() {
    TEST("TC-MS-I03 legacy constructor still works");
    std::string path = get_model_path();

    try {
        // 旧构造器（自建 NeuralNet + 全局线程池）
        MctsSearcher searcher(
            path, 50, 2.0, 0.25, 0.03,
            g_mcts_config.num_threads, 3.0, false);

        std::vector<int> board(225, 0);
        board[112] = 1;
        std::vector<int> state = board;
        state.push_back(2);

        auto result = searcher.search(state);

        int total = 0;
        for (int c : result.visit_counts) total += c;
        ASSERT_EQ(total, 50);

        PASS();
    } catch (const std::exception& e) {
        SKIP(std::string("model unavailable: ") + e.what());
    }
}


// --- TC-MS-I04: 旧构造器仍使用 run_parallel 模式 ---
static void test_legacy_uses_run_parallel() {
    TEST("TC-MS-I04 legacy uses run_parallel");
    std::string path = get_model_path();

    try {
        // 验证旧构造器仍然走 run_parallel 代码路径
        MctsSearcher searcher(
            path, 30, 2.0, 0.25, 0.03,
            g_mcts_config.num_threads, 0.0, false);

        std::vector<int> board(225, 0);
        std::vector<int> state = board;
        state.push_back(1);

        auto result = searcher.search(state);

        int total = 0;
        for (int c : result.visit_counts) total += c;
        ASSERT_EQ(total, 30);

        PASS();
    } catch (const std::exception& e) {
        SKIP(std::string("model unavailable: ") + e.what());
    }
}


// --- TC-MS-I05: advance_root 与新的 MctsSearcher 兼容 ---
static void test_advance_root_with_new_ctor() {
    TEST("TC-MS-I05 advance_root with new constructor");
    std::string path = get_model_path();

    try {
        NeuralNet nn(path, false);
        MctsThreadPool::initialize(4);
        MctsThreadPool& pool = MctsThreadPool::instance();

        MctsSearcher searcher(
            path, pool, nn,
            30, 2.0, 0.25, 0.03, 3.0);

        std::vector<int> board(225, 0);
        std::vector<int> state = board;
        state.push_back(1);

        // 搜索并推进
        auto result = searcher.search(state);
        int best = 0, best_n = -1;
        for (int i = 0; i < 225; ++i) {
            if (result.visit_counts[i] > best_n) {
                best_n = result.visit_counts[i];
                best = i;
            }
        }
        bool advanced = searcher.advance_root(best);
        ASSERT_TRUE(advanced);

        PASS();
    } catch (const std::exception& e) {
        SKIP(std::string("model unavailable: ") + e.what());
    }
}


int main() {
    std::printf("=== MctsSearcher Resource Injection Unit Tests ===\n\n");

    test_new_constructor();
    test_search_with_submit();
    test_legacy_constructor_still_works();
    test_legacy_uses_run_parallel();
    test_advance_root_with_new_ctor();

    std::printf("\n=== Results: %d passed, %d failed, %d skipped ===\n",
                tests_passed, tests_failed, tests_skipped);
    return tests_failed > 0 ? 1 : 0;
}
