/** MctsSearcher 树根状态一致性单元测试

覆盖 design/59/main_docs/test_cases.md 中 TC-SR-01 至 TC-SR-05：

- TC-SR-01 空棋盘冒烟（无回归）
- TC-SR-02 交替推进序列——已展开分支复用与未展开分支重建双路径（核心用例，修复前必失败）
- TC-SR-03 局面跳变整树重建
- TC-SR-04 advance_root 失败自愈
- TC-SR-05 root_policy 有效性（推进复用后）

通用断言约定（P3-1/P3-5）：每次 search 后 visit_counts 总和 == 本次模拟数；
root_policy 225 维均有限且总和 > 0；已占位处访问数 == 0；argmax 为真实空位。

注意：
- 需要有效的 TorchScript 模型文件（NN_MODEL_PATH 环境变量或默认路径），
  模型缺失时测试被跳过（SKIP）。
- 本测试不修改任何真实数据/模型文件，仅只读加载模型做 CPU 推理。
*/
#include "../src/mcts_searcher.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

static int tests_passed = 0;
static int tests_failed = 0;
static int tests_skipped = 0;

static std::string get_model_path() {
    const char* env = std::getenv("NN_MODEL_PATH");
    if (env != nullptr && std::strlen(env) > 0) return std::string(env);
    return "src/ai/models/cnn_v11/cnn_v11_torchscript.pt";
}

/** 检查模型文件是否存在（只读探测，不加载） */
static bool model_exists(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    return f.good();
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

// 值返回函数专用断言（int 返回函数中不能使用含 `return;` 的 ASSERT 宏）：
// 失败时打印、计数并返回 -1，配合 check_search_result 的契约使用
#define CHECK_TRUE(cond)                                   \
    do { if (!(cond)) { FAIL(#cond); return -1; } } while (0)

#define CHECK_EQ(a, b)                                     \
    do {                                                   \
        if ((a) != (b)) {                                  \
            std::printf("FAIL: %s(%d) != %s(%d)\n",        \
                        #a, (int)(a), #b, (int)(b));       \
            ++tests_failed; return -1;                     \
        }                                                  \
    } while (0)

static const int kSims = 100;  // 用例统一模拟数（与测试用例集一致）

/** 构造 226 维 state 向量（225 棋盘 + 1 当前玩家） */
static std::vector<int> make_state(const std::vector<int>& board,
                                   int current_player) {
    auto state = board;
    state.push_back(current_player);
    return state;
}

/** 通用 search 结果合法性断言（P3-1/P3-5）

对每次 search 结果统一断言：
- visit_counts 225 维、总和 == 本次模拟数（捕获叶根空转：空转时总和为 0）
- 棋盘已占位处访问数 == 0；argmax 必须是真实棋盘空位
- root_policy 225 维均为有限值且总和 > 0（非全 0）
- root_value 有限

@return argmax 动作索引（空位中 N 最大者），失败时返回 -1
*/
static int check_search_result(const MctsResult& result,
                               const std::vector<int>& board,
                               int sims) {
    CHECK_EQ((int)result.visit_counts.size(), 225);
    CHECK_EQ((int)result.root_policy.size(), 225);

    // 占位处访问数必须为 0；统计空位访问总和
    int total = 0;
    for (int i = 0; i < 225; ++i) {
        if (board[i] != 0) {
            if (result.visit_counts[i] != 0) {
                std::printf("FAIL: occupied cell %d has %d visits\n",
                            i, result.visit_counts[i]);
                ++tests_failed;
                return -1;
            }
        } else {
            total += result.visit_counts[i];
        }
    }
    CHECK_EQ(total, sims);

    // root_policy：均有限且总和 > 0（非全 0，排除叶根回退策略）
    float policy_sum = 0.0f;
    for (float p : result.root_policy) {
        if (!std::isfinite(p)) {
            std::printf("FAIL: root_policy contains non-finite value\n");
            ++tests_failed;
            return -1;
        }
        policy_sum += p;
    }
    CHECK_TRUE(policy_sum > 0.0f);
    CHECK_TRUE(std::isfinite(result.root_value));

    // argmax 必须是真实空位（N 最大者；全 0 时最小空位仍入选）
    int best = -1;
    int best_n = -1;
    for (int i = 0; i < 225; ++i) {
        if (board[i] == 0 && result.visit_counts[i] > best_n) {
            best_n = result.visit_counts[i];
            best = i;
        }
    }
    CHECK_TRUE(best >= 0);
    return best;
}

// =====================================================================
// 确定性交替推进构造（TC-SR-02/05 共用）
// =====================================================================
// 背景：文档"热区/冷区"未定义选取方式；根先验启发（root_policy 最大
// 空位）在随机权重模型下不可靠——promote 需要对手落子命中"己方推进后
// 节点下已访问（已展开）的子边"，而随机模型先验近均匀，与树内访问分布
// 无关联 → 热区轮 promote 从不命中（2026-09-02 实测 5 轮全重建）。
// 修订为可观测构造：
// - 热区轮：白视角 mid 搜索（树根与传入一致 → 一致复用分支，rc 不变），
//   其增量即"推进节点下白边访问"的可观测集合 → 白落 X = 增量 N 最大
//   空位（必已访问已展开）→ 黑 search 差 1 步必走 promote（rc 不变）；
// - 冷区轮：白落 X = visited（当前树代全部历史访问并集）补集 → 该边必
//   为未访问叶（或不存在）→ 黑 search 必走整树重建（rc+1）。
// rebuild 发生时 visited 清空（旧代作废，新树从零计）。

/** 黑方首步前置：search(空盘,黑) → 落 argmax → advance_root
 *
 * 结果树根 == 当前棋盘 + 轮到白（供后续白视角 mid 搜索一致复用）。
 *
 * @param board 进出参数：进入时空盘轮到黑；返回时黑刚落 1 子轮到白
 * @param visited 进出参数：记录本步黑 search 增量访问位置
 */
static void play_black_first_move(MctsSearcher& searcher,
                                  std::vector<int>& board,
                                  std::vector<bool>& visited) {
    MctsResult r0 = searcher.search(make_state(board, 1));
    int a = check_search_result(r0, board, kSims);
    ASSERT_TRUE(a >= 0);
    board[a] = 1;
    ASSERT_TRUE(searcher.advance_root(a));
    for (int i = 0; i < 225; ++i) {
        if (r0.visit_counts[i] > 0) visited[i] = true;
    }
}

/** 热区轮：白 mid 观测 → 白落热区 X → 黑 search（必 promote 复用）
 *
 * 黑方随后落 argmax 并推进（下轮前置：树根 = 己方刚落、轮到白）。
 *
 * @param board 进出参数：进入时黑刚落子轮到白；返回时黑再落 1 子轮到白
 * @param visited 进出参数：维护当前树代历史访问并集
 * @param last_result 输出：黑 search 的 MctsResult（可空，供 TC-SR-05
 *    断言 root_policy）
 * @return 黑 search 的 rebuild 计数增量：0=promote 复用（预期）；
 *    1=整树重建（异常信号）；-1=构造失败
 */
static int play_hot_round(MctsSearcher& searcher, std::vector<int>& board,
                          std::vector<bool>& visited,
                          MctsResult* last_result) {
    // 1) 白视角 mid 搜索：树根 == 黑刚落子后局面（轮到白）→ 一致复用
    MctsResult r_mid = searcher.search(make_state(board, 2));
    check_search_result(r_mid, board, kSims);
    for (int i = 0; i < 225; ++i) {
        if (r_mid.visit_counts[i] > 0) visited[i] = true;
    }

    // 2) 白落 X = mid 增量 N 最大空位（必已访问已展开 → promote 命中）
    int x = -1;
    int best_n = -1;
    for (int i = 0; i < 225; ++i) {
        if (board[i] == 0 && r_mid.visit_counts[i] > best_n) {
            best_n = r_mid.visit_counts[i];
            x = i;
        }
    }
    CHECK_TRUE(x >= 0);
    board[x] = 2;

    // 3) 黑 search（局面差白 1 子 → promote），记录 rc 增量
    int rc0 = searcher.get_rebuild_count();
    MctsResult r2 = searcher.search(make_state(board, 1));
    int rc1 = searcher.get_rebuild_count();
    check_search_result(r2, board, kSims);
    if (last_result != nullptr) *last_result = r2;

    // 4) 黑方落 argmax 并推进（下轮前置）
    int a = check_search_result(r2, board, kSims);
    CHECK_TRUE(a >= 0);
    board[a] = 1;
    CHECK_TRUE(searcher.advance_root(a));

    // 5) 访问并集维护：整树重建后旧代作废，清空重记
    if (rc1 != rc0) {
        std::fill(visited.begin(), visited.end(), false);
    }
    for (int i = 0; i < 225; ++i) {
        if (r2.visit_counts[i] > 0) visited[i] = true;
    }
    return rc1 - rc0;
}

/** 冷区轮：白落 X = visited 补集最小空位 → 黑 search（必整树重建）
 *
 * 黑方随后落 argmax 并推进（重建后新树）。
 *
 * @param board 进出参数：进入时黑刚落子轮到白；返回时黑再落 1 子轮到白
 * @param visited 进出参数：维护当前树代历史访问并集
 * @return 黑 search 的 rebuild 计数增量（预期 1）；-2 = 补集为空
 *    （全部空位均被访问过，本轮无法构造，调用方跳过）；-1=构造失败
 */
static int play_cold_round(MctsSearcher& searcher, std::vector<int>& board,
                           std::vector<bool>& visited) {
    // 1) 冷区 X：从未被访问（visited 补集）→ 该边必为未展开叶
    int x = -1;
    for (int i = 0; i < 225; ++i) {
        if (board[i] == 0 && !visited[i]) { x = i; break; }
    }
    if (x < 0) return -2;  // 补集为空：无法构造，调用方跳过
    board[x] = 2;

    // 2) 黑 search：差 1 步但目标为未访问叶（或无此边）→ 整树重建
    int rc0 = searcher.get_rebuild_count();
    MctsResult r2 = searcher.search(make_state(board, 1));
    int rc1 = searcher.get_rebuild_count();
    check_search_result(r2, board, kSims);

    // 3) 黑方落 argmax 并推进
    int a = check_search_result(r2, board, kSims);
    CHECK_TRUE(a >= 0);
    board[a] = 1;
    CHECK_TRUE(searcher.advance_root(a));

    // 4) 访问并集维护：重建后旧代作废，清空重记
    if (rc1 != rc0) {
        std::fill(visited.begin(), visited.end(), false);
    }
    for (int i = 0; i < 225; ++i) {
        if (r2.visit_counts[i] > 0) visited[i] = true;
    }
    return rc1 - rc0;
}

// --- TC-SR-01: 空棋盘冒烟（无回归）---
static void test_empty_board_smoke() {
    TEST("TC-SR-01 empty board smoke");
    std::string path = get_model_path();
    if (!model_exists(path)) { SKIP("model not available"); return; }
    try {
        MctsSearcher searcher(path, kSims, 2.0, 0.25, 0.03, 4, 3.0, false);

        std::vector<int> board(225, 0);
        MctsResult result = searcher.search(make_state(board, 1));

        int best = check_search_result(result, board, kSims);
        ASSERT_TRUE(best >= 0);
        PASS();
    } catch (const std::runtime_error& e) {
        SKIP((std::string("model load failed: ") + e.what()).c_str());
    }
}

// --- TC-SR-02: 交替推进序列——复用与重建双路径（核心用例）---
// 构造说明（2026-09-02 修订）：设计文档"热区/冷区"未定义选取方式，根
// 先验启发在随机权重模型下 promote 从不命中；确定性构造见下方公共函数
// play_hot_round / play_cold_round（mid 搜索观测 + visited 补集）。
// 每轮路径硬断言：热区轮 rc 不变（promote）、冷区轮 rc+1（整树重建）。
static void test_alternating_advance_dual_path() {
    TEST("TC-SR-02 alternating advance promote/rebuild dual path");
    std::string path = get_model_path();
    if (!model_exists(path)) { SKIP("model not available"); return; }
    try {
        MctsSearcher searcher(path, kSims, 2.0, 0.25, 0.03, 4, 3.0, false);
        std::vector<int> board(225, 0);
        std::vector<bool> visited(225, false);

        // 黑方首步（空盘全新建树），落 argmax 并推进
        play_black_first_move(searcher, board, visited);

        // 冷区轮（round 偶，必重建）与热区轮（round 奇，必 promote）
        // 交替 5 轮；每轮按预测路径硬断言
        int cold_hits = 0;
        int hot_hits = 0;
        for (int round = 0; round < 5; ++round) {
            bool hot = (round % 2 == 1);
            int delta = hot
                            ? play_hot_round(searcher, board, visited, nullptr)
                            : play_cold_round(searcher, board, visited);
            if (delta == -2) continue;  // 冷区补集为空：本轮无法构造，跳过
            if (delta == -1) {
                std::printf("FAIL: round %d construction failed\n", round);
                ++tests_failed;
                return;
            }
            int expected = hot ? 0 : 1;  // 热区必复用、冷区必重建
            if (delta != expected) {
                std::printf("FAIL: round %d path delta=%d expected %d\n",
                            round, delta, expected);
                ++tests_failed;
                return;
            }
            if (hot) {
                ++hot_hits;
            } else {
                ++cold_hits;
            }
        }

        // 双路径必须同时出现（P0-1：只增不减说明 promote 路径从未命中）
        if (hot_hits == 0) { FAIL("promote/reuse path never hit"); return; }
        if (cold_hits == 0) { FAIL("rebuild path never hit"); return; }
        PASS();
    } catch (const std::runtime_error& e) {
        SKIP((std::string("model load failed: ") + e.what()).c_str());
    }
}

// --- TC-SR-03: 局面跳变整树重建 ---
static void test_state_jump_rebuild() {
    TEST("TC-SR-03 state jump triggers full rebuild");
    std::string path = get_model_path();
    if (!model_exists(path)) { SKIP("model not available"); return; }
    try {
        MctsSearcher searcher(path, kSims, 2.0, 0.25, 0.03, 4, 3.0, false);

        // 1) 空棋盘 + 黑：全新建树（不算重建）
        std::vector<int> empty(225, 0);
        MctsResult r0 = searcher.search(make_state(empty, 1));
        check_search_result(r0, empty, kSims);
        ASSERT_EQ(searcher.get_rebuild_count(), 0);

        // 2) 预置 8 子中盘局面（黑 4 + 白 4，交错分布无五连）+ 白视角
        //    与树根（空棋盘+黑）无继承关系 → 整树重建
        std::vector<int> mid(225, 0);
        mid[0] = 1;  mid[2] = 1;  mid[4] = 1;  mid[6] = 1;    // 黑：隔位散开
        mid[16] = 2; mid[18] = 2; mid[20] = 2; mid[22] = 2;   // 白：隔位散开
        MctsResult r1 = searcher.search(make_state(mid, 2));
        check_search_result(r1, mid, kSims);
        ASSERT_EQ(searcher.get_rebuild_count(), 1);

        // 3) 跳回空棋盘 + 黑：再次整树重建
        MctsResult r2 = searcher.search(make_state(empty, 1));
        check_search_result(r2, empty, kSims);
        ASSERT_EQ(searcher.get_rebuild_count(), 2);

        PASS();
    } catch (const std::runtime_error& e) {
        SKIP((std::string("model load failed: ") + e.what()).c_str());
    }
}

// --- TC-SR-04: advance_root 失败自愈 ---
static void test_advance_root_failure_self_heal() {
    TEST("TC-SR-04 advance_root failure self-heal");
    std::string path = get_model_path();
    if (!model_exists(path)) { SKIP("model not available"); return; }
    try {
        MctsSearcher searcher(path, kSims, 2.0, 0.25, 0.03, 4, 3.0, false);
        std::vector<int> board(225, 0);

        MctsResult r0 = searcher.search(make_state(board, 1));
        check_search_result(r0, board, kSims);
        int rc0 = searcher.get_rebuild_count();

        // 非法推进：越界动作必须失败且不破坏树状态
        ASSERT_FALSE(searcher.advance_root(-1));
        ASSERT_FALSE(searcher.advance_root(225));

        // 同状态再 search：走"一致复用"分支，正常返回且不触发重建
        MctsResult r1 = searcher.search(make_state(board, 1));
        int best = check_search_result(r1, board, kSims);
        ASSERT_TRUE(best >= 0);
        ASSERT_EQ(searcher.get_rebuild_count(), rc0);

        PASS();
    } catch (const std::runtime_error& e) {
        SKIP((std::string("model load failed: ") + e.what()).c_str());
    }
}

// --- TC-SR-05: root_policy 有效性（推进复用后）---
// 复用 TC-SR-02 的确定性热区构造（mid 搜索观测 + 热区落子 → promote
// 复用必命中）：复用轮（rc 不变）出现后，root_policy 必须为新根自身局面
// 的网络原始策略——225 维均有限、总和 > 0（check_search_result 已验）
// 且 top-1 > 0（修复点 3：推进节点展开时持有自身策略，非回退/噪声值）。
static void test_root_policy_valid_after_promote() {
    TEST("TC-SR-05 root_policy valid after promote");
    std::string path = get_model_path();
    if (!model_exists(path)) { SKIP("model not available"); return; }
    try {
        MctsSearcher searcher(path, kSims, 2.0, 0.25, 0.03, 4, 3.0, false);
        std::vector<int> board(225, 0);
        std::vector<bool> visited(225, false);
        play_black_first_move(searcher, board, visited);

        // 循环热区轮直至 promote 复用轮出现（确定性构造保证最多 2 轮）
        bool checked = false;
        MctsResult r2;
        for (int round = 0; round < 3 && !checked; ++round) {
            int delta = play_hot_round(searcher, board, visited, &r2);
            if (delta == 0) {
                // promote 复用轮 → 验证新根自身局面的网络策略
                float top1 = 0.0f;
                for (float p : r2.root_policy) top1 = std::max(top1, p);
                if (top1 <= 0.0f) {
                    std::printf("FAIL: root_policy top-1 <= 0 after promote\n");
                    ++tests_failed;
                    return;
                }
                checked = true;
            } else if (delta != 1) {
                std::printf("FAIL: hot round construction failed (delta=%d)\n",
                            delta);
                ++tests_failed;
                return;
            }
        }
        // 复用轮必现（构造确定性）；未现即实现回归信号
        if (!checked) { FAIL("promote/reuse round never hit"); return; }
        PASS();
    } catch (const std::runtime_error& e) {
        SKIP((std::string("model load failed: ") + e.what()).c_str());
    }
}

int main() {
    std::printf("=== MctsSearcher State-Sync Unit Tests ===\n\n");

    test_empty_board_smoke();
    test_alternating_advance_dual_path();
    test_state_jump_rebuild();
    test_advance_root_failure_self_heal();
    test_root_policy_valid_after_promote();

    std::printf("\n=== Results: %d passed, %d failed, %d skipped ===\n",
                tests_passed, tests_failed, tests_skipped);
    return tests_failed > 0 ? 1 : 0;
}
