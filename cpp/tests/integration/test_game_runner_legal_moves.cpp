/** GameRunner 落子合法性集成测试

覆盖 design/59/main_docs/test_cases.md 中 TC-GR-01 / TC-GR-02：

- TC-GR-01 自对弈样本棋盘子数严格递增（核心用例，修复前必失败）
- TC-GR-02 双模型对战回归冒烟（含 rebuild 诊断输出观测）

注意：
- 需要有效的 TorchScript 模型文件（NN_MODEL_PATH 环境变量或默认路径），
  模型缺失时测试被跳过（SKIP）。
- 本测试不修改任何真实数据/模型文件，仅只读加载模型做 CPU 推理。
- GameRunner 内部管理全局单例资源（线程池 + NeuralNet），析构时自动关闭，
  故同一进程内顺序创建多个 GameRunner 是安全的。
*/
#include "../src/game_runner.h"
#include "../src/game_sample.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
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

#define ASSERT_EQ(a, b)                                    \
    do {                                                   \
        if ((a) != (b)) {                                  \
            std::printf("FAIL: %s(%d) != %s(%d)\n",        \
                        #a, (int)(a), #b, (int)(b));       \
            ++tests_failed; return;                        \
        }                                                  \
    } while (0)

/** stdout 捕获器（RAII）：构造后替换 std::cout 缓冲，析构时恢复 */
struct StdoutCapture {
    std::ostringstream buf_;
    std::streambuf* old_buf_;

    StdoutCapture() : old_buf_(std::cout.rdbuf(buf_.rdbuf())) {}
    ~StdoutCapture() { std::cout.rdbuf(old_buf_); }

    /** 已捕获的全部文本 */
    std::string text() const { return buf_.str(); }
};

// --- TC-GR-01: 自对弈样本棋盘子数严格递增 ---
static void test_selfplay_sample_stone_count() {
    TEST("TC-GR-01 selfplay sample stone count strictly increasing");
    std::string path = get_model_path();
    if (!model_exists(path)) { SKIP("model not available"); return; }

    try {
        // 单模型自对弈 1 局：sims=200, temp_steps=2, CPU, 采集样本
        GameConfig config;
        config.num_simulations = 200;
        config.temp_steps = 2;
        config.use_gpu = false;

        GameRunner runner(path, 1, config, 1, true);
        runner.start();
        runner.wait_completion();

        std::vector<GameResult> results = runner.get_results();
        ASSERT_EQ((int)results.size(), 1);

        const GameResult& res = results[0];
        ASSERT_TRUE(res.winner == 0 || res.winner == 1 || res.winner == 2);
        ASSERT_TRUE(res.move_count >= 1 && res.move_count <= 225);

        // 非平局时必有样本（平局时无样本生成，不判非空）
        if (res.winner != 0) {
            ASSERT_TRUE(!res.samples.empty());

            // D4 增强分组断言：samples[8k .. 8k+7] 每组 8 条
            // state 前 225 维棋子总数均 == k（第 k 条原始样本是第 k 步
            // 落子前的局面，应含 k 子；覆盖落子会使某步后棋子数停滞 → 失败）
            ASSERT_EQ((int)res.samples.size(), res.move_count * 8);
            for (int k = 0; k < res.move_count; ++k) {
                for (int t = 0; t < 8; ++t) {
                    const GameSample& s = res.samples[k * 8 + t];
                    ASSERT_EQ((int)s.state.size(), 226);
                    int stones = 0;
                    for (int i = 0; i < 225; ++i) {
                        if (s.state[i] != 0.0f) ++stones;
                    }
                    if (stones != k) {
                        std::printf(
                            "FAIL: sample[%d] (variant %d) has %d stones, "
                            "expected %d\n", k, t, stones, k);
                        ++tests_failed;
                        return;
                    }
                }
            }
        }
        PASS();
    } catch (const std::runtime_error& e) {
        SKIP((std::string("model load failed: ") + e.what()).c_str());
    }
}

// --- TC-GR-02: 双模型对战回归冒烟（含 rebuild 诊断输出观测）---
static void test_dual_model_battle_smoke() {
    TEST("TC-GR-02 dual-model battle regression smoke");
    std::string path = get_model_path();
    if (!model_exists(path)) { SKIP("model not available"); return; }

    try {
        // 同模型对战 1 局（path1 == path2 允许），捕获 stdout 观测诊断行
        StdoutCapture capture;

        GameConfig config;
        config.num_simulations = 100;
        config.temp_steps = 0;
        config.use_gpu = false;

        GameRunner runner(path, path, 1, config, 1);
        runner.start();
        runner.wait_completion();

        std::vector<GameResult> results = runner.get_results();
        ASSERT_EQ((int)results.size(), 1);
        ASSERT_TRUE(results[0].winner == 0 ||
                    results[0].winner == 1 ||
                    results[0].winner == 2);
        ASSERT_TRUE(results[0].move_count >= 1 &&
                    results[0].move_count <= 225);

        // 每局结束应打印 rebuild 诊断行（修复点可观测性）
        std::string out = capture.text();
        bool has_rebuild_line =
            out.find("[GameRunner] game=0 rebuild:") != std::string::npos;
        if (!has_rebuild_line) {
            std::printf("FAIL: stdout missing rebuild diagnostic line\n");
            ++tests_failed;
            return;
        }
        PASS();
    } catch (const std::runtime_error& e) {
        SKIP((std::string("model load failed: ") + e.what()).c_str());
    }
}

int main() {
    std::printf("=== GameRunner Legal-Moves Integration Tests ===\n\n");

    test_selfplay_sample_stone_count();
    test_dual_model_battle_smoke();

    std::printf("\n=== Results: %d passed, %d failed, %d skipped ===\n",
                tests_passed, tests_failed, tests_skipped);
    return tests_failed > 0 ? 1 : 0;
}
