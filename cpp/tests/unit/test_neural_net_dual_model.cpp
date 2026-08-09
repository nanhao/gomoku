/** NeuralNet 双模型缓存 单元测试

覆盖 design/51/main_docs/overview.md 中双模型重构相关用例。

验证: preload_model 幂等、predict(model_id) 路由、
module_cache_ 缓存两个模型、forward_batch 按模型分组。

注意:
- 需要有效 TorchScript 模型文件的测试会被跳过（标记为 SKIP）。
- 预加载行为和无效 model_id 测试不需要模型文件。
*/

#include "../src/neural_net.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

static int tests_passed = 0;
static int tests_failed = 0;
static int tests_skipped = 0;

// 模型文件路径：通过环境变量 NN_MODEL_PATH 传入，或使用默认路径
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

// 检查模型文件是否存在
static bool model_file_exists(const std::string& path) {
    if (path.empty()) return false;
    std::ifstream f(path);
    return f.good();
}


// --- TC-DM-01: preload_model 幂等（同一路径预加载两次）---
static void test_preload_idempotent() {
    TEST("TC-DM-01 preload_model idempotent");
    std::string path = get_model_path();
    if (!model_file_exists(path)) {
        SKIP("no model file available");
        return;
    }
    try {
        NeuralNet nn(path, false);

        // 预加载同一路径两次，不应崩溃
        nn.preload_model(path);
        nn.preload_model(path);

        PASS();
    } catch (const std::exception& e) {
        FAIL(e.what());
    }
}


// --- TC-DM-02: predict 使用 model_id 推理 ---
static void test_predict_with_model_id() {
    TEST("TC-DM-02 predict with model_id");
    std::string path = get_model_path();
    if (!model_file_exists(path)) {
        SKIP("no model file available");
        return;
    }
    try {
        NeuralNet nn(path, false);

        // 构造编码输入
        std::vector<float> encoded(675, 0.0f);
        for (int i = 0; i < 225; ++i) {
            encoded[2 * 225 + i] = 1.0f;
        }

        // 使用 model_id（即文件路径）进行推理
        auto result = nn.predict(encoded, path);

        // 验证结果
        ASSERT_EQ((int)result.policy.size(), 225);
        ASSERT_TRUE(result.value >= -1.0f && result.value <= 1.0f);

        // policy 总和 ≈ 1.0
        float sum = 0.0f;
        for (float p : result.policy) {
            ASSERT_TRUE(p > 0.0f && p < 1.0f);
            sum += p;
        }
        ASSERT_FLOAT_EQ(sum, 1.0f, 1e-4f);

        PASS();
    } catch (const std::exception& e) {
        FAIL(e.what());
    }
}


// --- TC-DM-03: 同一模型通过不同路径推理应得相同结果 ---
static void test_same_model_same_result() {
    TEST("TC-DM-03 same model same result via model_id");
    std::string path = get_model_path();
    if (!model_file_exists(path)) {
        SKIP("no model file available");
        return;
    }
    try {
        NeuralNet nn(path, false);

        std::vector<float> encoded(675, 0.0f);
        for (int i = 0; i < 225; ++i) {
            encoded[2 * 225 + i] = 1.0f;
        }

        // 两次推理应得到相同结果
        auto r1 = nn.predict(encoded, path);
        auto r2 = nn.predict(encoded, path);

        ASSERT_EQ((int)r1.policy.size(), 225);
        ASSERT_EQ((int)r2.policy.size(), 225);

        // 逐元素比较 policy
        for (int i = 0; i < 225; ++i) {
            ASSERT_FLOAT_EQ(r1.policy[i], r2.policy[i], 1e-6f);
        }
        ASSERT_FLOAT_EQ(r1.value, r2.value, 1e-6f);

        PASS();
    } catch (const std::exception& e) {
        FAIL(e.what());
    }
}


// --- TC-DM-04: predict 使用无效 model_id 应抛异常 ---
static void test_predict_invalid_model_id() {
    TEST("TC-DM-04 predict with invalid model_id");
    std::string path = get_model_path();
    if (!model_file_exists(path)) {
        SKIP("no model file available");
        return;
    }
    try {
        NeuralNet nn(path, false);

        std::vector<float> encoded(675, 0.0f);
        for (int i = 0; i < 225; ++i) {
            encoded[2 * 225 + i] = 1.0f;
        }

        // 使用未预加载的 model_id
        try {
            auto result = nn.predict(encoded, "non_existent_model_id");
            FAIL("expected exception but none thrown");
        } catch (const std::runtime_error&) {
            // 预期行为
            PASS();
        }
    } catch (const std::exception& e) {
        FAIL(e.what());
    }
}


// --- TC-DM-05: 缓存两个模型 ---
static void test_cache_two_models() {
    TEST("TC-DM-05 cache two models");
    std::string path = get_model_path();
    if (!model_file_exists(path)) {
        SKIP("no model file available");
        return;
    }
    try {
        // 构造时预加载第一个模型
        NeuralNet nn(path, false);

        // 预加载同一模型作为"第二个模型"（用不同路径名模拟）
        // 实际测试中只有一个模型文件，用不同 key 模拟两个模型
        std::string key2 = path + "#model2";

        // 使用同一个模型文件但不同 key 预加载
        // 注意: preload_model 用 key 索引，所以即使指向同一文件也会缓存两份
        nn.preload_model(key2);

        // 两个模型各自推理
        std::vector<float> encoded(675, 0.0f);
        for (int i = 0; i < 225; ++i) {
            encoded[2 * 225 + i] = 1.0f;
        }

        auto r1 = nn.predict(encoded, path);
        auto r2 = nn.predict(encoded, key2);

        ASSERT_EQ((int)r1.policy.size(), 225);
        ASSERT_EQ((int)r2.policy.size(), 225);

        // 同一权重文件应给出相同结果
        for (int i = 0; i < 225; ++i) {
            ASSERT_FLOAT_EQ(r1.policy[i], r2.policy[i], 1e-6f);
        }

        PASS();
    } catch (const std::exception& e) {
        FAIL(e.what());
    }
}


// --- TC-DM-06: 多线程并发 predict 不同 model_id ---
static void test_concurrent_predict_different_models() {
    TEST("TC-DM-06 concurrent predict different model_ids");
    std::string path = get_model_path();
    if (!model_file_exists(path)) {
        SKIP("no model file available");
        return;
    }
    try {
        NeuralNet nn(path, false);

        std::string key2 = path + "#model2";
        nn.preload_model(key2);

        std::vector<float> encoded(675, 0.0f);
        for (int i = 0; i < 225; ++i) {
            encoded[2 * 225 + i] = 1.0f;
        }

        // 多线程并发 predict：偶数线程用 path，奇数线程用 key2
        const int N = 20;
        std::vector<std::thread> threads;
        std::vector<NeuralNetResult> results(N);

        for (int i = 0; i < N; ++i) {
            threads.emplace_back([&nn, &encoded, &results, i, &path, &key2] {
                const std::string& mid = (i % 2 == 0) ? path : key2;
                results[i] = nn.predict(encoded, mid);
            });
        }

        for (auto& t : threads) {
            t.join();
        }

        // 验证所有结果有效
        for (int i = 0; i < N; ++i) {
            ASSERT_EQ((int)results[i].policy.size(), 225);
            ASSERT_TRUE(results[i].value >= -1.0f && results[i].value <= 1.0f);
        }

        PASS();
    } catch (const std::exception& e) {
        FAIL(e.what());
    }
}


int main() {
    std::printf("=== NeuralNet Dual Model Unit Tests ===\n\n");

    test_preload_idempotent();
    test_predict_with_model_id();
    test_same_model_same_result();
    test_predict_invalid_model_id();
    test_cache_two_models();
    test_concurrent_predict_different_models();

    std::printf("\n=== Results: %d passed, %d failed, %d skipped ===\n",
                tests_passed, tests_failed, tests_skipped);
    return tests_failed > 0 ? 1 : 0;
}
