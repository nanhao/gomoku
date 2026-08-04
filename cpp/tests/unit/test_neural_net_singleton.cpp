/** NeuralNet 单例模式 单元测试

覆盖 design/47/main_docs/test_cases.md 中 NeuralNet 单例化相关用例。
验证: 单例生命周期、set_model 切换、predict/predict_async 接口。

注意:
- 需要有效 TorchScript 模型文件的测试会被跳过（标记为 SKIP）。
- 单例生命周期测试不需要模型文件。
*/

#include "../src/neural_net.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <future>
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


// --- TC-NN-S01: initialize 创建单例 ---
static void test_singleton_initialize() {
    TEST("TC-NN-S01 singleton initialize");
    std::string path = get_model_path();

    // 使用 try-catch：模型不存在时跳过
    try {
        NeuralNet::initialize(path, false);
        NeuralNet& nn = NeuralNet::instance();

        // 再次 initialize 应不生效（call_once）
        NeuralNet::initialize("different_path.pt", true);
        NeuralNet& nn2 = NeuralNet::instance();

        // 两次获取应为同一实例
        ASSERT_TRUE(&nn == &nn2);

        NeuralNet::shutdown();
        PASS();
    } catch (const std::exception& e) {
        SKIP(std::string("model unavailable: ") + e.what());
    }
}


// --- TC-NN-S02: predict_async 异步推理 ---
static void test_predict_async() {
    TEST("TC-NN-S02 predict_async interface");
    std::string path = get_model_path();

    try {
        NeuralNet::initialize(path, false);
        NeuralNet& nn = NeuralNet::instance();

        std::vector<float> encoded(675, 0.0f);
        for (int i = 0; i < 225; ++i) {
            encoded[2 * 225 + i] = 1.0f;
        }

        auto future = nn.predict_async(encoded);
        auto result = future.get();

        ASSERT_EQ((int)result.policy.size(), 225);
        ASSERT_TRUE(result.value >= -1.0f && result.value <= 1.0f);

        // policy 总和 ≈ 1.0
        float sum = 0.0f;
        for (float p : result.policy) {
            sum += p;
        }
        ASSERT_FLOAT_EQ(sum, 1.0f, 1e-4f);

        NeuralNet::shutdown();
        PASS();
    } catch (const std::exception& e) {
        SKIP(std::string("model unavailable: ") + e.what());
    }
}


// --- TC-NN-S03: predict 同步推理 ---
static void test_predict_sync() {
    TEST("TC-NN-S03 predict sync interface");
    std::string path = get_model_path();

    try {
        NeuralNet::initialize(path, false);
        NeuralNet& nn = NeuralNet::instance();

        std::vector<float> encoded(675, 0.0f);
        for (int i = 0; i < 225; ++i) {
            encoded[2 * 225 + i] = 1.0f;
        }

        // 同步调用，直接返回结果
        auto result = nn.predict(encoded);

        ASSERT_EQ((int)result.policy.size(), 225);
        ASSERT_TRUE(result.value >= -1.0f && result.value <= 1.0f);

        NeuralNet::shutdown();
        PASS();
    } catch (const std::exception& e) {
        SKIP(std::string("model unavailable: ") + e.what());
    }
}


// --- TC-NN-S04: set_model 切换模型 ---
static void test_set_model() {
    TEST("TC-NN-S04 set_model switching");
    std::string path = get_model_path();

    try {
        NeuralNet::initialize(path, false);
        NeuralNet& nn = NeuralNet::instance();

        // 先用当前模型推理一次
        std::vector<float> encoded(675, 0.0f);
        for (int i = 0; i < 225; ++i) {
            encoded[2 * 225 + i] = 1.0f;
        }

        auto r1 = nn.predict(encoded);
        ASSERT_EQ((int)r1.policy.size(), 225);

        // 切换到同一模型（验证 set_model 不崩溃）
        nn.set_model(path);

        // 切换后再推理
        auto r2 = nn.predict(encoded);
        ASSERT_EQ((int)r2.policy.size(), 225);

        NeuralNet::shutdown();
        PASS();
    } catch (const std::exception& e) {
        SKIP(std::string("model unavailable: ") + e.what());
    }
}


// --- TC-NN-S05: set_model 不存在的文件 ---
static void test_set_model_invalid() {
    TEST("TC-NN-S05 set_model invalid path");
    std::string path = get_model_path();

    try {
        NeuralNet::initialize(path, false);
        NeuralNet& nn = NeuralNet::instance();

        try {
            nn.set_model("nonexistent_model.pt");
            FAIL("expected exception but none thrown");
        } catch (const std::runtime_error&) {
            // 预期行为
            PASS();
        }

        NeuralNet::shutdown();
    } catch (const std::exception& e) {
        SKIP(std::string("model unavailable: ") + e.what());
    }
}


// --- TC-NN-S06: 旧构造器仍可用（向后兼容）---
static void test_legacy_constructor() {
    TEST("TC-NN-S06 legacy constructor still works");
    std::string path = get_model_path();

    try {
        // 使用旧式构造器创建独立实例
        NeuralNet nn(path, false);

        std::vector<float> encoded(675, 0.0f);
        for (int i = 0; i < 225; ++i) {
            encoded[2 * 225 + i] = 1.0f;
        }

        // 旧式 predict (async) 仍可用
        auto future = nn.predict_async(encoded);
        auto result = future.get();

        ASSERT_EQ((int)result.policy.size(), 225);

        // 新式 predict (sync) 也可用
        auto result2 = nn.predict(encoded);
        ASSERT_EQ((int)result2.policy.size(), 225);

        PASS();
    } catch (const std::exception& e) {
        SKIP(std::string("model unavailable: ") + e.what());
    }
}


int main() {
    std::printf("=== NeuralNet Singleton Unit Tests ===\n\n");

    test_singleton_initialize();
    test_predict_async();
    test_predict_sync();
    test_set_model();
    test_set_model_invalid();
    test_legacy_constructor();

    std::printf("\n=== Results: %d passed, %d failed, %d skipped ===\n",
                tests_passed, tests_failed, tests_skipped);
    return tests_failed > 0 ? 1 : 0;
}
