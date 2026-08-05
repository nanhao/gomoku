/** SampleAugmentor 单元测试
 *
 * 覆盖 design/49/main_docs/test_cases.md 中 TC1.1 ~ TC1.9 的所有用例。
 * 测试 D4 群 8 种对称变换的正确性。
 *
 * 运行方式：编译后直接执行，exit code 0 = 全部通过。
 */
#include "../src/game_sample.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <vector>

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name)                                         \
    do {                                                   \
        std::printf("  TEST %s ... ", name);               \
    } while (0)

#define PASS()                                             \
    do {                                                   \
        std::printf("PASS\n");                             \
        ++tests_passed;                                    \
    } while (0)

#define FAIL(msg)                                          \
    do {                                                   \
        std::printf("FAIL: %s\n", msg);                    \
        ++tests_failed;                                    \
    } while (0)

#define ASSERT_TRUE(cond)                                  \
    do {                                                   \
        if (!(cond)) { FAIL(#cond); return; }              \
    } while (0)

#define ASSERT_FALSE(cond)                                 \
    do {                                                   \
        if (cond) { FAIL(#cond " (expected false)"); return; } \
    } while (0)

#define ASSERT_EQ(a, b)                                    \
    do {                                                   \
        if ((a) != (b)) {                                  \
            std::printf("FAIL: %s == %s, expected %d got %d\n", \
                        #a, #b, (int)(b), (int)(a));       \
            ++tests_failed;                                \
            return;                                        \
        }                                                  \
    } while (0)

#define ASSERT_FLOAT_EQ(a, b, eps)                         \
    do {                                                   \
        if (std::fabs((a) - (b)) > (eps)) {                \
            std::printf("FAIL: %s ~= %s, expected %.6f got %.6f\n", \
                        #a, #b, (float)(b), (float)(a));   \
            ++tests_failed;                                \
            return;                                        \
        }                                                  \
    } while (0)


// --- TC1.1: transform_index 恒等变换 ---
static void test_transform_index_identity() {
    TEST("TC1.1 transform_index identity");
    for (int idx = 0; idx < 225; ++idx) {
        int result = SampleAugmentor::transform_index(idx, 0);
        if (result != idx) {
            std::printf("FAIL: transform_index(%d, 0) = %d, expected %d\n",
                        idx, result, idx);
            ++tests_failed;
            return;
        }
    }
    PASS();
}


// --- TC1.2: transform_index 互逆性 ---
static void test_transform_index_inverse() {
    TEST("TC1.2 transform_index inverse");

    // tid → reverse_tid 映射
    static const int rev[8] = {0, 3, 2, 1, 4, 5, 6, 7};

    for (int tid = 1; tid <= 7; ++tid) {
        int rev_tid = rev[tid];
        for (int idx = 0; idx < 225; ++idx) {
            int t1 = SampleAugmentor::transform_index(idx, tid);
            int t2 = SampleAugmentor::transform_index(t1, rev_tid);
            if (t2 != idx) {
                std::printf("FAIL: tid=%d rev=%d idx=%d: t1=%d t2=%d\n",
                            tid, rev_tid, idx, t1, t2);
                ++tests_failed;
                return;
            }
        }
    }
    PASS();
}


// --- TC1.3: transform_index 棋盘角点映射 ---
static void test_transform_index_corners() {
    TEST("TC1.3 transform_index corners");

    // idx=0 → (r=0, c=0)
    // tid=1 (顺时针90°): (c, 14-r) = (0, 14) → idx = 14
    int result = SampleAugmentor::transform_index(0, 1);
    ASSERT_EQ(result, 14);

    // idx=14 → (r=0, c=14)
    // tid=1: (14, 14) → idx = 14*15+14 = 224
    result = SampleAugmentor::transform_index(14, 1);
    ASSERT_EQ(result, 224);

    // idx=224 → (r=14, c=14)
    // tid=1: (14, 0) → idx = 14*15+0 = 210
    result = SampleAugmentor::transform_index(224, 1);
    ASSERT_EQ(result, 210);

    // idx=210 → (r=14, c=0)
    // tid=1: (0, 0) → idx = 0
    result = SampleAugmentor::transform_index(210, 1);
    ASSERT_EQ(result, 0);

    PASS();
}


// --- TC1.4: transform_225 恒等变换 ---
static void test_transform_225_identity() {
    TEST("TC1.4 transform_225 identity");

    std::vector<float> v(225);
    for (int i = 0; i < 225; ++i) {
        v[i] = static_cast<float>(i);
    }

    std::vector<float> result = SampleAugmentor::transform_225(v, 0);
    for (int i = 0; i < 225; ++i) {
        ASSERT_FLOAT_EQ(result[i], v[i], 1e-6f);
    }
    PASS();
}


// --- TC1.5: transform_225 旋转180°两次等于恒等 ---
static void test_transform_225_double_180() {
    TEST("TC1.5 transform_225 double 180");

    std::vector<float> v(225);
    for (int i = 0; i < 225; ++i) {
        v[i] = static_cast<float>(i);
    }

    std::vector<float> v1 = SampleAugmentor::transform_225(v, 2);
    std::vector<float> v2 = SampleAugmentor::transform_225(v1, 2);

    for (int i = 0; i < 225; ++i) {
        ASSERT_FLOAT_EQ(v2[i], v[i], 1e-6f);
    }
    PASS();
}


// --- TC1.6: augment 返回 8 条样本 ---
static void test_augment_count() {
    TEST("TC1.6 augment returns 8 samples");

    GameSample sample;
    sample.state.resize(226);
    sample.pi.resize(225);
    sample.z = 1.0f;
    for (int i = 0; i < 225; ++i) {
        sample.state[i] = static_cast<float>(i % 3);
        sample.pi[i] = 1.0f / 225.0f;
    }
    sample.state[225] = 1.0f;

    std::vector<GameSample> augmented = SampleAugmentor::augment(sample);
    ASSERT_EQ((int)augmented.size(), 8);
    PASS();
}


// --- TC1.7: augment — z 值不变 ---
static void test_augment_z_preserved() {
    TEST("TC1.7 augment z preserved");

    // z = 1.0
    {
        GameSample sample;
        sample.state.resize(226);
        sample.pi.resize(225);
        sample.z = 1.0f;
        for (int i = 0; i < 225; ++i) {
            sample.state[i] = static_cast<float>(i % 3);
            sample.pi[i] = 1.0f / 225.0f;
        }
        sample.state[225] = 1.0f;

        std::vector<GameSample> augmented = SampleAugmentor::augment(sample);
        for (const auto& s : augmented) {
            ASSERT_FLOAT_EQ(s.z, 1.0f, 1e-6f);
        }
    }

    // z = -1.0
    {
        GameSample sample;
        sample.state.resize(226);
        sample.pi.resize(225);
        sample.z = -1.0f;
        for (int i = 0; i < 225; ++i) {
            sample.state[i] = static_cast<float>(i % 3);
            sample.pi[i] = 1.0f / 225.0f;
        }
        sample.state[225] = 2.0f;

        std::vector<GameSample> augmented = SampleAugmentor::augment(sample);
        for (const auto& s : augmented) {
            ASSERT_FLOAT_EQ(s.z, -1.0f, 1e-6f);
        }
    }
    PASS();
}


// --- TC1.8: augment — state[225] 不变 ---
static void test_augment_player_color_preserved() {
    TEST("TC1.8 augment state[225] preserved");

    GameSample sample;
    sample.state.resize(226);
    sample.pi.resize(225);
    sample.z = 1.0f;
    for (int i = 0; i < 225; ++i) {
        sample.state[i] = static_cast<float>(i % 3);
        sample.pi[i] = 1.0f / 225.0f;
    }
    sample.state[225] = 2.0f;

    std::vector<GameSample> augmented = SampleAugmentor::augment(sample);
    for (const auto& s : augmented) {
        ASSERT_FLOAT_EQ(s.state[225], 2.0f, 1e-6f);
    }
    PASS();
}


// --- TC1.9: augment — pi 总和不变 ---
static void test_augment_pi_sum_preserved() {
    TEST("TC1.9 augment pi sum preserved");

    GameSample sample;
    sample.state.resize(226);
    sample.pi.resize(225);
    sample.z = 1.0f;
    for (int i = 0; i < 225; ++i) {
        sample.state[i] = static_cast<float>(i % 3);
        sample.pi[i] = 1.0f / 225.0f;
    }
    sample.state[225] = 1.0f;

    std::vector<GameSample> augmented = SampleAugmentor::augment(sample);
    for (const auto& s : augmented) {
        float sum = std::accumulate(s.pi.begin(), s.pi.end(), 0.0f);
        if (sum < 0.999f || sum > 1.001f) {
            std::printf("FAIL: pi sum = %.6f, expected ~1.0\n", sum);
            ++tests_failed;
            return;
        }
    }
    PASS();
}


int main() {
    std::printf("=== SampleAugmentor Unit Tests ===\n\n");

    test_transform_index_identity();
    test_transform_index_inverse();
    test_transform_index_corners();
    test_transform_225_identity();
    test_transform_225_double_180();
    test_augment_count();
    test_augment_z_preserved();
    test_augment_player_color_preserved();
    test_augment_pi_sum_preserved();

    std::printf("\n=== Results: %d passed, %d failed ===\n",
                tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
