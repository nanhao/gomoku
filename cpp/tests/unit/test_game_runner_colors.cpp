/** GameRunner 执方分配逻辑 单元测试

验证 resolve_colors 的执方分配正确性：
- alternate_colors=true（或自对弈）：按局号奇偶严格交替执方
- alternate_colors=false（双模型默认）：随机分配，黑白比例约 50/50
- 边界：局号 0、奇数局、执方值合法性（1/2 互补）
*/

#include <cstdio>
#include <utility>

#include "game_runner.h"

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

#define ASSERT_EQ(a, b)                                    \
    do {                                                   \
        if ((a) != (b)) {                                  \
            std::printf("FAIL: %s(%d) != %s(%d)\n",        \
                        #a, (int)(a), #b, (int)(b));       \
            ++tests_failed; return;                        \
        }                                                  \
    } while (0)

/** 校验一对执方值的合法性：均为 1/2 且互补 */
static bool is_valid_colors(int m1, int m2) {
    return (m1 == 1 || m1 == 2) && (m1 + m2 == 3);
}

// TC-CPP-01: 双模型 + alternate_colors=false → 随机分配（合法且比例约 50%）
static void test_random_colors_when_not_alternating() {
    TEST("TC-CPP-01 random colors — legal & ~50/50 over 2000 games");

    int black_count = 0;
    const int total = 2000;
    for (int g = 0; g < total; ++g) {
        auto [m1, m2] = GameRunner::resolve_colors(g, /*is_self_play=*/false,
                                                   /*alternate_colors=*/false);
        ASSERT_TRUE(is_valid_colors(m1, m2));
        if (m1 == 1) {
            black_count++;
        }
    }
    // 2000 局中 model1 执黑比例应在 45%~55% 之间（统计噪声容忍）
    double ratio = static_cast<double>(black_count) / total;
    ASSERT_TRUE(ratio > 0.45 && ratio < 0.55);
    PASS();
}

// TC-CPP-02: 双模型 + alternate_colors=true → 严格交替执方
static void test_alternate_colors_strict() {
    TEST("TC-CPP-02 alternate colors — even games black, odd games white");

    for (int g = 0; g < 10; ++g) {
        auto [m1, m2] = GameRunner::resolve_colors(g, /*is_self_play=*/false,
                                                   /*alternate_colors=*/true);
        if (g % 2 == 0) {
            // 偶数局：model1 执黑（1），model2 执白（2）
            ASSERT_EQ(m1, 1);
            ASSERT_EQ(m2, 2);
        } else {
            // 奇数局：model1 执白（2），model2 执黑（1）
            ASSERT_EQ(m1, 2);
            ASSERT_EQ(m2, 1);
        }
    }
    PASS();
}

// TC-CPP-03: 自对弈模式 → 始终交替（回归保护，与 alternate_colors 无关）
static void test_self_play_always_alternates() {
    TEST("TC-CPP-03 self-play always alternates regardless of flag");

    for (int g = 0; g < 10; ++g) {
        auto [m1, m2] = GameRunner::resolve_colors(g, /*is_self_play=*/true,
                                                   /*alternate_colors=*/false);
        if (g % 2 == 0) {
            ASSERT_EQ(m1, 1);
            ASSERT_EQ(m2, 2);
        } else {
            ASSERT_EQ(m1, 2);
            ASSERT_EQ(m2, 1);
        }
    }
    PASS();
}

// TC-CPP-04: 边界 — 局号 0 与 1（首两局的交替起点）
static void test_boundary_first_two_games() {
    TEST("TC-CPP-04 boundary — game 0 and game 1");

    auto [m1_0, m2_0] = GameRunner::resolve_colors(0, false, true);
    ASSERT_EQ(m1_0, 1);
    ASSERT_EQ(m2_0, 2);

    auto [m1_1, m2_1] = GameRunner::resolve_colors(1, false, true);
    ASSERT_EQ(m1_1, 2);
    ASSERT_EQ(m2_1, 1);
    PASS();
}


int main() {
    std::printf("=== GameRunner Color Assignment Unit Tests ===\n\n");

    test_random_colors_when_not_alternating();
    test_alternate_colors_strict();
    test_self_play_always_alternates();
    test_boundary_first_two_games();

    std::printf("\n=== Results: %d passed, %d failed ===\n",
                tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
