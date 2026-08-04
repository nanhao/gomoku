/** GameRunner worker 分发逻辑 单元测试

验证 run_worker_games 中轮询分发算法的正确性：
- 每个对局恰好分配给一个 worker
- 分配遵循 worker w → game w, w+K, w+2K, ... 模式
- 边界条件：1 worker、worker 数 ≡ 对局数、worker > 对局数
*/

#include <algorithm>
#include <cstdio>
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

#define ASSERT_EQ(a, b)                                    \
    do {                                                   \
        if ((a) != (b)) {                                  \
            std::printf("FAIL: %s(%d) != %s(%d)\n",        \
                        #a, (int)(a), #b, (int)(b));       \
            ++tests_failed; return;                        \
        }                                                  \
    } while (0)

/** 模拟 run_worker_games 的轮询分发算法

与 GameRunner::run_worker_games 保持一致的逻辑：
worker w 负责对局 w, w+K, w+2K, ... （K = num_workers）

@return assignments[g] = worker_index（每个对局被哪个 worker 执行）
*/
static std::vector<int> simulate_distribution(int num_games, int num_workers) {
    std::vector<int> assignments(num_games, -1);
    for (int w = 0; w < num_workers; ++w) {
        for (int g = w; g < num_games; g += num_workers) {
            assignments[g] = w;  // worker w 负责对局 g
        }
    }
    return assignments;
}

// TC-GR-W01: 所有对局都被分配（无遗漏）
static void test_all_games_assigned() {
    TEST("TC-GR-W01 all games assigned — no gaps");

    auto a = simulate_distribution(100, 4);
    for (int g = 0; g < 100; ++g) {
        ASSERT_TRUE(a[g] >= 0 && a[g] < 4);
    }
    PASS();
}

// TC-GR-W02: 轮询模式 — worker 0 负责 0, 4, 8, ...
static void test_round_robin_pattern() {
    TEST("TC-GR-W02 round-robin pattern — worker 0 gets 0,4,8,...");

    auto a = simulate_distribution(100, 4);

    // worker 0: games 0, 4, 8, ... (步长为 num_workers)
    for (int g = 0; g < 100; g += 4) {
        ASSERT_EQ(a[g], 0);
    }
    // worker 1: games 1, 5, 9, ...
    for (int g = 1; g < 100; g += 4) {
        ASSERT_EQ(a[g], 1);
    }
    // worker 2: games 2, 6, 10, ...
    for (int g = 2; g < 100; g += 4) {
        ASSERT_EQ(a[g], 2);
    }
    // worker 3: games 3, 7, 11, ...
    for (int g = 3; g < 100; g += 4) {
        ASSERT_EQ(a[g], 3);
    }
    PASS();
}

// TC-GR-W03: 单 worker — 所有对局归同一个 worker
static void test_single_worker() {
    TEST("TC-GR-W03 single worker — all games to worker 0");

    auto a = simulate_distribution(10, 1);
    for (int g = 0; g < 10; ++g) {
        ASSERT_EQ(a[g], 0);
    }
    PASS();
}

// TC-GR-W04: worker 数 ≡ 对局数（每 worker 一局）
static void test_workers_equal_games() {
    TEST("TC-GR-W04 workers == games — one game per worker");

    auto a = simulate_distribution(5, 5);
    for (int g = 0; g < 5; ++g) {
        ASSERT_EQ(a[g], g);  // worker g 负责 game g
    }
    PASS();
}

// TC-GR-W05: worker 数 > 对局数 — 多余 worker 空闲
static void test_more_workers_than_games() {
    TEST("TC-GR-W05 more workers than games — extra workers idle");

    auto a = simulate_distribution(3, 10);
    // game 0 → worker 0, game 1 → worker 1, game 2 → worker 2
    ASSERT_EQ(a[0], 0);
    ASSERT_EQ(a[1], 1);
    ASSERT_EQ(a[2], 2);
    // workers 3-9 无对局（确认所有对局仍被正常分配）
    for (int g = 0; g < 3; ++g) {
        ASSERT_TRUE(a[g] >= 0);
    }
    PASS();
}

// TC-GR-W06: 不均等分配 — 验证每个 worker 的对局数合理
static void test_uneven_distribution() {
    TEST("TC-GR-W06 uneven distribution — game counts per worker");

    int num_games = 100;
    int num_workers = 6;
    auto a = simulate_distribution(num_games, num_workers);

    // 统计每个 worker 的对局数
    std::vector<int> counts(num_workers, 0);
    for (int g = 0; g < num_games; ++g) {
        counts[a[g]]++;
    }

    // 100/6 = 16 余 4，前 4 个 worker 负责 17 局，后 2 个负责 16 局
    for (int w = 0; w < 4; ++w) {
        ASSERT_EQ(counts[w], 17);
    }
    for (int w = 4; w < 6; ++w) {
        ASSERT_EQ(counts[w], 16);
    }
    PASS();
}


int main() {
    std::printf("=== GameRunner Worker Distribution Unit Tests ===\n\n");

    test_all_games_assigned();
    test_round_robin_pattern();
    test_single_worker();
    test_workers_equal_games();
    test_more_workers_than_games();
    test_uneven_distribution();

    std::printf("\n=== Results: %d passed, %d failed ===\n",
                tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
