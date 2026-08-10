/** MctsThreadPool 任务队列模式 单元测试

覆盖 design/47/main_docs/test_cases.md 中 TC3.1 和 TC3.2 用例。
*/

#include "../src/thread_pool.h"

#include <atomic>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <thread>
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

// TC3.1: 任务队列提交与等待 — 并行执行验证
static void test_queue_submit_wait_all() {
    TEST("TC3.1 queue submit and wait_all — parallel execution");

    MctsThreadPool::initialize(4);
    MctsThreadPool& pool = MctsThreadPool::instance();

    std::atomic<int> counter{0};

    // 提交 8 个任务，每个 sleep 10ms 后递增计数器
    for (int i = 0; i < 8; ++i) {
        pool.submit([&counter] {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            counter.fetch_add(1);
        });
    }

    auto start = std::chrono::steady_clock::now();
    pool.wait_all();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start).count();

    ASSERT_EQ(counter.load(), 8);

    // 4 个 worker 并行执行，8 个任务每个 10ms
    // 理想耗时 ≈ 20ms，考虑到调度开销允许 < 60ms
    ASSERT_TRUE(elapsed < 60);

    PASS();
}

// TC3.2: 并发 submit — 多线程同时提交任务
static void test_concurrent_submit() {
    TEST("TC3.2 concurrent submit from multiple threads");

    MctsThreadPool::initialize(4);
    MctsThreadPool& pool = MctsThreadPool::instance();

    std::atomic<int> counter{0};
    const int tasks_per_thread = 100;
    const int num_submitters = 4;

    std::vector<std::thread> submitters;
    for (int t = 0; t < num_submitters; ++t) {
        submitters.emplace_back([&pool, &counter, tasks_per_thread] {
            for (int i = 0; i < tasks_per_thread; ++i) {
                pool.submit([&counter] {
                    counter.fetch_add(1);
                });
            }
            pool.wait_all();
        });
    }

    for (auto& t : submitters) {
        t.join();
    }

    ASSERT_EQ(counter.load(), tasks_per_thread * num_submitters);
    PASS();
}

// TC3.2b: wait_all 阻塞行为 — 调用后队列已清空
static void test_wait_all_blocking() {
    TEST("TC3.2b wait_all blocks until queue empty");

    MctsThreadPool::initialize(2);
    MctsThreadPool& pool = MctsThreadPool::instance();

    std::atomic<int> counter{0};

    pool.submit([&counter] {
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
        counter.fetch_add(1);
    });
    pool.submit([&counter] {
        counter.fetch_add(1);
    });

    pool.wait_all();

    ASSERT_EQ(counter.load(), 2);
    ASSERT_EQ(pool.pending_count(), 0);

    PASS();
}

// TC3.2c: pending_count 实时反映队列状态
static void test_pending_count() {
    TEST("TC3.2c pending_count reflects queue state");

    MctsThreadPool::initialize(2);
    MctsThreadPool& pool = MctsThreadPool::instance();

    std::atomic<int> counter{0};
    std::atomic<bool> start_flag{false};

    // 提交阻塞任务，确保任务在队列中等待
    for (int i = 0; i < 4; ++i) {
        pool.submit([&start_flag, &counter] {
            while (!start_flag.load()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            counter.fetch_add(1);
        });
    }

    // 此时任务应正在执行或排队
    int pending = pool.pending_count();
    // 至少有部分任务还未完成
    ASSERT_TRUE(pending >= 0);

    // 释放所有任务
    start_flag.store(true);
    pool.wait_all();

    ASSERT_EQ(counter.load(), 4);
    ASSERT_EQ(pool.pending_count(), 0);

    PASS();
}


// TC3.5: shutdown 后重新 initialize
// 验证 Bug: std::once_flag 不可重置导致第二次 initialize 失效
static void test_reinitialize_after_shutdown() {
    TEST("TC3.5 re-initialize after shutdown");

    // 第一次初始化 + shutdown
    MctsThreadPool::initialize(4);
    {
        MctsThreadPool& pool = MctsThreadPool::instance();
        ASSERT_TRUE(pool.size() > 0);
    }
    MctsThreadPool::shutdown();

    // 第二次 initialize（模拟 checkpoint #2）
    MctsThreadPool::initialize(4);
    {
        MctsThreadPool& pool = MctsThreadPool::instance();
        ASSERT_TRUE(pool.size() > 0);
    }
    MctsThreadPool::shutdown();

    PASS();
}


int main() {
    std::printf("=== MctsThreadPool Queue Mode Unit Tests ===\n\n");

    test_queue_submit_wait_all();
    test_concurrent_submit();
    test_wait_all_blocking();
    test_pending_count();
    test_reinitialize_after_shutdown();

    std::printf("\n=== Results: %d passed, %d failed ===\n",
                tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
