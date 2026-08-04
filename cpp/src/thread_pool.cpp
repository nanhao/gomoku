#include "thread_pool.h"
#include "mcts_config.h"

#include <mutex>

// ---------------------------------------------------------------------------
// Singleton management
// ---------------------------------------------------------------------------

static std::unique_ptr<MctsThreadPool> s_instance;
static std::once_flag s_init_flag;

MctsThreadPool& MctsThreadPool::instance() {
    std::call_once(s_init_flag, [] {
        s_instance.reset(new MctsThreadPool(g_mcts_config.num_threads));
    });
    return *s_instance;
}

void MctsThreadPool::initialize(int num_threads) {
    std::call_once(s_init_flag, [num_threads] {
        s_instance.reset(new MctsThreadPool(num_threads));
    });
}

void MctsThreadPool::shutdown() {
    // unique_ptr reset → destructor joins all workers
    s_instance.reset();
}

// ---------------------------------------------------------------------------
// Construction / destruction
// ---------------------------------------------------------------------------

MctsThreadPool::MctsThreadPool(int num_threads) {
    workers_.reserve(num_threads);
    for (int i = 0; i < num_threads; ++i) {
        workers_.emplace_back(&MctsThreadPool::worker_loop, this);
    }
}

MctsThreadPool::~MctsThreadPool() {
    {
        std::lock_guard<std::mutex> lock(mtx_);
        stop_ = true;
    }
    cv_start_.notify_all();
    for (auto& w : workers_) {
        if (w.joinable()) w.join();
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

void MctsThreadPool::run_parallel(std::function<void()> task_fn) {
    int n = static_cast<int>(workers_.size());

    // 1. Publish the task with one launch token per worker.
    {
        std::lock_guard<std::mutex> lock(mtx_);
        current_task_ = std::move(task_fn);
        pending_count_ = n;
        launch_count_ = n;
        running_ = true;
    }
    cv_start_.notify_all();

    // 2. Block until every worker has finished.
    {
        std::unique_lock<std::mutex> lock(mtx_);
        cv_done_.wait(lock, [this] { return pending_count_ == 0; });
        running_ = false;
    }
}

// ---------------------------------------------------------------------------
// Worker loop
// ---------------------------------------------------------------------------

void MctsThreadPool::worker_loop() {
    while (true) {
        // Wait for a launch token (one per worker per batch).
        // Using launch_count_ prevents re-entering the task after
        // completion while running_ is still true.
        {
            std::unique_lock<std::mutex> lock(mtx_);
            cv_start_.wait(lock, [this] {
                return (running_ && launch_count_ > 0) || stop_;
            });
        }

        if (stop_) break;

        // Claim a token.
        {
            std::lock_guard<std::mutex> lock(mtx_);
            if (launch_count_ > 0) {
                --launch_count_;
            } else {
                continue;  // spurious wakeup
            }
        }

        // Execute the task.
        if (current_task_) {
            current_task_();
        }

        // Signal completion — last worker wakes the caller.
        {
            std::lock_guard<std::mutex> lock(mtx_);
            if (--pending_count_ == 0) {
                cv_done_.notify_one();
            }
        }
    }
}
