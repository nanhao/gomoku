#include "thread_pool.h"
#include "mcts_config.h"

#include <atomic>
#include <mutex>
#include <queue>

// ---------------------------------------------------------------------------
// Singleton management
// ---------------------------------------------------------------------------

static std::unique_ptr<MctsThreadPool> s_instance;
static std::atomic<bool> s_initialized{false};
static std::mutex s_init_mutex;

MctsThreadPool& MctsThreadPool::instance() {
    // 快速路径：已初始化则直接返回（无锁）
    if (!s_initialized.load(std::memory_order_acquire)) {
        std::lock_guard<std::mutex> lock(s_init_mutex);
        // 双重检查：可能在等锁期间被其他线程初始化
        if (!s_initialized.load(std::memory_order_relaxed)) {
            s_instance.reset(new MctsThreadPool(g_mcts_config.num_threads));
            s_initialized.store(true, std::memory_order_release);
        }
    }
    return *s_instance;
}

void MctsThreadPool::initialize(int num_threads) {
    if (!s_initialized.load(std::memory_order_acquire)) {
        std::lock_guard<std::mutex> lock(s_init_mutex);
        if (!s_initialized.load(std::memory_order_relaxed)) {
            s_instance.reset(new MctsThreadPool(num_threads));
            s_initialized.store(true, std::memory_order_release);
        }
    }
}

void MctsThreadPool::shutdown() {
    // 加锁保护：防止 shutdown 与 initialize/instance 并发时出现 use-after-free
    std::lock_guard<std::mutex> lock(s_init_mutex);
    s_instance.reset();
    s_initialized.store(false, std::memory_order_release);
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
// Public API — batch mode (backward compatible)
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
// Public API — queue mode (new)
// ---------------------------------------------------------------------------

void MctsThreadPool::submit(std::function<void()> task) {
    {
        std::lock_guard<std::mutex> lock(mtx_);
        task_queue_.push(std::move(task));
        ++queue_pending_;
    }
    cv_start_.notify_one();
}

void MctsThreadPool::wait_all() {
    std::unique_lock<std::mutex> lock(mtx_);
    cv_done_.wait(lock, [this] {
        return queue_pending_ == 0 && task_queue_.empty();
    });
}

int MctsThreadPool::pending_count() const {
    // queue_pending_ counts submitted-but-not-completed tasks.
    // Since we don't have a separate "in-flight" counter,
    // return the queue size as a conservative estimate.
    // The actual pending count is queue_pending_ (in queue + executing).
    return queue_pending_.load();
}

// ---------------------------------------------------------------------------
// Worker loop
// ---------------------------------------------------------------------------

void MctsThreadPool::worker_loop() {
    while (true) {
        std::function<void()> task;
        int mode = 0;  // 0=none, 1=batch, 2=queue

        // Wait for work (batch launch token or queue task).
        {
            std::unique_lock<std::mutex> lock(mtx_);
            cv_start_.wait(lock, [this] {
                return stop_
                    || (running_ && launch_count_ > 0)
                    || !task_queue_.empty();
            });

            if (stop_) break;

            // Priority: batch mode first (launch token)
            if (running_ && launch_count_ > 0) {
                --launch_count_;
                task = current_task_;
                mode = 1;
            } else if (!task_queue_.empty()) {
                task = std::move(task_queue_.front());
                task_queue_.pop();
                mode = 2;
            } else {
                continue;  // spurious wakeup
            }
        }

        // Execute the task.
        if (task) {
            task();
        }

        // Signal completion.
        {
            std::lock_guard<std::mutex> lock(mtx_);
            if (mode == 1) {
                // Batch mode: decrement pending count
                if (--pending_count_ == 0) {
                    cv_done_.notify_one();
                }
            } else if (mode == 2) {
                // Queue mode: decrement queue pending
                if (--queue_pending_ == 0 && task_queue_.empty()) {
                    cv_done_.notify_all();
                }
            }
        }
    }
}
