#pragma once

#include <atomic>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

/// Global persistent thread pool for MCTS simulations.
///
/// Worker threads are created once (on first use) and reused across ALL
/// searches and ALL games.  This avoids the per-search thread creation/
/// destruction overhead (32 threads × 30+ moves = ~1000 spawns per game).
///
/// Supports two modes:
/// - Batch mode (run_parallel): all workers execute the same task_fn.
///   Blocking — only one batch can run at a time.
/// - Queue mode (submit + wait_all): multiple callers submit individual
///   tasks to a shared queue.  Non-blocking submits, wait_all blocks.
///   Supports concurrent multi-game usage.
class MctsThreadPool {
public:
    /// Get the singleton instance (auto-init with kNumThreads on first call).
    static MctsThreadPool& instance();

    /// Explicitly initialise with a specific thread count.
    /// Must be called before any other usage; call_once protected.
    static void initialize(int num_threads);

    /// Shut down the pool and join all workers (idempotent, safe at exit).
    static void shutdown();

    /// Launch task_fn on ALL worker threads and block until every one
    /// has finished.  task_fn is expected to internally loop via a shared
    /// atomic counter (e.g. sim_counter_.fetch_add()).
    void run_parallel(std::function<void()> task_fn);

    /// Queue mode: submit a single task.  Workers pull tasks from the
    /// shared queue in FIFO order.  Thread-safe, multiple callers OK.
    void submit(std::function<void()> task);

    /// Queue mode: block until all queued tasks have been executed.
    void wait_all();

    /// Queue mode: approximate number of tasks submitted but not yet
    /// completed (includes tasks currently executing).
    int pending_count() const;

    /// Number of worker threads.
    int size() const { return static_cast<int>(workers_.size()); }

    ~MctsThreadPool();

private:
    explicit MctsThreadPool(int num_threads);

    MctsThreadPool(const MctsThreadPool&) = delete;
    MctsThreadPool& operator=(const MctsThreadPool&) = delete;

    void worker_loop();

    std::vector<std::thread> workers_;

    std::mutex mtx_;
    std::condition_variable cv_start_;   // wake workers (both modes)
    std::condition_variable cv_done_;    // wake caller (both modes)

    // --- batch mode state ---
    std::function<void()> current_task_;
    int pending_count_{0};    // workers that haven't finished yet
    int launch_count_{0};     // token counter: one per worker per batch
    bool running_{false};     // barrier flag

    // --- queue mode state ---
    std::queue<std::function<void()>> task_queue_;
    std::atomic<int> queue_pending_{0};  // tasks submitted but not yet completed

    bool stop_{false};        // shutdown flag
};
