#pragma once

#include <atomic>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

/// Global persistent thread pool for MCTS simulations.
///
/// Worker threads are created once (on first use) and reused across ALL
/// searches and ALL games.  This avoids the per-search thread creation/
/// destruction overhead (32 threads × 30+ moves = ~1000 spawns per game).
///
/// Thread safety: run_parallel() is blocking — only one batch of tasks
/// executes at a time.  This matches the single-game, single-searcher
/// usage pattern.
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
    std::condition_variable cv_start_;   // wake workers
    std::condition_variable cv_done_;    // wake caller

    std::function<void()> current_task_;
    int pending_count_{0};    // workers that haven't finished yet
    int launch_count_{0};     // token counter: one per worker per batch
    bool running_{false};     // barrier flag
    bool stop_{false};        // shutdown flag
};
