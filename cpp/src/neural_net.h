#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <shared_mutex>
#include <string>
#include <thread>
#include <vector>

/** 神经网络推理结果 */
struct NeuralNetResult {
    std::vector<float> policy;  // 225 维 softmax 概率分布，值域 (0, 1)，总和 ≈ 1
    float value;                // 局面价值 [-1, 1]
};

/** GPU 推理线程的运行时监控统计

所有耗时以微秒为单位。由推理线程单线程更新，get_stats() 返回快照。
*/
struct GpuInferenceStats {
    // ---- 全周期累计统计 ----
    int64_t total_batches = 0;    // 已处理的批次数
    int64_t total_requests = 0;   // 已处理的请求总数
    int64_t total_wait_us = 0;    // 等待攒批的总耗时（微秒）
    int64_t total_infer_us = 0;   // GPU 前向推理的总耗时（微秒）
    int queue_peak_size = 0;      // 队列长度的历史峰值
    int batch_size_min = 0;       // 历史最小批次大小
    int batch_size_max = 0;       // 历史最大批次大小

    // ---- 滑动窗口统计（由 get_stats() 计算填充）----
    int recent_batch_count = 0;           // 滑动窗口内的批次数（≤10）
    double recent_avg_batch_size = 0.0;   // 最近 10 个 batch 的平均大小
    double recent_wait_ratio = 0.0;       // 最近 5 秒内的 GPU 等待比例，0~1

    /** 全周期 GPU 线程等待比例 = wait / (wait + infer)，0~1 */
    double wait_ratio() const {
        int64_t total = total_wait_us + total_infer_us;
        return total > 0 ? static_cast<double>(total_wait_us) / total : 0.0;
    }

    /** 全周期平均批次大小 */
    double avg_batch_size() const {
        return total_batches > 0
            ? static_cast<double>(total_requests) / total_batches : 0.0;
    }
};

/** 神经网络推理引擎（全局单例）

使用 LibTorch 加载 TorchScript 模型，提供同步推理接口。

内部架构：专用推理线程 + 请求队列攒批
- 构造时加载模型到 CPU（主线程），启动推理线程
- 推理线程自行初始化 GPU（module->to(kCUDA)），避免跨线程 CUDA context 冲突
- MCTS worker 线程调用 predict() → 提交请求到队列，阻塞等待 promise 结果
- 推理线程收集 batch 后批量 GPU forward，结果通过 promise 返回
- set_model 时获取独占写锁，加载新模型到 CPU，推理线程下一 batch 自动迁移

单例模式：
- initialize() 创建全局唯一实例（call_once 保护）
- instance() 获取实例（若未初始化则用默认配置自动创建）
- set_model() 切换模型权重（线程安全）
- 旧式构造器保留，支持独立实例（向后兼容）

线程安全：predict() 可多线程并发提交；推理线程单线程执行 GPU 操作；
set_model() 独占写锁阻塞推理线程的下一 batch。
*/
class NeuralNet {
public:
    // ---- 单例生命周期 ----

    /** 获取全局单例实例（首次调用时自动初始化） */
    static NeuralNet& instance();

    /** 显式初始化全局单例（call_once 保护，仅首次生效）

    @param model_path TorchScript 模型文件路径 (.pt)
    @param use_gpu 是否使用 GPU 推理
    */
    static void initialize(const std::string& model_path, bool use_gpu);

    /** 关闭全局单例 */
    static void shutdown();

    // ---- 模型管理 ----

    /** 切换模型权重（线程安全，独占写锁）

    若新路径与当前模型路径相同则跳过（幂等）。
    加载失败时保留旧模型并抛出异常。

    @param model_path 新模型文件路径
    @throws std::runtime_error 若文件不存在或加载失败
    */
    void set_model(const std::string& model_path);

    // ---- 推理接口 ----

    /** 同步推理（内部通过推理线程攒批后批量 GPU/CPU forward）

    MCTS worker 线程调用此方法，内部提交请求到推理线程队列，
    阻塞等待 promise 结果。推理线程收集多个请求后批量执行 forward。

    @param encoded_board 3×15×15 = 675 维相对视角编码张量
    @return NeuralNetResult 推理结果
    */
    NeuralNetResult predict(const std::vector<float>& encoded_board);

    // ---- 运行时监控 ----

    /** 获取 GPU 推理线程的运行时统计（线程安全，返回快照）

    @return GpuInferenceStats 当前统计快照
    */
    GpuInferenceStats get_stats() const;

    // ---- 构造/析构（保留旧接口，向后兼容）----

    /** 加载 TorchScript 模型

    攒批参数从 mcts_config.json 读取（batch_lower_bound / batch_upper_bound）。

    @param model_path TorchScript 模型文件路径 (.pt)
    @param use_gpu 是否使用 GPU 推理，true=GPU, false=CPU
    */
    NeuralNet(std::string model_path, bool use_gpu);

    /** 释放 LibTorch 模块 */
    ~NeuralNet();

    // 禁止拷贝和移动
    NeuralNet(const NeuralNet&) = delete;
    NeuralNet& operator=(const NeuralNet&) = delete;

private:
    // ---- 推理线程 ----
    void inference_loop();

    std::thread inference_thread_;
    std::atomic<bool> stop_flag_{false};

    struct PendingRequest {
        std::vector<float> encoded_board;
        std::promise<NeuralNetResult> promise;
    };
    std::queue<PendingRequest> pending_requests_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;

    // ---- 模型 ----
    std::string model_path_;
    int batch_lower_bound_;  // GPU 攒批下限
    int batch_upper_bound_;  // GPU 攒批上限
    bool use_gpu_;

    // LibTorch 模型模块（前向声明，实际类型为 torch::jit::Module）
    void* module_impl_;  // torch::jit::script::Module*

    // 模型保护锁：inference_loop 用 shared_lock（GPU forward），set_model 用 unique_lock（独占写）
    mutable std::shared_mutex model_mutex_;

    // ---- 运行时监控（仅推理线程写入，get_stats 加锁读取）----
    GpuInferenceStats stats_;
    mutable std::mutex stats_mutex_;
    int stats_log_interval_ = 100;    // 每 N 个 batch 输出一次监控日志

    // ---- 滑动窗口原始数据（仅推理线程写入，get_stats 加锁读取）----
    static constexpr int kRecentBatchWindow = 10;          // avg_batch 窗口大小
    static constexpr int64_t kRecentTimeWindowUs = 5'000'000;  // wait_ratio 窗口: 5秒（微秒）
    std::deque<int> recent_batch_sizes_;                       // 最近 N 个 batch 大小
    std::deque<std::pair<std::chrono::steady_clock::time_point,
                         std::pair<int64_t, int64_t>>>
        recent_timings_;  // (timestamp, (wait_us, infer_us)) 按时间排序

    /** 执行批量前向推理（调用方需先获取 model_mutex_ 的 shared_lock）

    @param batch 输入张量列表（每个 675 维）
    @return 等长的 NeuralNetResult 列表
    */
    std::vector<NeuralNetResult> forward_batch(
        const std::vector<std::vector<float>>& batch);

    /** 内部：加载模型到 module_impl_（不加锁，由调用方保证线程安全） */
    void load_model(const std::string& model_path);
};
