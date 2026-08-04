#pragma once

#include <atomic>
#include <condition_variable>
#include <future>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

/** 神经网络推理结果 */
struct NeuralNetResult {
    std::vector<float> policy;  // 225 维 softmax 概率分布，值域 (0, 1)，总和 ≈ 1
    float value;                // 局面价值 [-1, 1]
};

/** 神经网络推理引擎（全局单例）

使用 LibTorch 加载 TorchScript 模型，提供异步批量推理接口。

内部架构：生产者-消费者模式
- MCTS 工作线程投递 predict() 请求 → 入队 → 立即返回 std::future
- 专有 GPU 推理线程循环：攒批 → GPU forward → 拆分结果 → 唤醒等待线程

单例模式：
- initialize() 创建全局唯一实例（call_once 保护）
- instance() 获取实例（若未初始化则用默认配置自动创建）
- set_model() 切换模型权重（线程安全）
- 旧式构造器保留，支持独立实例（向后兼容）

线程安全：所有公开方法均可多线程并发调用。
*/
class NeuralNet {
public:
    // ---- 单例生命周期 ----

    /** 获取全局单例实例（首次调用时自动初始化） */
    static NeuralNet& instance();

    /** 显式初始化全局单例（call_once 保护，仅首次生效）

    @param model_path TorchScript 模型文件路径 (.pt)
    @param batch_size GPU 推理攒批大小
    @param use_gpu 是否使用 GPU 推理
    */
    static void initialize(const std::string& model_path,
                           int batch_size, bool use_gpu);

    /** 关闭全局单例，等待推理线程退出 */
    static void shutdown();

    // ---- 模型管理 ----

    /** 切换模型权重（线程安全）

    阻塞直到当前推理批次完成，重新加载模型，
    然后恢复推理线程。若加载失败，保留旧模型并抛出异常。

    @param model_path 新模型文件路径
    @throws std::runtime_error 若文件不存在或加载失败
    */
    void set_model(const std::string& model_path);

    // ---- 推理接口 ----

    /** 异步投递推理请求

    将 encoded_board 加入推理队列，立即返回 future。
    调用方通过 future.get() 阻塞等待结果。

    @param encoded_board 3×15×15 = 675 维相对视角编码张量
    @return 异步结果的 std::future
    */
    std::future<NeuralNetResult> predict_async(
        const std::vector<float>& encoded_board);

    /** 同步推理（便捷方法）

    等同 predict_async(state).get()。

    @param encoded_board 3×15×15 = 675 维相对视角编码张量
    @return NeuralNetResult 推理结果
    */
    NeuralNetResult predict(const std::vector<float>& encoded_board);

    // ---- 构造/析构（保留旧接口，向后兼容）----

    /** 加载 TorchScript 模型并启动 GPU 推理线程

    @param model_path TorchScript 模型文件路径 (.pt)
    @param batch_size GPU 推理攒批大小，默认 16
    @param use_gpu 是否使用 GPU 推理，true=GPU, false=CPU
    */
    NeuralNet(std::string model_path, int batch_size, bool use_gpu);

    /** 停止推理线程并等待其退出 */
    ~NeuralNet();

    // 禁止拷贝和移动
    NeuralNet(const NeuralNet&) = delete;
    NeuralNet& operator=(const NeuralNet&) = delete;

private:
    /** 单次推理请求 */
    struct InferenceRequest {
        std::vector<float> encoded_board;           // 675 维输入
        std::promise<NeuralNetResult> promise;      // 结果回传 promise
    };

    std::string model_path_;
    int batch_lower_bound_;  // GPU 攒批下限（队列 < 此值继续等待）
    int batch_upper_bound_;  // GPU 攒批上限（单次最多取走这么多）
    bool use_gpu_;

    // LibTorch 模型模块（前向声明，实际类型为 torch::jit::Module）
    void* module_impl_;  // torch::jit::script::Module*

    // 推理线程
    std::thread inference_thread_;

    // 请求队列
    std::vector<InferenceRequest> pending_requests_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;

    // 停止标志
    std::atomic<bool> stop_flag_{false};

    // 模型切换互斥锁（保护 set_model / 推理线程的 model 访问）
    std::mutex model_mutex_;

    /** GPU 推理线程主循环 */
    void inference_loop();

    /** 执行批量前向推理

    @param batch 输入张量列表（每个 675 维）
    @return 等长的 NeuralNetResult 列表
    */
    std::vector<NeuralNetResult> forward_batch(
        const std::vector<std::vector<float>>& batch);

    /** 内部：加载模型到 module_impl_（不加锁，由调用方保证线程安全） */
    void load_model(const std::string& model_path);
};
