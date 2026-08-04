#pragma once

#include <atomic>
#include <condition_variable>
#include <future>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

/** 神经网络推理结果 */
struct NeuralNetResult {
    std::vector<float> policy;  // 225 维 softmax 概率分布，值域 (0, 1)，总和 ≈ 1
    float value;                // 局面价值 [-1, 1]
};

/** 神经网络推理引擎

使用 LibTorch 加载 TorchScript 模型，提供异步批量推理接口。

内部架构：生产者-消费者模式
- MCTS 工作线程投递 predict() 请求 → 入队 → 立即返回 std::future
- 专有 GPU 推理线程循环：攒批 → GPU forward → 拆分结果 → 唤醒等待线程

线程安全：所有公开方法均可多线程并发调用。
*/
class NeuralNet {
public:
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

    /** 异步投递推理请求

    将 encoded_board 加入推理队列，立即返回 future。
    调用方通过 future.get() 阻塞等待结果。

    @param encoded_board 3×15×15 = 675 维相对视角编码张量
    @return 异步结果的 std::future
    */
    std::future<NeuralNetResult> predict(const std::vector<float>& encoded_board);

private:
    /** 单次推理请求 */
    struct InferenceRequest {
        std::vector<float> encoded_board;           // 675 维输入
        std::promise<NeuralNetResult> promise;      // 结果回传 promise
    };

    std::string model_path_;
    int batch_size_;
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

    /** GPU 推理线程主循环 */
    void inference_loop();

    /** 执行批量前向推理

    @param batch 输入张量列表（每个 675 维）
    @return 等长的 NeuralNetResult 列表
    */
    std::vector<NeuralNetResult> forward_batch(
        const std::vector<std::vector<float>>& batch);
};
