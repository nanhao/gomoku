#include "neural_net.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>

// LibTorch headers
#include <torch/script.h>
#include <torch/torch.h>

// =====================================================================
// 构造 & 析构
// =====================================================================

NeuralNet::NeuralNet(std::string model_path, int batch_size, bool use_gpu)
    : model_path_(std::move(model_path))
    , batch_size_(batch_size)
    , use_gpu_(use_gpu)
    , module_impl_(nullptr)
{
    // 确定设备
    torch::Device device(torch::kCPU);
    if (use_gpu_ && torch::cuda::is_available()) {
        device = torch::Device(torch::kCUDA);
    }

    // 加载 TorchScript 模型
    try {
        torch::jit::script::Module module = torch::jit::load(model_path_);
        module.to(device);
        module.eval();

        // 将模块存储在堆上（void* 封装避免头文件暴露 LibTorch 类型）
        auto* heap_module = new torch::jit::script::Module(std::move(module));
        module_impl_ = static_cast<void*>(heap_module);
    } catch (const c10::Error& e) {
        throw std::runtime_error(
            "NeuralNet: 加载 TorchScript 模型失败: " + model_path_
            + " — " + std::string(e.what()));
    }

    // 启动 GPU 推理线程
    stop_flag_ = false;
    inference_thread_ = std::thread(&NeuralNet::inference_loop, this);
}

NeuralNet::~NeuralNet() {
    // 通知推理线程停止
    stop_flag_ = true;
    queue_cv_.notify_all();

    // 等待推理线程退出
    if (inference_thread_.joinable()) {
        inference_thread_.join();
    }

    // 销毁 LibTorch 模块
    auto* module = static_cast<torch::jit::script::Module*>(module_impl_);
    delete module;
}

// =====================================================================
// 公开 API: predict
// =====================================================================

std::future<NeuralNetResult> NeuralNet::predict(
    const std::vector<float>& encoded_board)
{
    // 校验输入维度：3 通道 × 15 行 × 15 列 = 675
    if (encoded_board.size() != 675) {
        auto promise = std::promise<NeuralNetResult>();
        promise.set_exception(std::make_exception_ptr(
            std::runtime_error(
                "NeuralNet::predict: encoded_board 大小必须为 675, 实际 "
                + std::to_string(encoded_board.size()))));
        return promise.get_future();
    }

    InferenceRequest request;
    request.encoded_board = encoded_board;

    auto future = request.promise.get_future();

    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        pending_requests_.push_back(std::move(request));
    }

    // 唤醒推理线程（若有足够请求或超时后也会自动处理）
    queue_cv_.notify_one();

    return future;
}

// =====================================================================
// 推理线程主循环
// =====================================================================

void NeuralNet::inference_loop() {
    while (!stop_flag_) {
        std::vector<InferenceRequest> batch;

        {
            std::unique_lock<std::mutex> lock(queue_mutex_);

            // 阻塞等待：有请求到达 或 停止标志
            queue_cv_.wait_for(lock, std::chrono::milliseconds(10),
                [this]() {
                    return stop_flag_.load()
                           || pending_requests_.size() >= static_cast<size_t>(batch_size_);
                });

            if (stop_flag_ && pending_requests_.empty()) {
                break;
            }

            if (pending_requests_.empty()) {
                continue;
            }

            // 取出一批请求（最多 batch_size_ 个）
            size_t take_count = std::min(pending_requests_.size(),
                                         static_cast<size_t>(batch_size_));
            auto start = pending_requests_.begin();
            auto end = start + static_cast<ptrdiff_t>(take_count);
            batch.assign(
                std::make_move_iterator(start),
                std::make_move_iterator(end));
            pending_requests_.erase(start, end);
        }

        if (batch.empty()) continue;

        // 收集所有输入
        std::vector<std::vector<float>> inputs;
        inputs.reserve(batch.size());
        for (auto& req : batch) {
            inputs.push_back(std::move(req.encoded_board));
        }

        // 批量前向推理
        std::vector<NeuralNetResult> results;
        try {
            results = forward_batch(inputs);
        } catch (const std::exception& e) {
            // 推理失败：为每个请求设置异常
            for (auto& req : batch) {
                req.promise.set_exception(std::current_exception());
            }
            continue;
        }

        // 分发结果到各个 promise
        for (size_t i = 0; i < batch.size(); ++i) {
            batch[i].promise.set_value(std::move(results[i]));
        }
    }
}

// =====================================================================
// 批量前向推理
// =====================================================================

std::vector<NeuralNetResult> NeuralNet::forward_batch(
    const std::vector<std::vector<float>>& batch)
{
    auto* module = static_cast<torch::jit::script::Module*>(module_impl_);
    torch::NoGradGuard no_grad;

    // 获取模块所在设备
    torch::Device device = torch::kCPU;
    for (const auto& param : module->parameters()) {
        device = param.device();
        break;
    }

    size_t batch_size = batch.size();

    // 构造输入张量 (batch, 3, 15, 15)
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(device);
    auto input_tensor = torch::zeros(
        {static_cast<long>(batch_size), 3, 15, 15}, options);

    for (size_t i = 0; i < batch_size; ++i) {
        const auto& encoded = batch[i];
        // encoded 是 channel-major: [0..224]=ch0, [225..449]=ch1, [450..674]=ch2
        float* data = input_tensor[i].data_ptr<float>();
        for (int ch = 0; ch < 3; ++ch) {
            for (int row = 0; row < 15; ++row) {
                for (int col = 0; col < 15; ++col) {
                    int idx = ch * 225 + row * 15 + col;
                    int tensor_idx = ch * 225 + row * 15 + col;
                    data[tensor_idx] = encoded[idx];
                }
            }
        }
    }

    // 前向推理
    auto output = module->forward({input_tensor}).toTuple();
    torch::Tensor policy_log_probs = output->elements()[0].toTensor();  // (batch, 225)
    torch::Tensor value_tensor = output->elements()[1].toTensor();       // (batch, 1)

    // 转换到 CPU（若在 GPU 上）
    policy_log_probs = policy_log_probs.to(torch::kCPU);
    value_tensor = value_tensor.to(torch::kCPU);

    // 提取结果
    std::vector<NeuralNetResult> results;
    results.reserve(batch_size);

    for (size_t i = 0; i < batch_size; ++i) {
        NeuralNetResult result;

        // 策略：LogSoftmax → exp → softmax 概率
        auto policy_row = policy_log_probs[i];  // shape (225)
        auto policy_exp = torch::exp(policy_row);
        auto policy_sum = policy_exp.sum();
        // 归一化确保和为 1（应对数值精度）
        if (policy_sum.item<float>() > 0) {
            policy_exp = policy_exp / policy_sum;
        }

        result.policy.resize(225);
        float* policy_data = policy_exp.data_ptr<float>();
        for (int j = 0; j < 225; ++j) {
            result.policy[j] = policy_data[j];
        }

        // 价值：Tanh 输出，值域 [-1, 1]
        result.value = value_tensor[i].item<float>();

        results.push_back(std::move(result));
    }

    return results;
}
