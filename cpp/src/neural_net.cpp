#include "neural_net.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <future>
#include <iostream>
#include <mutex>
#include <queue>
#include <shared_mutex>
#include <stdexcept>
#include <string>
#include <thread>

#include "mcts_config.h"

// LibTorch headers
#include <torch/script.h>
#include <torch/torch.h>

// =====================================================================
// 单例管理
// =====================================================================

static std::unique_ptr<NeuralNet> s_nn_instance;
static std::once_flag s_nn_init_flag;

NeuralNet& NeuralNet::instance() {
    std::call_once(s_nn_init_flag, [] {
        // 使用默认参数自动初始化
        s_nn_instance.reset(new NeuralNet(
            "src/ai/models/cnn_v11/cnn_v11_torchscript.pt", true));
    });
    return *s_nn_instance;
}

void NeuralNet::initialize(const std::string& model_path, bool use_gpu) {
    std::call_once(s_nn_init_flag, [&] {
        s_nn_instance.reset(new NeuralNet(model_path, use_gpu));
    });
}

void NeuralNet::shutdown() {
    s_nn_instance.reset();
}

// =====================================================================
// 内部: load_model_to_cache（不加锁，由调用方保证线程安全）
// =====================================================================

void NeuralNet::load_model_to_cache(const std::string& model_path) {
    // 始终加载到 CPU（避免主线程 CUDA 初始化与 worker 线程冲突）
    // GPU 迁移延迟到 forward_batch 首次调用时，在 worker 线程上执行
    torch::Device device(torch::kCPU);

    // 加载 TorchScript 模型
    torch::jit::script::Module module = torch::jit::load(model_path);
    module.to(device);
    module.eval();

    // 存入缓存
    CachedModule cached;
    cached.module_impl = new torch::jit::script::Module(std::move(module));
    cached.is_on_gpu = false;
    module_cache_[model_path] = cached;
}

// =====================================================================
// 构造 & 析构
// =====================================================================

NeuralNet::NeuralNet(std::string model_path, bool use_gpu)
    : batch_lower_bound_(g_mcts_config.batch_lower_bound > 0
                         ? g_mcts_config.batch_lower_bound : 16)
    , batch_upper_bound_(g_mcts_config.batch_upper_bound > 0
                         ? g_mcts_config.batch_upper_bound : 16)
    , use_gpu_(use_gpu)
{
    std::cout << "[NeuralNet] use_gpu=" << use_gpu_
              << " cuda_avail=" << torch::cuda::is_available()
              << " batch_range=[" << batch_lower_bound_
              << "," << batch_upper_bound_ << "]"
              << std::endl;

    // 预加载首个模型到缓存
    try {
        load_model_to_cache(model_path);
    } catch (const c10::Error& e) {
        throw std::runtime_error(
            "NeuralNet: 加载 TorchScript 模型失败: " + model_path
            + " — " + std::string(e.what()));
    }

    std::cout << "[NeuralNet] preloaded: " << model_path << std::endl;

    // 启动推理线程（GPU 初始化在线程内部自行完成）
    stop_flag_ = false;
    inference_thread_ = std::thread(&NeuralNet::inference_loop, this);
}

NeuralNet::~NeuralNet() {
    // 停止推理线程
    stop_flag_ = true;
    queue_cv_.notify_all();
    if (inference_thread_.joinable()) {
        inference_thread_.join();
    }

    // 销毁所有缓存的 LibTorch 模块
    for (auto& entry : module_cache_) {
        auto* module = static_cast<torch::jit::script::Module*>(
            entry.second.module_impl);
        delete module;
    }
    module_cache_.clear();
}

// =====================================================================
// 公开 API: preload_model
// =====================================================================

void NeuralNet::preload_model(const std::string& model_path) {
    // 快速路径：已缓存则跳过（不加写锁）
    {
        std::shared_lock<std::shared_mutex> lock(model_mutex_);
        if (module_cache_.count(model_path)) {
            return;
        }
    }

    // 独占写锁（阻塞推理线程的下一 batch）
    std::unique_lock<std::shared_mutex> lock(model_mutex_);

    // 双重检查：可能在等待写锁期间被其他线程加载
    if (module_cache_.count(model_path)) {
        return;
    }

    // 加载新模型到 CPU（推理线程下一 batch 自动检测并迁移到 GPU）
    try {
        load_model_to_cache(model_path);
    } catch (const c10::Error& e) {
        throw std::runtime_error(
            "NeuralNet::preload_model: 加载模型失败: " + model_path
            + " — " + std::string(e.what()));
    }

    std::cout << "[NeuralNet] preloaded: " << model_path << std::endl;
}

// =====================================================================
// 公开 API: predict（提交到推理线程队列，阻塞等待结果）
// =====================================================================

NeuralNetResult NeuralNet::predict(
    const std::vector<float>& encoded_board,
    const std::string& model_id)
{
    if (encoded_board.size() != 675) {
        throw std::runtime_error(
            "NeuralNet::predict: encoded_board 大小必须为 675, 实际 "
            + std::to_string(encoded_board.size()));
    }

    // 快速检查 model_id 是否已缓存（不加写锁）
    {
        std::shared_lock<std::shared_mutex> lock(model_mutex_);
        if (!module_cache_.count(model_id)) {
            throw std::runtime_error(
                "NeuralNet::predict: model_id 未预加载: " + model_id);
        }
    }

    // 创建 promise/future 对
    std::promise<NeuralNetResult> promise;
    auto future = promise.get_future();

    // 提交请求到推理线程队列
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        pending_requests_.push({encoded_board, model_id, std::move(promise)});
        int qsize = static_cast<int>(pending_requests_.size());
        {
            std::lock_guard<std::mutex> slock(stats_mutex_);
            if (qsize > stats_.queue_peak_size) {
                stats_.queue_peak_size = qsize;
            }
        }
    }
    queue_cv_.notify_one();

    // 阻塞等待推理线程返回结果
    return future.get();
}

// =====================================================================
// 推理线程主循环
// =====================================================================

void NeuralNet::inference_loop() {
    while (!stop_flag_) {
        // === 阶段 1: 等待并收集请求 ===
        std::vector<PendingRequest> batch;
        auto t_wait_start = std::chrono::steady_clock::now();

        {
            std::unique_lock<std::mutex> lock(queue_mutex_);

            // 等待第一个请求（最多 500μs 超时，避免空转）
            if (pending_requests_.empty()) {
                queue_cv_.wait_for(lock, std::chrono::microseconds(500));
            }

            if (stop_flag_ && pending_requests_.empty()) break;

            // 收集最多 batch_upper_bound_ 个请求
            while (!pending_requests_.empty()
                   && static_cast<int>(batch.size()) < batch_upper_bound_) {
                batch.push_back(std::move(pending_requests_.front()));
                pending_requests_.pop();
            }
        }

        if (batch.empty()) continue;

        // === 阶段 2: 攒批等待（不足 batch_lower_bound_ 时再等最多 2ms）===
        while (static_cast<int>(batch.size()) < batch_lower_bound_) {
            auto elapsed = std::chrono::steady_clock::now() - t_wait_start;
            if (elapsed > std::chrono::microseconds(2000)) break;

            {
                std::lock_guard<std::mutex> lock(queue_mutex_);
                while (!pending_requests_.empty()
                       && static_cast<int>(batch.size()) < batch_upper_bound_) {
                    batch.push_back(std::move(pending_requests_.front()));
                    pending_requests_.pop();
                }
            }

            if (static_cast<int>(batch.size()) >= batch_lower_bound_) break;
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }

        auto t_infer_start = std::chrono::steady_clock::now();
        int64_t wait_us = std::chrono::duration_cast<std::chrono::microseconds>(
            t_infer_start - t_wait_start).count();

        // === 阶段 3: 按 model_id 分组，每组独立 GPU 批量推理 ===
        std::vector<NeuralNetResult> results(batch.size());
        {
            // 按 model_id 分组
            std::unordered_map<std::string, std::vector<size_t>> groups;
            for (size_t i = 0; i < batch.size(); ++i) {
                groups[batch[i].model_id].push_back(i);
            }

            std::shared_lock<std::shared_mutex> lock(model_mutex_);
            for (auto& kv : groups) {
                const std::string& model_id = kv.first;
                const std::vector<size_t>& indices = kv.second;

                // 确保模型已迁移到 GPU
                auto& cached = module_cache_[model_id];
                if (use_gpu_ && !cached.is_on_gpu
                    && torch::cuda::is_available()) {
                    static_cast<torch::jit::script::Module*>(
                        cached.module_impl)->to(torch::kCUDA);
                    cached.is_on_gpu = true;
                }

                // 收集该组的输入数据
                std::vector<std::vector<float>> group_data;
                group_data.reserve(indices.size());
                for (size_t idx : indices) {
                    group_data.push_back(
                        std::move(batch[idx].encoded_board));
                }

                // 执行推理
                auto group_results = forward_batch(group_data, model_id);
                for (size_t j = 0; j < indices.size(); ++j) {
                    results[indices[j]] = std::move(group_results[j]);
                }
            }
        }

        auto t_end = std::chrono::steady_clock::now();
        int64_t infer_us = std::chrono::duration_cast<std::chrono::microseconds>(
            t_end - t_infer_start).count();

        // === 阶段 5: 返回结果（fulfill promises）===
        for (size_t i = 0; i < batch.size(); ++i) {
            batch[i].promise.set_value(std::move(results[i]));
        }

        // === 阶段 6: 更新统计 ===
        {
            auto t_now = std::chrono::steady_clock::now();
            int bs = static_cast<int>(batch.size());

            std::lock_guard<std::mutex> slock(stats_mutex_);
            // -- 全周期累计 --
            stats_.total_batches++;
            stats_.total_requests += static_cast<int64_t>(bs);
            stats_.total_wait_us += wait_us;
            stats_.total_infer_us += infer_us;
            if (stats_.batch_size_min == 0 || bs < stats_.batch_size_min) {
                stats_.batch_size_min = bs;
            }
            if (bs > stats_.batch_size_max) {
                stats_.batch_size_max = bs;
            }

            // -- 滑动窗口: batch 大小（最近 10 个）--
            recent_batch_sizes_.push_back(bs);
            while (static_cast<int>(recent_batch_sizes_.size()) > kRecentBatchWindow) {
                recent_batch_sizes_.pop_front();
            }

            // -- 滑动窗口: 时间窗口（最近 5 秒的 wait/infer 记录）--
            recent_timings_.emplace_back(t_now, std::make_pair(wait_us, infer_us));
            auto cutoff = t_now - std::chrono::microseconds(kRecentTimeWindowUs);
            while (!recent_timings_.empty() && recent_timings_.front().first < cutoff) {
                recent_timings_.pop_front();
            }

            if (stats_.total_batches % stats_log_interval_ == 0) {
                // 即时计算滑动窗口值用于日志输出
                double recent_avg = 0.0;
                if (!recent_batch_sizes_.empty()) {
                    int64_t sum = 0;
                    for (int sz : recent_batch_sizes_) sum += sz;
                    recent_avg = static_cast<double>(sum) / recent_batch_sizes_.size();
                }
                double recent_wr = 0.0;
                {
                    int64_t rw = 0, ri = 0;
                    for (const auto& entry : recent_timings_) {
                        rw += entry.second.first;
                        ri += entry.second.second;
                    }
                    int64_t total = rw + ri;
                    if (total > 0) recent_wr = static_cast<double>(rw) / total;
                }
                std::printf(
                    "[NeuralNet] batches=%lld reqs=%lld "
                    "avg_batch=%.1f(rec=%.1f) wait_ratio=%.1f%%(rec=%.1f%%) "
                    "queue_peak=%d batch_range=[%d,%d]\n",
                    static_cast<long long>(stats_.total_batches),
                    static_cast<long long>(stats_.total_requests),
                    stats_.avg_batch_size(), recent_avg,
                    stats_.wait_ratio() * 100.0, recent_wr * 100.0,
                    stats_.queue_peak_size,
                    stats_.batch_size_min, stats_.batch_size_max);
            }
        }
    }
}

// =====================================================================
// 运行时监控
// =====================================================================

GpuInferenceStats NeuralNet::get_stats() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    GpuInferenceStats s = stats_;  // 拷贝累计字段

    // ---- 计算滑动窗口: avg_batch_size（最近 10 个 batch）----
    s.recent_batch_count = static_cast<int>(recent_batch_sizes_.size());
    if (s.recent_batch_count > 0) {
        int64_t sum = 0;
        for (int sz : recent_batch_sizes_) sum += sz;
        s.recent_avg_batch_size = static_cast<double>(sum) / s.recent_batch_count;
    }

    // ---- 计算滑动窗口: wait_ratio（最近 5 秒）----
    {
        auto now = std::chrono::steady_clock::now();
        auto cutoff = now - std::chrono::microseconds(kRecentTimeWindowUs);
        int64_t rw = 0, ri = 0;
        for (const auto& entry : recent_timings_) {
            if (entry.first >= cutoff) {
                rw += entry.second.first;
                ri += entry.second.second;
            }
        }
        int64_t total = rw + ri;
        if (total > 0) {
            s.recent_wait_ratio = static_cast<double>(rw) / total;
        }
    }

    return s;
}

// =====================================================================
// 批量前向推理
// =====================================================================

std::vector<NeuralNetResult> NeuralNet::forward_batch(
    const std::vector<std::vector<float>>& batch,
    const std::string& model_id)
{
    // 从缓存获取模块
    auto it = module_cache_.find(model_id);
    if (it == module_cache_.end()) {
        throw std::runtime_error(
            "NeuralNet::forward_batch: model_id 未缓存: " + model_id);
    }
    auto* module = static_cast<torch::jit::script::Module*>(
        it->second.module_impl);
    torch::NoGradGuard no_grad;

    // 获取模块所在设备
    torch::Device device = torch::kCPU;
    for (const auto& param : module->parameters()) {
        device = param.device();
        break;
    }

    // GPU 迁移：若模型在 CPU 且需要 GPU，则迁移
    // 推理线程是唯一调用者，且持有 shared_lock，无需 std::call_once
    if (use_gpu_ && !device.is_cuda() && torch::cuda::is_available()) {
        module->to(torch::kCUDA);
        device = torch::kCUDA;
        std::cout << "[NeuralNet] 模型已迁移到 GPU" << std::endl;
    }

    size_t batch_size = batch.size();

    // 构造输入张量 (batch, 3, 15, 15)
    // 关键：先在 CPU 上创建并填充数据，再搬到目标设备
    // 直接往 GPU tensor 的 data_ptr() 写数据属于 UB，会导致 segfault
    auto cpu_options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    auto input_tensor = torch::zeros(
        {static_cast<long>(batch_size), 3, 15, 15}, cpu_options);

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

    // 数据填充完毕后再搬到目标设备（CPU → GPU 或保持 CPU）
    input_tensor = input_tensor.to(device);

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
