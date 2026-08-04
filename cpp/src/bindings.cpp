/** pybind11 绑定定义
 *
 * 将 C++ MctsSearcher 暴露为 Python 模块 cpp_mcts。
 * 模块初始化时从 mcts_config.json 加载运行时配置。
 *
 * MctsResult 自动转换为 Python tuple: (visit_counts, root_value, root_policy)
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

#include <Python.h>

#include "game_runner.h"
#include "mcts_config.h"
#include "mcts_searcher.h"
#include "thread_pool.h"

namespace py = pybind11;

/** 将 MctsResult 转换为 Python tuple
 *
 * @param result C++ 搜索结果
 * @return Python tuple (visit_counts[225], root_value, root_policy[225])
 */
static py::tuple result_to_tuple(const MctsResult& result) {
    return py::make_tuple(result.visit_counts, result.root_value,
                          result.root_policy);
}

/** 从 mcts_config.json 加载配置到 g_mcts_config
 *
 * JSON 文件位于 cpp_mcts.pyd 同目录下。
 * 若文件不存在或解析失败，静默使用 MctsConfig 结构体默认值。
 */
static void load_mcts_config(py::module_& m) {
    try {
        // 获取 .pyd 文件路径 → 同目录下找 mcts_config.json
        py::module_ os = py::module_::import("os");
        PyObject* filename_obj = PyModule_GetFilenameObject(m.ptr());
        if (!filename_obj) return;

        py::str pyd_path = py::reinterpret_steal<py::str>(filename_obj);
        py::str config_path = os.attr("path").attr("join")(
            os.attr("path").attr("dirname")(pyd_path),
            "mcts_config.json"
        );

        // 读取并解析 JSON
        py::module_ json_mod = py::module_::import("json");
        py::object config_file = py::module_::import("builtins")
            .attr("open")(config_path, "r", py::arg("encoding") = "utf-8");
        py::dict d = json_mod.attr("load")(config_file);

        // 填充全局配置（json.get(key, default)）
        auto get_int = [&](const char* key, int fallback) {
            return d.attr("get")(key, fallback).cast<int>();
        };
        auto get_double = [&](const char* key, double fallback) {
            return d.attr("get")(key, fallback).cast<double>();
        };
        auto get_bool = [&](const char* key, bool fallback) {
            return d.attr("get")(key, fallback).cast<bool>();
        };

        g_mcts_config.num_simulations   = get_int("num_simulations", 800);
        g_mcts_config.num_threads       = get_int("num_threads", 32);
        g_mcts_config.c_puct            = get_double("c_puct", 2.0);
        g_mcts_config.dirichlet_epsilon = get_double("dirichlet_epsilon", 0.25);
        g_mcts_config.dirichlet_alpha   = get_double("dirichlet_alpha", 0.03);
        g_mcts_config.batch_lower_bound = get_int("batch_lower_bound", 16);
        g_mcts_config.batch_upper_bound = get_int("batch_upper_bound", 500);
        g_mcts_config.virtual_loss      = get_double("virtual_loss", 3.0);
        g_mcts_config.use_gpu           = get_bool("use_gpu", true);

        py::print("[cpp_mcts] 配置已从", config_path, "加载");
    } catch (py::error_already_set&) {
        PyErr_Clear();
        py::print("[cpp_mcts] 配置文件未找到，使用内置默认值");
    }
}

PYBIND11_MODULE(cpp_mcts, m) {
    m.doc() = "C++ MCTS 搜索器核心模块 — 多线程 + GPU 异步批量推理";

    // ---- 加载运行时配置（必须在绑定定义之前，因为绑定使用配置值）----
    load_mcts_config(m);

    // MctsResult 结构体（仅用于文档，实际 search 返回 tuple）
    py::class_<MctsResult>(m, "MctsResult",
        "MCTS 搜索结果（内部使用，search() 返回 tuple）")
        .def_readonly("visit_counts", &MctsResult::visit_counts,
            "225 维访问计数增量")
        .def_readonly("root_value", &MctsResult::root_value,
            "根节点平均价值 Q")
        .def_readonly("root_policy", &MctsResult::root_policy,
            "根节点原始策略（狄利克雷噪声施加前）");

    // ---- GpuInferenceStats 监控统计 ----
    py::class_<GpuInferenceStats>(m, "GpuInferenceStats",
        "GPU 推理线程运行时监控统计")
        .def_readonly("total_batches", &GpuInferenceStats::total_batches,
            "已处理的批次数")
        .def_readonly("total_requests", &GpuInferenceStats::total_requests,
            "已处理的请求总数")
        .def_readonly("total_wait_us", &GpuInferenceStats::total_wait_us,
            "等待攒批总耗时（微秒）")
        .def_readonly("total_infer_us", &GpuInferenceStats::total_infer_us,
            "GPU 推理总耗时（微秒）")
        .def_readonly("queue_peak_size", &GpuInferenceStats::queue_peak_size,
            "队列长度历史峰值")
        .def_readonly("batch_size_min", &GpuInferenceStats::batch_size_min,
            "历史最小批次大小")
        .def_readonly("batch_size_max", &GpuInferenceStats::batch_size_max,
            "历史最大批次大小")
        .def("wait_ratio", &GpuInferenceStats::wait_ratio,
            "GPU 线程等待比例 = wait/(wait+infer)，0~1")
        .def("avg_batch_size", &GpuInferenceStats::avg_batch_size,
            "历史平均批次大小");

    // MctsSearcher 主类
    py::class_<MctsSearcher>(m, "MctsSearcher",
        "多线程 MCTS 搜索器\n\n"
        "封装搜索树管理、Virtual Loss 协调、狄利克雷噪声、子树复用。\n"
        "对应 Python 端 MCTSSearcher。")
        .def(py::init<
                std::string, int, double, double, double,
                int, double, bool>(),
             py::arg("model_path"),
             py::arg("num_simulations")   = g_mcts_config.num_simulations,
             py::arg("c_puct")            = g_mcts_config.c_puct,
             py::arg("dirichlet_epsilon") = g_mcts_config.dirichlet_epsilon,
             py::arg("dirichlet_alpha")   = g_mcts_config.dirichlet_alpha,
             py::arg("num_threads")       = g_mcts_config.num_threads,
             py::arg("virtual_loss")      = g_mcts_config.virtual_loss,
             py::arg("use_gpu")           = g_mcts_config.use_gpu,
             "Multi-threaded MCTS Searcher with GPU batch inference")
        .def("search", [](MctsSearcher& self, const std::vector<int>& state) {
                return result_to_tuple(self.search(state));
             },
             py::arg("state"),
             "执行 MCTS 搜索\n\n"
             ":param state: 226 维 int 列表（225 board + 1 player）\n"
             ":return: tuple (visit_counts[225], root_value, root_policy[225])")
        .def("advance_root", &MctsSearcher::advance_root,
             py::arg("action"),
             "将 action 对应子节点提升为新根\n\n"
             ":param action: 动作索引 0-224\n"
             ":return: True 如果子节点存在并成功提升");

    // ---- 全局线程池管理 ----
    m.def("init_thread_pool", &MctsThreadPool::initialize,
          py::arg("num_threads") = g_mcts_config.num_threads,
          "初始化全局 MCTS 线程池（进程内只创建一次）。"
          "必须在首次搜索前调用，否则自动按默认线程数创建。");

    m.def("shutdown_thread_pool", &MctsThreadPool::shutdown,
          "关闭全局线程池，等待所有工作线程退出。");

    m.def("thread_pool_size", [] { return MctsThreadPool::instance().size(); },
          "返回线程池中的工作线程数。");

    // ---- GPU 推理监控 ----
    m.def("get_gpu_stats", []() -> GpuInferenceStats {
        return NeuralNet::instance().get_stats();
    }, "获取 GPU 推理线程的运行时统计快照。");

    // ---- GameConfig 配置结构体 ----
    py::class_<GameConfig>(m, "GameConfig",
        "MCTS 对战配置参数")
        .def(py::init<>())
        .def_readwrite("num_simulations", &GameConfig::num_simulations,
            "每步 MCTS 模拟次数")
        .def_readwrite("c_puct", &GameConfig::c_puct,
            "PUCT 探索系数")
        .def_readwrite("dirichlet_epsilon", &GameConfig::dirichlet_epsilon,
            "狄利克雷噪声混合比例")
        .def_readwrite("dirichlet_alpha", &GameConfig::dirichlet_alpha,
            "狄利克雷分布浓度参数")
        .def_readwrite("virtual_loss", &GameConfig::virtual_loss,
            "Virtual loss 值")
        .def_readwrite("temperature", &GameConfig::temperature,
            "温度参数（τ）")
        .def_readwrite("temp_steps", &GameConfig::temp_steps,
            "温度探索步数")
        .def_readwrite("use_gpu", &GameConfig::use_gpu,
            "是否使用 GPU 推理");

    // ---- GameResult 结果结构体 ----
    py::class_<GameResult>(m, "GameResult",
        "单局对战结果")
        .def_readonly("game_index", &GameResult::game_index,
            "局编号 0..N-1")
        .def_readonly("winner", &GameResult::winner,
            "0=平局, 1=黑胜, 2=白胜")
        .def_readonly("model1_color", &GameResult::model1_color,
            "model1 执方 1=黑, 2=白")
        .def_readonly("model2_color", &GameResult::model2_color,
            "model2 执方")
        .def_readonly("move_count", &GameResult::move_count,
            "本局总步数");

    // ---- GameRunner 对局管理器 ----
    py::class_<GameRunner>(m, "GameRunner",
        "C++ 并行对局管理器\n\n"
        "管理 N 个游戏线程，共享全局 MCTS 线程池和 GPU 推理引擎。\n"
        "通过 CppMctsBattleAdapter 包装后使用。")
        .def(py::init<const std::string&, const std::string&, int, const GameConfig&, int>(),
             py::arg("model1_path"),
             py::arg("model2_path"),
             py::arg("num_games"),
             py::arg("config"),
             py::arg("num_workers") = 0,
             "创建对局管理器\n\n"
             ":param model1_path: 模型1 TorchScript 路径\n"
             ":param model2_path: 模型2 TorchScript 路径\n"
             ":param num_games: 对战局数\n"
             ":param config: GameConfig 配置\n"
             ":param num_workers: 游戏线程数（0=每局一线程，>0=限定线程数）")
        .def("start", &GameRunner::start,
             "启动所有游戏线程（非阻塞）")
        .def("get_completed_count", &GameRunner::get_completed_count,
             "返回已完成局数")
        .def("get_total_count", &GameRunner::get_total_count,
             "返回总局数")
        .def("is_finished", &GameRunner::is_finished,
             "所有对局是否已完成")
        .def("wait_completion", &GameRunner::wait_completion,
             "阻塞直到所有对局完成")
        .def("get_results", &GameRunner::get_results,
             "获取所有对局结果列表");
}
