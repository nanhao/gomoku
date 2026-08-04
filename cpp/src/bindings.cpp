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

#include "mcts_config.h"
#include "mcts_searcher.h"
#include "thread_pool.h"

namespace py = pybind11;

/// 全局运行时配置实例（定义在此处，mcts_config.h 中为 extern 声明）
MctsConfig g_mcts_config;

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
        g_mcts_config.batch_size        = get_int("batch_size", 16);
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

    // MctsSearcher 主类
    py::class_<MctsSearcher>(m, "MctsSearcher",
        "多线程 MCTS 搜索器\n\n"
        "封装搜索树管理、Virtual Loss 协调、狄利克雷噪声、子树复用。\n"
        "对应 Python 端 MCTSSearcher。")
        .def(py::init<
                std::string, int, double, double, double,
                int, int, double, bool>(),
             py::arg("model_path"),
             py::arg("num_simulations")   = g_mcts_config.num_simulations,
             py::arg("c_puct")            = g_mcts_config.c_puct,
             py::arg("dirichlet_epsilon") = g_mcts_config.dirichlet_epsilon,
             py::arg("dirichlet_alpha")   = g_mcts_config.dirichlet_alpha,
             py::arg("num_threads")       = g_mcts_config.num_threads,
             py::arg("batch_size")        = g_mcts_config.batch_size,
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
}
