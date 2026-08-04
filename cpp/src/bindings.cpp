/** pybind11 绑定定义
 *
 * 将 C++ MctsSearcher 暴露为 Python 模块 cpp_mcts。
 *
 * MctsResult 自动转换为 Python tuple: (visit_counts, root_value, root_policy)
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

#include "mcts_searcher.h"

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

PYBIND11_MODULE(cpp_mcts, m) {
    m.doc() = "C++ MCTS 搜索器核心模块 — 多线程 + GPU 异步批量推理";

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
             py::arg("num_simulations") = 200,
             py::arg("c_puct") = 2.0,
             py::arg("dirichlet_epsilon") = 0.25,
             py::arg("dirichlet_alpha") = 0.03,
             py::arg("num_threads") = 4,
             py::arg("batch_size") = 16,
             py::arg("virtual_loss") = 3.0,
             py::arg("use_gpu") = true,
             "创建 MCTS 搜索器\n\n"
             ":param model_path: TorchScript 模型文件路径 (.pt)\n"
             ":param num_simulations: 每次搜索的总模拟次数，默认 200\n"
             ":param c_puct: PUCT 探索系数，默认 2.0\n"
             ":param dirichlet_epsilon: 狄利克雷噪声混合比例，默认 0.25\n"
             ":param dirichlet_alpha: 狄利克雷分布浓度参数，默认 0.03\n"
             ":param num_threads: MCTS 工作线程数，默认 4\n"
             ":param batch_size: GPU 推理攒批大小，默认 16\n"
             ":param virtual_loss: 虚拟损失值，默认 3.0\n"
             ":param use_gpu: 是否使用 GPU 推理，默认 True")
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
}
