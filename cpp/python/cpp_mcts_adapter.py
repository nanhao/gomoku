"""C++ MCTS 搜索器 Python 适配层

提供 CppMctsAdapter 类，将 C++ cpp_mcts.MctsSearcher 包装为与现有 Python
MCTSSearcher 兼容的接口，供 PolicyValueAiEngine 直接使用。

使用示例:
    from cpp.python.cpp_mcts_adapter import CppMctsAdapter
    from ai.engine.policy_value import PolicyValueAiEngine

    searcher = CppMctsAdapter(
        model_path="src/ai/models/cnn_v11/cnn_v11_torchscript.pt",
        num_simulations=200,
        num_threads=64,
        batch_size=32,
        use_gpu=True,
    )
    engine = PolicyValueAiEngine(searcher, temperature=1.0, temp_steps=8)
    move = engine.get_move(board, current_player, strategy="sampling")
    engine.after_move(move.x, move.y)
"""

from typing import List, Optional, Tuple


class CppMctsAdapter:
    """C++ MCTS 搜索器适配器

    将 cpp_mcts.MctsSearcher 包装为与 Python MCTSSearcher 兼容的接口。
    log_key 参数在 C++ 端暂不支持日志，传入后会被忽略。
    """

    def __init__(
        self,
        model_path: str,
        num_simulations: int = 200,
        c_puct: float = 2.0,
        dirichlet_epsilon: float = 0.25,
        dirichlet_alpha: float = 0.03,
        num_threads: int = 4,
        batch_size: int = 16,
        virtual_loss: float = 3.0,
        use_gpu: bool = True,
    ):
        """创建 C++ MCTS 搜索器适配器

        Args:
            model_path: TorchScript 模型文件路径 (.pt)
            num_simulations: 每次搜索的总模拟次数，默认 200
            c_puct: PUCT 探索系数，默认 2.0
            dirichlet_epsilon: 狄利克雷噪声混合比例，默认 0.25
            dirichlet_alpha: 狄利克雷分布浓度参数，默认 0.03
            num_threads: MCTS 工作线程数，默认 4
            batch_size: GPU 推理攒批大小，默认 16
            virtual_loss: 虚拟损失值，默认 3.0
            use_gpu: 是否使用 GPU 推理，默认 True
        """
        import cpp_mcts  # type: ignore[import-not-found]

        self._cpp_searcher = cpp_mcts.MctsSearcher(
            model_path,
            num_simulations=num_simulations,
            c_puct=c_puct,
            dirichlet_epsilon=dirichlet_epsilon,
            dirichlet_alpha=dirichlet_alpha,
            num_threads=num_threads,
            batch_size=batch_size,
            virtual_loss=virtual_loss,
            use_gpu=use_gpu,
        )

    def search(
        self,
        state: List[int],
        num_simulations: Optional[int] = None,
        log_key: Optional[str] = None,
    ) -> Tuple[List[int], float, List[float]]:
        """执行 MCTS 搜索（委托 C++ 实现）

        Args:
            state: 226 维 int 列表（225 board + 1 current_player）
            num_simulations: 忽略（C++ 端在构造时固定）
            log_key: 忽略（C++ 端暂不支持日志）

        Returns:
            Tuple[List[int], float, List[float]]:
                - visit_counts: 225 维访问计数增量
                - root_value: 根节点平均价值 Q
                - root_policy: 根节点原始策略（狄利克雷噪声施加前）
        """
        return self._cpp_searcher.search(state)

    def advance_root(self, action: int) -> bool:
        """将 action 对应子节点提升为新根

        Args:
            action: 动作索引 0-224

        Returns:
            True 如果子节点存在并成功提升
        """
        return self._cpp_searcher.advance_root(action)


def create_cpp_searcher(model_path: str, **kwargs) -> CppMctsAdapter:
    """工厂函数：根据模型路径创建 C++ MCTS 搜索器

    若模型路径以 .pt 结尾，创建 CppMctsAdapter。
    否则可扩展为回退到 Python 实现。

    Args:
        model_path: 模型文件路径
        **kwargs: 传递给 CppMctsAdapter 的额外参数

    Returns:
        CppMctsAdapter 实例
    """
    if model_path.endswith(".pt"):
        return CppMctsAdapter(model_path, **kwargs)

    raise ValueError(
        f"create_cpp_searcher: 不支持的模型格式: {model_path!r}。"
        f"仅支持 TorchScript (.pt) 模型。"
    )
