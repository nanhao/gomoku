"""C++ MCTS 对战适配器模块

提供 CppMctsBattleAdapter 类，将 C++ GameRunner 包装为 Python 友好接口。
负责从 registry spec 解析模型路径、创建 C++ GameRunner、进度查询和结果获取。

使用示例:
    from cpp.python.cpp_mcts_battle_adapter import CppMctsBattleAdapter
    from ai.scripts.mcts_config import MctsSelfPlayConfig

    config = MctsSelfPlayConfig(simulations=800)
    adapter = CppMctsBattleAdapter("cnn_v11.1", "cnn_v11.2", 100, config)
    adapter.start()
    while not adapter.is_finished():
        completed, total = adapter.get_progress()
        print(f"[进度] {completed}/{total}")
    results = adapter.get_results()
"""

import os
from typing import List, Tuple

from ai.scripts.mcts_config import MctsSelfPlayConfig
from cpp.python.cpp_battle_game_result import CppBattleGameResult


class CppMctsBattleAdapter:
    """C++ MCTS 对战适配器

    将 C++ GameRunner 包装为 Python 友好的接口，负责：
    - 从 registry spec 解析模型路径并查找 TorchScript 文件
    - 创建 C++ GameRunner 实例
    - 提供进度查询和结果获取
    """

    def __init__(
        self,
        model1_spec: str,
        model2_spec: str,
        num_games: int,
        config: MctsSelfPlayConfig,
        num_workers: int = 0,
    ):
        """初始化适配器

        构造时完成：通过 registry 解析两个模型 spec → 查找 TorchScript 模型路径 →
        构建 GameConfig 并创建 C++ GameRunner。

        Args:
            model1_spec: registry spec，如 "cnn_v11.1"
            model2_spec: registry spec，如 "cnn_v11.2"
            num_games: 对战局数（≥1）
            config: MCTS 自对弈/对战超参数配置
            num_workers: 游戏线程数（0=每局一个线程，>0=限定线程数）

        Raises:
            FileNotFoundError: TorchScript 模型文件不存在
            ValueError: registry spec 无效
            RuntimeError: C++ GameRunner 创建失败（如模型加载失败）
        """
        # 先解析路径（验证 spec 和 TorchScript 文件存在性），
        # 成功后才导入 cpp_mcts，确保错误能正确抛出而非被 ImportError 掩盖
        model1_pt = self._resolve_torchscript_path(model1_spec)
        model2_pt = self._resolve_torchscript_path(model2_spec)

        print(f"[CppMctsBattleAdapter] model1={model1_spec} → {model1_pt}")
        print(f"[CppMctsBattleAdapter] model2={model2_spec} → {model2_pt}")

        import cpp_mcts  # type: ignore[import-not-found]

        # 构建 C++ GameConfig（从 Python MctsSelfPlayConfig 映射）
        cpp_config = cpp_mcts.GameConfig()
        cpp_config.num_simulations = config.simulations
        cpp_config.c_puct = config.c_puct
        cpp_config.dirichlet_epsilon = config.dirichlet_epsilon
        cpp_config.dirichlet_alpha = config.dirichlet_alpha
        cpp_config.virtual_loss = config.virtual_loss
        cpp_config.temperature = config.temperature
        cpp_config.temp_steps = config.temp_steps
        # C++ 对战默认使用 GPU（若可用）
        cpp_config.use_gpu = True

        # 创建 C++ GameRunner（内部初始化全局线程池和 NeuralNet 单例）
        self._runner = cpp_mcts.GameRunner(
            model1_pt, model2_pt, num_games, cpp_config, num_workers
        )
        self._num_games = num_games

    # ---------- 静态工具方法 ----------

    @staticmethod
    def _resolve_torchscript_path(spec: str) -> str:
        """从 registry spec 解析 TorchScript 模型文件路径

        使用 registry 的 _parse_model_spec + _import_class（不实例化模型，避免
        依赖 .pth 文件），从类属性 DEFAULT_MODEL_PATH 推导 TorchScript 路径。

        Args:
            spec: registry spec，如 "cnn_v11.1"

        Returns:
            TorchScript 模型文件绝对路径

        Raises:
            FileNotFoundError: TorchScript 文件不存在或模型非神经网络类型
            ValueError: registry spec 无效
        """
        from ai.registry import (
            MODEL_REGISTRY, _parse_model_spec, _is_multi_gen_model, _import_class,
        )

        parsed = _parse_model_spec(spec)
        model_name = parsed.name
        gen = parsed.gen

        # 1. 校验模型名是否在注册表中
        if model_name not in MODEL_REGISTRY:
            raise ValueError(f"未知的模型: '{model_name}'")

        # 2. 校验 gen 参数合法性
        if gen is not None and not _is_multi_gen_model(model_name):
            raise ValueError("非多代模型不能指定代数")
        if gen is None and _is_multi_gen_model(model_name):
            raise ValueError(f"多代模型必须指定代数，如 {model_name}.1")

        # 3. 导入模型类（不实例化，避免触发 .pth 文件存在性检查）
        model_class_path = MODEL_REGISTRY[model_name].model_class_path
        model_cls = _import_class(model_class_path)

        # 4. 检查模型类是否支持 TorchScript
        if not hasattr(model_cls, "DEFAULT_MODEL_PATH") or model_cls.DEFAULT_MODEL_PATH is None:
            raise FileNotFoundError(
                f"模型 {spec} 不支持 TorchScript 导出（非神经网络模型）\n"
                f"  TorchScript 对战仅支持神经网络模型"
            )

        # 5. 从类属性 DEFAULT_MODEL_PATH 推导目录
        default_path = model_cls.DEFAULT_MODEL_PATH
        model_dir = os.path.dirname(default_path)

        # 6. 如果指定了 gen，调整到 gen{i}/ 子目录
        if gen is not None:
            model_dir = os.path.join(model_dir, f"gen{gen}")

        # 7. 构造 TorchScript 文件名（使用注册表名称，即模型类 name 属性值）
        ts_filename = f"{model_name}_torchscript.pt"
        ts_path = os.path.join(model_dir, ts_filename)

        if not os.path.isfile(ts_path):
            raise FileNotFoundError(
                f"TorchScript 模型文件不存在: {ts_path}\n"
                f"  spec: {spec}\n"
                f"  请先导出 TorchScript 模型后再运行对战"
            )

        return os.path.abspath(ts_path)

    @staticmethod
    def _convert_result(cpp_result) -> CppBattleGameResult:
        """将 C++ GameResult 转换为 Python CppBattleGameResult

        在 Python 层计算 model1_won / model2_won，避免 C++ 侧冗余。

        Args:
            cpp_result: C++ GameResult 对象（来自 cpp_mcts.GameRunner.get_results()）

        Returns:
            CppBattleGameResult 数据类实例
        """
        winner = cpp_result.winner
        m1_color = cpp_result.model1_color
        m2_color = cpp_result.model2_color

        # 根据 winner 和执方计算各方是否获胜
        model1_won = (winner == m1_color) and winner != 0
        model2_won = (winner == m2_color) and winner != 0

        return CppBattleGameResult(
            game_index=cpp_result.game_index,
            winner=winner,
            model1_color=m1_color,
            model2_color=m2_color,
            model1_won=model1_won,
            model2_won=model2_won,
            move_count=cpp_result.move_count,
        )

    # ---------- 公共 API ----------

    def start(self) -> None:
        """启动所有游戏线程（非阻塞，立即返回）

        C++ 侧创建 N 个 std::thread，每个线程运行一局完整对局。
        调用后立即返回，对局在后台并行执行。
        """
        self._runner.start()

    def get_progress(self) -> Tuple[int, int]:
        """返回 (已完成局数, 总局数)（线程安全）

        Returns:
            (completed, total) 元组
        """
        return (self._runner.get_completed_count(), self._runner.get_total_count())

    def is_finished(self) -> bool:
        """所有对局是否已完成

        Returns:
            True 若所有游戏线程已结束
        """
        return self._runner.is_finished()

    def wait_completion(self) -> None:
        """阻塞直到所有对局完成

        内部 join 所有游戏线程。支持在 Ctrl+C 后调用以等待进行中的对局。
        """
        self._runner.wait_completion()

    def get_results(self) -> List[CppBattleGameResult]:
        """获取所有对局结果

        仅在 is_finished() 为 True 后调用。返回结果列表按 game_index 升序。

        Returns:
            CppBattleGameResult 列表，长度 == num_games
        """
        cpp_results = self._runner.get_results()
        results: List[CppBattleGameResult] = []
        for r in cpp_results:
            results.append(self._convert_result(r))
        return results
