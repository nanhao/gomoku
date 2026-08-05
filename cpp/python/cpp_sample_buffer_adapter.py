"""C++ MCTS 自对弈样本缓冲区适配器模块

提供 CppMctsSampleBufferAdapter 类，将 C++ GameRunner（自对弈模式）包装为
Python 友好接口。负责从 registry spec 解析模型路径、创建 C++ GameRunner、
进度轮询、样本缓冲管理和落盘。

使用示例:
    from cpp.python.cpp_sample_buffer_adapter import CppMctsSampleBufferAdapter
    from ai.scripts.mcts_config import MctsSelfPlayConfig

    config = MctsSelfPlayConfig(simulations=800)
    adapter = CppMctsSampleBufferAdapter(
        "cnn_v11.1", 1000, config,
        num_workers=4, buffer_dir="sample_buffer",
    )
    adapter.start()
    while not adapter.is_finished():
        completed, total = adapter.get_progress()
        stats = adapter.get_sample_stats()
        print(f"[进度] {completed}/{total} | 缓冲区: {stats}")
"""

import os
import time
from typing import Dict, List, Tuple

from ai.scripts.mcts_config import MctsSelfPlayConfig


class CppMctsSampleBufferAdapter:
    """C++ MCTS 自对弈样本缓冲区适配器

    将 C++ GameRunner（自对弈模式）包装为 Python 友好的接口，负责：
    - 从 registry spec 解析模型路径并查找 TorchScript 文件
    - 创建 C++ GameRunner 实例（自对弈模式，collect_samples=true）
    - 创建 SampleBufferManager 实例
    - 进度轮询、样本缓冲管理和落盘
    """

    def __init__(
        self,
        model_spec: str,
        num_games: int,
        config: MctsSelfPlayConfig,
        num_workers: int = 0,
        buffer_dir: str = "sample_buffer",
        buffer_max_size: int = 80000,
        chunk_games: int = 10,
    ):
        """初始化适配器

        构造时完成：
        - 从 registry spec 解析 TorchScript 模型路径
        - 创建 C++ GameRunner（自对弈模式，collect_samples=true）
        - 创建 SampleBufferManager

        Args:
            model_spec: registry spec，如 "cnn_v11.1"
            num_games: 自对弈总局数（≥1）
            config: MCTS 超参数配置
            num_workers: 游戏线程数（0=每局一线程，>0=限定线程数）
            buffer_dir: 样本缓冲区目录
            buffer_max_size: 缓冲区最大样本数
            chunk_games: 每 chunk 包含的对局数

        Raises:
            FileNotFoundError: TorchScript 模型文件不存在
            ValueError: registry spec 无效
        """
        # 先解析模型路径（验证 spec 和 TorchScript 文件存在性）
        model_pt = self._resolve_torchscript_path(model_spec)

        print(f"[CppMctsSampleBufferAdapter] model={model_spec} → {model_pt}")

        # 将 torch\lib 加入 PATH，以便 LibTorch 运行时动态加载 CUDA DLL
        _torch_lib_dir = os.path.join(os.path.dirname(os.path.dirname(
            __import__('torch').__file__)), 'lib')
        if _torch_lib_dir not in os.environ.get('PATH', ''):
            os.environ['PATH'] = _torch_lib_dir + os.pathsep + os.environ.get('PATH', '')

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
        cpp_config.use_gpu = True

        # 创建 C++ GameRunner（自对弈模式，开启样本采集）
        self._runner = cpp_mcts.GameRunner(
            model_pt, num_games, cpp_config, num_workers, True  # collect_samples=True
        )
        self._num_games = num_games

        # 创建 SampleBufferManager
        from ai.sample_buffer_manager import SampleBufferManager
        self._buffer = SampleBufferManager(
            buffer_dir=buffer_dir,
            max_size=buffer_max_size,
            chunk_games=chunk_games,
        )
        self._chunk_games = chunk_games
        self._next_game_start = 1  # 下一批 chunk 的起始局号（1-based）

        # 累计统计
        self._evicted_total = 0
        self._samples_total = 0

    # ---------- 静态工具方法 ----------

    @staticmethod
    def _resolve_torchscript_path(spec: str) -> str:
        """从 registry spec 解析 TorchScript 模型文件路径

        使用 registry 的 _parse_model_spec + _import_class（不实例化模型，避免
        依赖 .pth 文件），从类属性 DEFAULT_MODEL_PATH 推导 TorchScript 路径。

        与 CppMctsBattleAdapter._resolve_torchscript_path 共用相同逻辑。

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

    # ---------- 公共 API ----------

    def start(self) -> None:
        """启动所有 C++ 游戏线程（非阻塞，立即返回）

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

    def get_sample_stats(self) -> dict:
        """返回实时样本统计

        Returns:
            {"completed_games": N, "total_games": N,
             "buffer_samples": N, "buffer_chunks": N,
             "evicted_samples": N}
        """
        completed, total = self.get_progress()
        return {
            "completed_games": completed,
            "total_games": total,
            "buffer_samples": self._buffer.get_total_samples(),
            "buffer_chunks": self._buffer.get_chunk_count(),
            "evicted_samples": self._evicted_total,
            "total_produced": self._samples_total,
        }

    def is_finished(self) -> bool:
        """所有对局是否已完成"""
        return self._runner.is_finished()

    def wait_completion(self) -> None:
        """阻塞直到所有对局完成、所有样本落盘

        内部 join 所有 C++ 游戏线程，并将最后一批样本落盘。
        支持在 Ctrl+C 后调用以等待进行中的对局。
        """
        self._runner.wait_completion()
        # 将最后剩余样本落盘
        self._flush_remaining()

    def get_buffer_summary(self) -> dict:
        """返回缓冲区最终摘要（同 SampleBufferManager.get_summary()）"""
        return self._buffer.get_summary()

    def poll_and_ingest(self) -> int:
        """轮询 C++ 进度，若新完成 >= chunk_games 局则落盘

        每检测到 chunk_games 局新完成即触发 buffer.ingest()。

        Returns:
            新落盘的样本数（0 表示本次无新落盘）
        """
        completed, _ = self.get_progress()
        new_ingested = 0

        # 当已完成局数足以形成新的 chunk 时
        while completed >= self._next_game_start + self._chunk_games - 1:
            game_start = self._next_game_start
            game_end = game_start + self._chunk_games - 1

            # 获取已完成的结果
            try:
                results = self._runner.get_results()
            except Exception:
                break

            # 提取 game_start..game_end 范围内的样本
            all_samples: list = []
            for r in results:
                gi = r.game_index + 1  # 转为 1-based
                if game_start <= gi <= game_end and r.samples:
                    # 将 C++ GameSample 转为 Python dict
                    for s in r.samples:
                        all_samples.append({
                            "state": list(s.state),
                            "pi": list(s.pi),
                            "z": float(s.z),
                        })

            if all_samples:
                evicted = self._buffer.ingest(game_start, game_end, all_samples)
                self._evicted_total += evicted
                self._samples_total += len(all_samples)
                new_ingested += len(all_samples)

            self._next_game_start = game_end + 1

        return new_ingested

    def _flush_remaining(self) -> None:
        """将最后一个不足 chunk_games 局的对局样本落盘"""
        completed, _ = self.get_progress()
        if completed < self._next_game_start:
            return  # 无双新对局

        # 获取结果
        try:
            results = self._runner.get_results()
        except Exception:
            return

        # 提取剩余对局的样本
        all_samples: list = []
        for r in results:
            gi = r.game_index + 1  # 转为 1-based
            if gi >= self._next_game_start and r.samples:
                for s in r.samples:
                    all_samples.append({
                        "state": list(s.state),
                        "pi": list(s.pi),
                        "z": float(s.z),
                    })

        if all_samples and completed >= self._next_game_start:
            evicted = self._buffer.ingest(
                self._next_game_start, completed, all_samples
            )
            self._evicted_total += evicted
            self._samples_total += len(all_samples)
            self._next_game_start = completed + 1
