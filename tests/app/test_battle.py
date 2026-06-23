"""BattleSession 主流程测试

覆盖设计文档中 BattleSession 的主流程测试用例：
- run() 返回正确的结果列表
- on_progress 回调被正确调用
- get_recorder() 返回有效 DataRecorder
- 对战完成后生成训练样本
"""

import sys
import os
import shutil
import tempfile
from unittest.mock import patch

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from app import BattleSession


# 使用独立的测试数据目录，避免污染 ai_data / train_data
_TEST_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_test_battle_root")


def _clean_test_root():
    """清理测试数据目录"""
    if os.path.exists(_TEST_ROOT):
        shutil.rmtree(_TEST_ROOT)
    os.makedirs(os.path.join(_TEST_ROOT, "ai_data"), exist_ok=True)
    os.makedirs(os.path.join(_TEST_ROOT, "train_data"), exist_ok=True)


class TestBattleSessionSharedRun:
    """BattleSession.run() 简单断言共享一次对弈结果（性能优化）

    以下 5 个断言原本各自执行一次 num_games=1 的 random 对弈（合计 ~5s），
    但断言都只读取 run() 的返回值或目录产物，不相互污染。
    改为 class-scoped 跑一次 run()，5 个断言共享同一结果（~1s）。
    """

    @classmethod
    def setup_class(cls):
        _clean_test_root()
        cls.test_save_dir = tempfile.mkdtemp(prefix="test_battle_save_shared_")
        cls._patcher = patch(
            'ai.game_manager.GomokuController',
            lambda save_dir=None: __import__('game').GomokuController(save_dir=cls.test_save_dir)
        )
        cls._patcher.start()

        cls.ai_data_dir = os.path.join(_TEST_ROOT, "ai_data")
        cls.train_data_dir = os.path.join(_TEST_ROOT, "train_data")
        cls.session = BattleSession(
            "random", "random", num_games=1,
            ai_data_dir=cls.ai_data_dir, train_data_dir=cls.train_data_dir
        )
        cls.results = cls.session.run()

    @classmethod
    def teardown_class(cls):
        cls._patcher.stop()
        shutil.rmtree(cls.test_save_dir, ignore_errors=True)

    def test_run_returns_results_with_correct_length(self):
        """run() 应返回长度等于 num_games 的结果列表"""
        assert len(self.results) == 1

    def test_run_result_has_game_and_winner(self):
        """run() 返回的每项结果应包含 game 和 winner 字段"""
        assert "game" in self.results[0]
        assert "winner" in self.results[0]

    def test_run_winner_is_valid_value(self):
        """winner 应为 1（黑胜）、2（白胜）或 3（平局）"""
        assert self.results[0]["winner"] in (1, 2, 3)

    def test_run_generates_train_samples(self):
        """run() 完成后应在 train_data 目录下生成 sample.json"""
        sample_files = []
        for root, _dirs, files in os.walk(self.train_data_dir):
            for f in files:
                if f == "sample.json":
                    sample_files.append(os.path.join(root, f))

        assert len(sample_files) > 0, "应至少生成一个 sample.json"

    def test_get_recorder_returns_recorder_after_run(self):
        """run() 后 get_recorder() 应返回有效的 DataRecorder"""
        from ai import DataRecorder
        recorder = self.session.get_recorder()
        assert recorder is not None
        assert isinstance(recorder, DataRecorder)


class TestBattleSessionRun:
    """需要独立 setup/teardown 的 BattleSession 用例

    保留 per-method 隔离：
    - test_run_calls_on_progress：num_games=2 且使用自定义 on_progress 回调
    - test_get_recorder_returns_none_before_run：构造 session 但不跑 run
    """

    def setup_method(self):
        _clean_test_root()
        self.test_save_dir = tempfile.mkdtemp(prefix="test_battle_save_")
        # Patch GomokuController 使其使用临时目录，避免写入 game/save/
        self._patcher = patch(
            'ai.game_manager.GomokuController',
            lambda save_dir=None: __import__('game').GomokuController(save_dir=self.test_save_dir)
        )
        self._patcher.start()

    def teardown_method(self):
        self._patcher.stop()
        shutil.rmtree(self.test_save_dir, ignore_errors=True)

    def test_run_calls_on_progress(self):
        """run() 每完成一局应调用 on_progress 一次"""
        calls = []

        def on_progress(current, total):
            calls.append((current, total))

        ai_data_dir = os.path.join(_TEST_ROOT, "ai_data")
        train_data_dir = os.path.join(_TEST_ROOT, "train_data")
        session = BattleSession("random", "random", num_games=2, ai_data_dir=ai_data_dir, train_data_dir=train_data_dir)
        session.run(on_progress=on_progress)

        assert len(calls) == 2
        assert calls[0] == (1, 2)
        assert calls[1] == (2, 2)

    def test_get_recorder_returns_none_before_run(self):
        """run() 前 get_recorder() 应返回 None"""
        ai_data_dir = os.path.join(_TEST_ROOT, "ai_data")
        train_data_dir = os.path.join(_TEST_ROOT, "train_data")
        session = BattleSession("random", "random", num_games=1, ai_data_dir=ai_data_dir, train_data_dir=train_data_dir)
        assert session.get_recorder() is None


# ---------------------------------------------------------------------------
# 第四十一次追加：BattleSession 透传 temp_config 测试
# ---------------------------------------------------------------------------

class TestBattleSessionTempConfig:
    """验证 BattleSession 接受并透传 temp_config 给 AIGameManager。"""

    def setup_method(self):
        _clean_test_root()
        self.test_save_dir = tempfile.mkdtemp(prefix="test_battle_tempcfg_")
        # Patch GomokuController 使其使用临时目录
        self._patcher = patch(
            'ai.game_manager.GomokuController',
            lambda save_dir=None: __import__('game').GomokuController(
                save_dir=self.test_save_dir
            ),
        )
        self._patcher.start()

    def teardown_method(self):
        self._patcher.stop()
        shutil.rmtree(self.test_save_dir, ignore_errors=True)

    def test_temp_config_parameter(self):
        """temp_config 参数：传则存储，不传则为 None。"""
        from ai.move_sampler import TempConfig
        # 传入时应存储
        cfg = TempConfig(temperature=2.0, temp_moves=5, temp_side="black")
        session_with = BattleSession(
            "random", "random", num_games=1,
            ai_data_dir=os.path.join(_TEST_ROOT, "ai_data"),
            train_data_dir=os.path.join(_TEST_ROOT, "train_data"),
            temp_config=cfg,
        )
        assert session_with._temp_config is cfg
        assert session_with._temp_config.temperature == 2.0
        # 不传时应为 None
        session_without = BattleSession(
            "random", "random", num_games=1,
            ai_data_dir=os.path.join(_TEST_ROOT, "ai_data"),
            train_data_dir=os.path.join(_TEST_ROOT, "train_data"),
        )
        assert session_without._temp_config is None

    def test_runs_with_temp_config(self):
        """BattleSession.run() 在有 temp_config 时不报错。"""
        from ai.move_sampler import TempConfig
        cfg = TempConfig(temperature=1.0, temp_moves=12, temp_side="both")
        session = BattleSession(
            "random", "random", num_games=1,
            ai_data_dir=os.path.join(_TEST_ROOT, "ai_data"),
            train_data_dir=os.path.join(_TEST_ROOT, "train_data"),
            temp_config=cfg,
        )
        results = session.run()
        assert len(results) == 1
        assert results[0]["winner"] in (1, 2, 3)
