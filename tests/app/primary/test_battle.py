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


class TestBattleSessionRun:
    """BattleSession.run() 主流程测试"""

    def setup_method(self):
        _clean_test_root()

    def test_run_returns_results_with_correct_length(self):
        """run() 应返回长度等于 num_games 的结果列表"""
        session = BattleSession("random", "random", num_games=2, project_root=_TEST_ROOT)
        results = session.run()
        assert len(results) == 2

    def test_run_result_has_game_and_winner(self):
        """run() 返回的每项结果应包含 game 和 winner 字段"""
        session = BattleSession("random", "random", num_games=1, project_root=_TEST_ROOT)
        results = session.run()
        assert "game" in results[0]
        assert "winner" in results[0]

    def test_run_winner_is_valid_value(self):
        """winner 应为 1（黑胜）、2（白胜）或 3（平局）"""
        session = BattleSession("random", "random", num_games=1, project_root=_TEST_ROOT)
        results = session.run()
        assert results[0]["winner"] in (1, 2, 3)

    def test_run_calls_on_progress(self):
        """run() 每完成一局应调用 on_progress 一次"""
        calls = []

        def on_progress(current, total):
            calls.append((current, total))

        session = BattleSession("random", "random", num_games=3, project_root=_TEST_ROOT)
        session.run(on_progress=on_progress)

        assert len(calls) == 3
        assert calls[0] == (1, 3)
        assert calls[1] == (2, 3)
        assert calls[2] == (3, 3)

    def test_run_generates_train_samples(self):
        """run() 完成后应在 train_data 目录下生成 sample.json"""
        session = BattleSession("random", "random", num_games=1, project_root=_TEST_ROOT)
        session.run()

        train_data_dir = os.path.join(_TEST_ROOT, "train_data")
        sample_files = []
        for root, dirs, files in os.walk(train_data_dir):
            for f in files:
                if f == "sample.json":
                    sample_files.append(os.path.join(root, f))

        assert len(sample_files) > 0, "应至少生成一个 sample.json"

    def test_get_recorder_returns_recorder_after_run(self):
        """run() 后 get_recorder() 应返回有效的 DataRecorder"""
        from ai import DataRecorder
        session = BattleSession("random", "random", num_games=1, project_root=_TEST_ROOT)
        session.run()
        recorder = session.get_recorder()
        assert recorder is not None
        assert isinstance(recorder, DataRecorder)

    def test_get_recorder_returns_none_before_run(self):
        """run() 前 get_recorder() 应返回 None"""
        session = BattleSession("random", "random", num_games=1, project_root=_TEST_ROOT)
        assert session.get_recorder() is None
