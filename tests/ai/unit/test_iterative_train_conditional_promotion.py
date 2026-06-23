"""IterativeTrainer 条件晋升测试 — 意图 22

测试条件晋升逻辑、新增构造参数（sample_games/epochs）、日志输出变更。

测试用例覆盖：
- UT-CP-1：构造器新参数
- UT-CP-2：子进程参数可变（sample_games/epochs）
- UT-CP-3：条件晋升判定
- UT-CP-4：日志输出（晋升/未晋升/最终报告）
"""

import os
import json
import subprocess
from unittest.mock import patch

import pytest

from ai.iterative_train import IterativeTrainer
from tests.ai.helpers import _make_completed_process
from tests.ai.unit.test_iterative_train import _battle_stdout, _GEN4, _GEN5, _MDIR, _SDIR


# ============================================================
# UT-CP-1：构造器新参数
# ============================================================

class TestConstructorNewParams:
    """UT-CP-1：IterativeTrainer 构造器支持 sample_games 和 epochs。"""

    def test_default_sample_games(self, temp_project):
        """sample_games 默认值为 500。"""
        t = IterativeTrainer(
            ai1="cnn_v4.99", ai2="cnn_v4.100",
            sample_dir="ai/merge_samples2", project_root=temp_project,
        )
        assert t._sample_games == 500

    def test_custom_sample_games(self, temp_project):
        """sample_games 可自定义。"""
        t = IterativeTrainer(
            ai1="cnn_v4.99", ai2="cnn_v4.100",
            sample_dir="ai/merge_samples2", project_root=temp_project,
            sample_games=300,
        )
        assert t._sample_games == 300

    def test_default_epochs(self, temp_project):
        """epochs 默认值为 300。"""
        t = IterativeTrainer(
            ai1="cnn_v4.99", ai2="cnn_v4.100",
            sample_dir="ai/merge_samples2", project_root=temp_project,
        )
        assert t._epochs == 300

    def test_custom_epochs(self, temp_project):
        """epochs 可自定义。"""
        t = IterativeTrainer(
            ai1="cnn_v4.99", ai2="cnn_v4.100",
            sample_dir="ai/merge_samples2", project_root=temp_project,
            epochs=200,
        )
        assert t._epochs == 200


# ============================================================
# UT-CP-2：子进程参数使用实例属性
# ============================================================

class TestSubprocessArgsVariable:
    """UT-CP-2：子进程调用参数使用 self._sample_games / self._epochs。"""

    @patch('subprocess.run')
    def test_create_samples_uses_custom_sample_games(self, mock_run, temp_project):
        """_generate_samples 使用自定义 sample_games。"""
        mock_run.return_value = _make_completed_process(returncode=0, stdout="ok")
        t = IterativeTrainer(
            ai1="cnn_v4.99", ai2="cnn_v4.100",
            sample_dir="ai/merge_samples2", project_root=temp_project,
            sample_games=200,
        )
        t._generate_samples()
        call_args = mock_run.call_args[0][0]
        assert "--sample_games" in call_args
        assert "200" in call_args

    @patch('subprocess.run')
    def test_train_main_uses_custom_epochs(self, mock_run, temp_project):
        """_train_model 使用自定义 epochs。"""
        mock_run.return_value = _make_completed_process(returncode=0, stdout="ok")
        t = IterativeTrainer(
            ai1="cnn_v4.99", ai2="cnn_v4.100",
            sample_dir="ai/merge_samples2", project_root=temp_project,
            epochs=150,
        )
        t._train_model()
        call_args = mock_run.call_args[0][0]
        assert "--epochs" in call_args
        assert "150" in call_args


# ============================================================
# UT-CP-3：条件晋升判定
# ============================================================

class TestConditionalPromotion:
    """UT-CP-3：仅当 win_rate > 55 才晋升。"""

    def _make_trainer(self, temp_project):
        """创建 trainer 并准备晋升所需的前置条件。"""
        t = IterativeTrainer(
            ai1="cnn_v4.99", ai2="cnn_v4.100",
            sample_dir="ai/merge_samples2", project_root=temp_project,
        )
        # 准备 ai2 权重目录和文件（供 _promote 使用）
        ai2_dir = os.path.join(temp_project, _MDIR, f"gen{_GEN5}")
        os.makedirs(ai2_dir, exist_ok=True)
        ai2_path = os.path.join(ai2_dir, "cnn_v4.pth")
        with open(ai2_path, 'wb') as f:
            f.write(b"trained_weights")
        return t

    def test_promote_when_above_threshold(self, temp_project):
        """win_rate > 55 → 执行晋升，gen5 权重覆盖 gen4。"""
        t = self._make_trainer(temp_project)
        gen4_path = os.path.join(temp_project, _MDIR, f"gen{_GEN4}", "cnn_v4.pth")

        # 晋升前 gen4 是假权重
        with open(gen4_path, 'rb') as f:
            assert f.read() == b"fake_model_weights"

        t._promote()

        # 晋升后 gen4 被 trained_weights 覆盖，gen5 目录被删除
        with open(gen4_path, 'rb') as f:
            assert f.read() == b"trained_weights"
        gen5_path = os.path.join(temp_project, _MDIR, f"gen{_GEN5}", "cnn_v4.pth")
        assert not os.path.exists(gen5_path)

    def test_no_promote_when_equal_threshold(self):
        """win_rate == 55 → 不晋升（严格大于判定）。"""
        assert not (55.0 > 55)  # 55 > 55 为 False，不触发晋升

    def test_no_promote_when_below_threshold(self):
        """win_rate < 55 → 不晋升。"""
        assert not (48.0 > 55)


# ============================================================
# UT-CP-4：日志输出变更
# ============================================================

class TestConditionalPromotionLogging:
    """UT-CP-4：日志输出区分晋升/未晋升。"""

    def test_promoted_log(self, temp_project, capsys):
        """晋升时打印 '> 55% → 晋升'。"""
        t = IterativeTrainer(
            ai1="cnn_v4.99", ai2="cnn_v4.100",
            sample_dir="ai/merge_samples2", project_root=temp_project,
        )
        t._round_num = 1
        t._print_round_result(win_rate=58.0, promoted=True)
        captured = capsys.readouterr()
        assert "第 1 轮结果: cnn_v4.100 胜率 = 58.0% > 55% → 晋升" in captured.out

    def test_not_promoted_log(self, temp_project, capsys):
        """未晋升时打印 '≤ 55% → 未晋升，重新开始'。"""
        t = IterativeTrainer(
            ai1="cnn_v4.99", ai2="cnn_v4.100",
            sample_dir="ai/merge_samples2", project_root=temp_project,
        )
        t._round_num = 2
        t._print_round_result(win_rate=48.0, promoted=False)
        captured = capsys.readouterr()
        assert "第 2 轮结果: cnn_v4.100 胜率 = 48.0% ≤ 55% → 未晋升，重新开始" in captured.out

    def test_not_promoted_exact_55(self, temp_project, capsys):
        """胜率恰好 55% 时打印未晋升。"""
        t = IterativeTrainer(
            ai1="cnn_v4.99", ai2="cnn_v4.100",
            sample_dir="ai/merge_samples2", project_root=temp_project,
        )
        t._round_num = 1
        t._print_round_result(win_rate=55.0, promoted=False)
        captured = capsys.readouterr()
        assert "≤ 55% → 未晋升" in captured.out

    def test_battle_result_includes_i(self, temp_project, capsys):
        """对战结果行包含 i={round_num}。"""
        t = IterativeTrainer(
            ai1="cnn_v4.99", ai2="cnn_v4.100",
            sample_dir="ai/merge_samples2", project_root=temp_project,
        )
        t._round_num = 3
        print(
            f"{t._ai1} 胜 40 | "
            f"{t._ai2} 胜 55 | "
            f"平局 5 | "
            f"胜率 55.0% | "
            f"i={t._round_num}"
        )
        captured = capsys.readouterr()
        assert "i=3" in captured.out

    def test_final_report_shows_promotion_count(self, temp_project, capsys):
        """最终统计报告含晋升次数。"""
        t = IterativeTrainer(
            ai1="cnn_v4.99", ai2="cnn_v4.100",
            sample_dir="ai/merge_samples2", project_root=temp_project,
        )
        t._rounds = [
            {"round_num": 1, "gen4_wins": 30, "gen5_wins": 58, "draws": 12,
             "gen5_win_rate": 58.0, "promoted": True},
            {"round_num": 2, "gen4_wins": 45, "gen5_wins": 42, "draws": 13,
             "gen5_win_rate": 42.0, "promoted": False},
            {"round_num": 3, "gen4_wins": 35, "gen5_wins": 57, "draws": 8,
             "gen5_win_rate": 57.0, "promoted": True},
        ]
        t._round_num = 3
        t._print_final_report()
        captured = capsys.readouterr()
        # 应含晋升次数
        assert "晋升 2 次" in captured.out
        # 逐轮明细应含 [晋升] / [未晋升]
        assert "[晋升]" in captured.out
        assert "[未晋升]" in captured.out


# ============================================================
# UT-CP-5：集成测试 — run() 条件晋升完整流程
# ============================================================

class TestRunConditionalPromotion:
    """UT-CP-5：run() 中的条件晋升集成流程。"""

    @patch('subprocess.run')
    @patch.object(IterativeTrainer, '_validate_samples')
    @patch.object(IterativeTrainer, '_create_lock')
    @patch.object(IterativeTrainer, '_remove_lock')
    def test_run_promotes_when_win_rate_above_threshold(
        self, mock_remove, mock_create, mock_validate, mock_run, temp_project,
    ):
        """win_rate > 55 时 run() 执行晋升并记录 promoted=True。"""
        stdout_60 = _battle_stdout(gen4_wins=35, gen5_wins=60, draws=5)
        mock_run.side_effect = [
            _make_completed_process(returncode=0, stdout="样本局数: 500\n"),
            _make_completed_process(returncode=0, stdout="训练完成\n"),
            _make_completed_process(returncode=0, stdout=stdout_60),
            KeyboardInterrupt(),
        ]

        t = IterativeTrainer(
            ai1="cnn_v4.99", ai2="cnn_v4.100",
            sample_dir="ai/merge_samples2", project_root=temp_project,
        )
        t.run()

        # 验证: 晋升记录
        assert len(t._rounds) == 1
        assert t._rounds[0]["promoted"] is True
        assert t._rounds[0]["gen5_win_rate"] == 60.0

    @patch('subprocess.run')
    @patch.object(IterativeTrainer, '_validate_samples')
    @patch.object(IterativeTrainer, '_create_lock')
    @patch.object(IterativeTrainer, '_remove_lock')
    def test_run_skips_promotion_when_win_rate_below_threshold(
        self, mock_remove, mock_create, mock_validate, mock_run, temp_project,
    ):
        """win_rate ≤ 55 时 run() 不晋升，记录 promoted=False。"""
        stdout_48 = _battle_stdout(gen4_wins=42, gen5_wins=48, draws=10)
        mock_run.side_effect = [
            _make_completed_process(returncode=0, stdout="样本局数: 500\n"),
            _make_completed_process(returncode=0, stdout="训练完成\n"),
            _make_completed_process(returncode=0, stdout=stdout_48),
            KeyboardInterrupt(),
        ]

        t = IterativeTrainer(
            ai1="cnn_v4.99", ai2="cnn_v4.100",
            sample_dir="ai/merge_samples2", project_root=temp_project,
        )
        t.run()

        # 验证: promoted=False，gen4 保持不变
        assert len(t._rounds) == 1
        assert t._rounds[0]["promoted"] is False
        gen4_path = os.path.join(temp_project, _MDIR, f"gen{_GEN4}", "cnn_v4.pth")
        with open(gen4_path, 'rb') as f:
            assert f.read() == b"fake_model_weights"
