"""IterativeTrainer 日志输出与最终统计报告测试

从 test_iterative_train.py 拆分出的子文件，覆盖：
- UT-6：日志输出格式验证
- UT-7：最终统计报告内容和格式
"""

import pytest

from tests.ai.unit.test_iterative_train import trainer  # noqa: F401 - fixture 供测试使用


# ============================================================
# UT-6：日志输出 (§2.6)
# ============================================================

class TestLogOutput:
    """UT-6：日志输出格式验证。"""

    def test_round_start_header(self, trainer, capsys):
        """UT-6.1：每轮开始时打印轮次标识。"""
        trainer._round_num = 1
        trainer._print_round_header()
        captured = capsys.readouterr()
        assert "第 1 轨迭代开始" in captured.out

    def test_round_result_promoted(self, trainer, capsys):
        """UT-6.2：每轮结束时打印结果摘要（意图 12：始终显示'晋升'）。"""
        trainer._round_num = 1
        trainer._print_round_result(win_rate=55.0)
        captured = capsys.readouterr()
        assert "第 1 轨结果: cnn_v4.100 胜率 = 55.0% → 晋升" in captured.out

    def test_round_result_stopped(self, trainer, capsys):
        """UT-6.3：每轮结束时打印结果摘要（意图 12：始终显示'晋升'）。"""
        trainer._round_num = 1
        trainer._print_round_result(win_rate=48.0)
        captured = capsys.readouterr()
        assert "第 1 轨结果: cnn_v4.100 胜率 = 48.0% → 晋升" in captured.out

    def test_detail_line_promoted(self, trainer, capsys):
        """UT-6.4：最终统计逐轮明细行格式（意图 12：不再显示晋升/停止标识）。"""
        trainer._rounds = [{
            "round_num": 1,
            "gen4_wins": 38,
            "gen5_wins": 55,
            "draws": 7,
            "gen5_win_rate": 55.0,
            "promoted": False,
        }]
        trainer._print_final_report()
        captured = capsys.readouterr()
        expected = (
            "第1轮: cnn_v4.99 胜 38 | cnn_v4.100 胜 55 | "
            "平局 7 | cnn_v4.100 胜率 55.0%"
        )
        assert expected in captured.out

    def test_detail_line_stopped(self, trainer, capsys):
        """UT-6.5：最终统计逐轮明细行格式（意图 12：不再显示停止标识）。"""
        trainer._rounds = [{
            "round_num": 3,
            "gen4_wins": 44,
            "gen5_wins": 47,
            "draws": 9,
            "gen5_win_rate": 47.0,
            "promoted": False,
        }]
        trainer._print_final_report()
        captured = capsys.readouterr()
        expected = (
            "第3轮: cnn_v4.99 胜 44 | cnn_v4.100 胜 47 | "
            "平局 9 | cnn_v4.100 胜率 47.0%"
        )
        assert expected in captured.out


# ============================================================
# UT-7：最终统计报告 (§2.7)
# ============================================================

class TestFinalReport:
    """UT-7：最终统计报告内容和格式。"""

    def _add_rounds(self, trainer, rounds_data):
        """辅助：填充 trainer._rounds 数据。"""
        trainer._rounds = rounds_data
        trainer._round_num = len(rounds_data)

    def test_report_total_iterations(self, trainer, capsys):
        """UT-7.1：报告含总迭代次数（意图 12：所有轮次都计入迭代次数）。"""
        self._add_rounds(trainer, [
            {"round_num": 1, "gen4_wins": 38, "gen5_wins": 55, "draws": 7,
             "gen5_win_rate": 55.0, "promoted": False},
            {"round_num": 2, "gen4_wins": 40, "gen5_wins": 52, "draws": 8,
             "gen5_win_rate": 52.0, "promoted": False},
            {"round_num": 3, "gen4_wins": 44, "gen5_wins": 47, "draws": 9,
             "gen5_win_rate": 47.0, "promoted": False},
        ])
        trainer._print_final_report()
        captured = capsys.readouterr()
        assert "总迭代次数: 3" in captured.out

    def test_report_round_details_all(self, trainer, capsys):
        """UT-7.2：报告含逐轮明细表（意图 12：3行均无晋升/停止标识）。"""
        self._add_rounds(trainer, [
            {"round_num": 1, "gen4_wins": 38, "gen5_wins": 55, "draws": 7,
             "gen5_win_rate": 55.0, "promoted": False},
            {"round_num": 2, "gen4_wins": 40, "gen5_wins": 52, "draws": 8,
             "gen5_win_rate": 52.0, "promoted": False},
            {"round_num": 3, "gen4_wins": 44, "gen5_wins": 47, "draws": 9,
             "gen5_win_rate": 47.0, "promoted": False},
        ])
        trainer._print_final_report()
        captured = capsys.readouterr()

        # 验证3行逐轮明细
        assert "第1轮:" in captured.out
        assert "第2轮:" in captured.out
        assert "第3轮:" in captured.out
        # 验证不应含有晋升/停止标识
        assert "→ 晋升" not in captured.out
        assert "→ 停止" not in captured.out
        # 验证含胜场和平局
        assert "cnn_v4.99 胜" in captured.out
        assert "cnn_v4.100 胜" in captured.out
        assert "平局" in captured.out

    def test_report_stop_reason(self, trainer, capsys):
        """UT-7.3：报告含停止原因（意图 12：显示用户中断）。"""
        self._add_rounds(trainer, [
            {"round_num": 1, "gen4_wins": 38, "gen5_wins": 55, "draws": 7,
             "gen5_win_rate": 55.0, "promoted": False},
            {"round_num": 2, "gen4_wins": 40, "gen5_wins": 52, "draws": 8,
             "gen5_win_rate": 52.0, "promoted": False},
            {"round_num": 3, "gen4_wins": 44, "gen5_wins": 47, "draws": 9,
             "gen5_win_rate": 47.0, "promoted": False},
        ])
        trainer._print_final_report()
        captured = capsys.readouterr()
        assert "停止原因" in captured.out
        assert "第3轮" in captured.out
        assert "47.0%" in captured.out
        assert "用户中断" in captured.out

    def test_report_final_model(self, trainer, capsys):
        """UT-7.4：报告含当前模型（意图 12：显示'当前模型'及进化轮次）。"""
        self._add_rounds(trainer, [
            {"round_num": 1, "gen4_wins": 44, "gen5_wins": 47, "draws": 9,
             "gen5_win_rate": 47.0, "promoted": True},
        ])
        trainer._print_final_report()
        captured = capsys.readouterr()
        assert "当前模型" in captured.out
        assert "gen99" in captured.out
        assert "cnn_v4.pth" in captured.out
        assert "经 1 轮进化" in captured.out

    def test_report_first_round_stop(self, trainer, capsys):
        """UT-7.5：首轮停止时统计正确（意图 12：所有轮次都计入迭代次数，停止原因为用户中断）。

        总迭代次数 1（不再减掉停止轮），逐轮明细 1 行。
        """
        self._add_rounds(trainer, [
            {"round_num": 1, "gen4_wins": 45, "gen5_wins": 48, "draws": 7,
             "gen5_win_rate": 48.0, "promoted": False},
        ])
        trainer._print_final_report()
        captured = capsys.readouterr()
        assert "总迭代次数: 1" in captured.out
        assert "第1轮:" in captured.out
        assert "停止原因" in captured.out
