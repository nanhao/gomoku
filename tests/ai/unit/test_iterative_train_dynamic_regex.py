"""IterativeTrainer 参数化单元测试（意图 13）— 动态正则解析

测试对战输出解析中的动态正则构造逻辑。
对应设计文档 test_design.md §2.4。
"""

import re

import pytest


# ============================================================
# 辅助函数：构造对战输出文本
# ============================================================

def _battle_stdout_for(ai1: str, ai2: str,
                       ai1_wins=40, ai2_wins=55, draws=5):
    """构造 generic 对战输出文本，使用动态模型名。

    Args:
        ai1: 父代模型 spec，如 "cnn_v4.99"
        ai2: 子代模型 spec，如 "cnn_v4.100"
        ai1_wins: 父代胜场数
        ai2_wins: 子代胜场数
        draws: 平局数
    """
    total = ai1_wins + ai2_wins + draws
    ai1_rate = ai1_wins / total * 100 if total > 0 else 0.0
    ai2_rate = ai2_wins / total * 100 if total > 0 else 0.0
    draw_rate = draws / total * 100 if total > 0 else 0.0

    lines = [
        f"开始 AI 对战: {ai1} vs {ai2}, 共 100 局",
        "",
    ]
    for i in range(1, total + 1):
        lines.append(f"第 {i} 局: {ai1}(黑) vs {ai2}(白) -> 黑方胜")
    lines += [
        "",
        "=== 对战结果 ===",
        f"总局数: {total}",
        f"{ai1} 胜: {ai1_wins} 局 ({ai1_rate:.1f}%)",
        f"{ai2} 胜: {ai2_wins} 局 ({ai2_rate:.1f}%)",
        f"平局: {draws} 局 ({draw_rate:.1f}%)",
    ]
    return "\n".join(lines)


# ============================================================
# 辅助函数：动态正则解析（与 _parse_battle_output 中的逻辑一致）
# ============================================================

def _parse_battle_output_dynamic(stdout: str, ai1: str, ai2: str) -> dict:
    """使用动态正则解析对战输出。

    与 main_design.md §2.6 动态正则为一致实现：
    使用 re.escape(ai2) / re.escape(ai1) 构造正则。
    """
    # 解析子代胜场和胜率
    gen5_match = re.search(
        rf'{re.escape(ai2)} 胜:\s*(\d+)\s*局\s*\(([\d.]+)%\)', stdout
    )
    if not gen5_match:
        raise ValueError(f"无法解析 {ai2} 胜率，原始输出:\n{stdout}")

    gen5_wins = int(gen5_match.group(1))
    gen5_win_rate = float(gen5_match.group(2))

    # 解析父代胜场（可为 0）
    gen4_match = re.search(
        rf'{re.escape(ai1)} 胜:\s*(\d+)\s*局', stdout
    )
    gen4_wins = int(gen4_match.group(1)) if gen4_match else 0

    # 解析平局数（可为 0）
    draw_match = re.search(r'平局:\s*(\d+)\s*局', stdout)
    draws = int(draw_match.group(1)) if draw_match else 0

    return {
        "ai1_wins": gen4_wins,
        "ai2_wins": gen5_wins,
        "draws": draws,
        "ai2_win_rate": gen5_win_rate,
    }


# ============================================================
# UT-4：对战输出解析（动态正则）（test_design.md §2.4）
# ============================================================

class TestDynamicRegexParsing:
    """UT-4：动态正则对战输出解析测试。

    使用 _parse_battle_output_dynamic 模拟
    IterativeTrainer._parse_battle_output 的动态正则逻辑。
    """

    def test_parse_cnn_v4_battle(self):
        """UT-4.1：解析 cnn_v4.99 vs cnn_v4.100 对战输出。"""
        ai1 = "cnn_v4.99"
        ai2 = "cnn_v4.100"
        stdout = _battle_stdout_for(ai1, ai2, ai1_wins=40, ai2_wins=55, draws=5)
        result = _parse_battle_output_dynamic(stdout, ai1, ai2)
        assert result["ai2_wins"] == 55
        assert result["ai2_win_rate"] == 55.0
        assert result["ai1_wins"] == 40
        assert result["draws"] == 5

    def test_parse_model_name_with_dots(self):
        """UT-4.2：model_name 中含点号时正则不误匹配。

        测试 spec 为 "cnn_v4.2.99" / "cnn_v4.2.100" 时，
        re.escape 正确处理点号转义。
        """
        ai1 = "cnn_v4.2.99"
        ai2 = "cnn_v4.2.100"
        stdout = _battle_stdout_for(ai1, ai2, ai1_wins=35, ai2_wins=55, draws=10)
        result = _parse_battle_output_dynamic(stdout, ai1, ai2)
        assert result["ai2_wins"] == 55
        assert result["ai2_win_rate"] == 55.0

    def test_parse_with_draws(self):
        """UT-4.3：解析含平局输出的对战结果。"""
        ai1 = "cnn_v4.99"
        ai2 = "cnn_v4.100"
        stdout = _battle_stdout_for(ai1, ai2, ai1_wins=30, ai2_wins=30, draws=40)
        result = _parse_battle_output_dynamic(stdout, ai1, ai2)
        assert result["draws"] == 40

    def test_parse_with_parent_wins(self):
        """UT-4.4：解析含父代胜场输出的对战结果。"""
        ai1 = "cnn_v4.99"
        ai2 = "cnn_v4.100"
        stdout = _battle_stdout_for(ai1, ai2, ai1_wins=60, ai2_wins=30, draws=10)
        result = _parse_battle_output_dynamic(stdout, ai1, ai2)
        assert result["ai1_wins"] == 60

    def test_parse_different_model_specs(self):
        """解析不同模型 spec（cnn_v3.1 vs cnn_v3.2）。"""
        ai1 = "cnn_v3.1"
        ai2 = "cnn_v3.2"
        stdout = _battle_stdout_for(ai1, ai2, ai1_wins=40, ai2_wins=50, draws=10)
        result = _parse_battle_output_dynamic(stdout, ai1, ai2)
        assert result["ai2_wins"] == 50

    def test_parse_fc_v1(self):
        """解析 fc_v1.1 vs fc_v1.2 对战输出。"""
        ai1 = "fc_v1.1"
        ai2 = "fc_v1.2"
        stdout = _battle_stdout_for(ai1, ai2, ai1_wins=45, ai2_wins=48, draws=7)
        result = _parse_battle_output_dynamic(stdout, ai1, ai2)
        assert result["ai2_win_rate"] == 48.0

    def test_parse_no_match_raises(self):
        """无匹配行时抛 ValueError。"""
        ai1 = "cnn_v4.99"
        ai2 = "cnn_v4.100"
        stdout = "cnn_v4.100 胜: 无效格式"
        with pytest.raises(ValueError, match="胜率"):
            _parse_battle_output_dynamic(stdout, ai1, ai2)

    def test_parse_100_percent(self):
        """胜率 100% 解析。"""
        ai1 = "cnn_v4.99"
        ai2 = "cnn_v4.100"
        stdout = _battle_stdout_for(ai1, ai2, ai1_wins=0, ai2_wins=100, draws=0)
        result = _parse_battle_output_dynamic(stdout, ai1, ai2)
        assert result["ai2_win_rate"] == 100.0

    def test_parse_0_percent(self):
        """胜率 0% 解析。"""
        ai1 = "cnn_v4.99"
        ai2 = "cnn_v4.100"
        stdout = _battle_stdout_for(ai1, ai2, ai1_wins=100, ai2_wins=0, draws=0)
        result = _parse_battle_output_dynamic(stdout, ai1, ai2)
        assert result["ai2_win_rate"] == 0.0

    def test_re_escape_handles_special_chars(self):
        """验证 re.escape 正确处理模型名中的特殊字符。

        构造一个含 regex 特殊字符的模拟模型名来验证。
        """
        ai1 = "test+v1.1"
        ai2 = "test+v1.2"
        escaped = re.escape(ai1)
        # re.escape 应转义 + 号
        assert "\\+" in escaped
        # 用 re.escape 构造的正则应能正确匹配
        stdout = _battle_stdout_for(ai1, ai2, ai1_wins=30, ai2_wins=60, draws=10)
        result = _parse_battle_output_dynamic(stdout, ai1, ai2)
        assert result["ai2_wins"] == 60
