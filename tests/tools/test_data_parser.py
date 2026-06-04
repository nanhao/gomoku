"""data_parser 模块测试

测试 scan_round_dir、parse_game_steps、match_samples、infer_temp_side 四个函数，
覆盖设计文档 data_parser_design.md 中的 T1-T11 全部测试点。
"""

import json
import os
import tempfile
import shutil

import pytest

from tools.create_samples_tool.data_parser import (
    scan_round_dir,
    parse_game_steps,
    match_samples,
    infer_temp_side,
)


# ------------------------------------------------------------------
# 辅助：构造测试数据
# ------------------------------------------------------------------

def _make_record(history, winner=1, end_board=None):
    """构造 record.json 内容"""
    if end_board is None:
        end_board = [0] * 226
    return {"history": history, "winner": winner, "end_board": end_board}


def _make_step(step_id=None, input_vec=None, output_vec=None,
               position=None, strategy=None, sampling_probs=None,
               temperature=None):
    """构造单步 history entry"""
    if input_vec is None:
        input_vec = [0] * 225 + [1]  # 226维：225棋盘+1落子方
    if output_vec is None:
        output_vec = [0.01] * 225
        output_vec[7 * 15 + 7] = 0.6  # 落子位概率最高
    if position is None:
        position = [7, 7]
    entry = {
        "input": input_vec,
        "output": output_vec,
        "position": position,
    }
    if step_id is not None:
        entry["step_id"] = step_id
    if strategy is not None:
        entry["strategy"] = strategy
        if strategy == "sampling" and sampling_probs is not None:
            entry["sampling_probs"] = sampling_probs
            entry["temperature"] = temperature or 1.0
    return entry


def _make_sample(step_id=None, input_vec=None, output_vec=None, mask=None):
    """构造样本输出条目"""
    if input_vec is None:
        input_vec = [0] * 225 + [1]
    if output_vec is None:
        output_vec = [0.0] * 225
        output_vec[7 * 15 + 7] = 1.0
    if mask is None:
        mask = [1] * 225
    sample = {
        "input": input_vec,
        "output": output_vec,
        "mask": mask,
    }
    if step_id is not None:
        sample["step_id"] = step_id
    return sample


@pytest.fixture
def round_dir(tmp_path):
    """创建临时 round_dir 目录结构"""
    rd = os.path.join(tmp_path, "test_round")
    os.makedirs(rd)
    return rd


def _write_record(round_dir, game_id, record_data):
    """向 round_dir 中写入指定局号的 record.json"""
    game_dir = os.path.join(round_dir, str(game_id))
    os.makedirs(game_dir, exist_ok=True)
    path = os.path.join(game_dir, "record.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record_data, f)


def _write_samples(tmp_path, samples_data):
    """向临时目录写入样本文件"""
    path = os.path.join(tmp_path, "samples.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(samples_data, f)
    return path


# ------------------------------------------------------------------
# T1: scan_round_dir 正常扫描
# ------------------------------------------------------------------

class TestScanRoundDir:
    """测试 scan_round_dir 函数"""

    def test_normal_scan(self, round_dir):
        """T1: 正常扫描 → 返回正确的 game_id、winner、temp_side"""
        # 创建 3 局数据：黑胜+双方采样、白胜+黑方采样、平局+全argmax
        h1 = [
            _make_step(step_id="sid-1", strategy="sampling",
                       sampling_probs=[0.01]*225, temperature=1.0,
                       input_vec=[0]*225+[1]),  # 黑方
            _make_step(step_id="sid-2", strategy="sampling",
                       sampling_probs=[0.01]*225, temperature=1.0,
                       input_vec=[0]*225+[2]),  # 白方
        ]
        h2 = [
            _make_step(step_id="sid-3", strategy="sampling",
                       sampling_probs=[0.01]*225, temperature=1.0,
                       input_vec=[0]*225+[1]),
            _make_step(step_id="sid-4", strategy="argmax"),
        ]
        h3 = [
            _make_step(strategy="argmax"),
        ]

        _write_record(round_dir, 1, _make_record(h1, winner=1))
        _write_record(round_dir, 2, _make_record(h2, winner=2))
        _write_record(round_dir, 3, _make_record(h3, winner=3))

        games = scan_round_dir(round_dir)
        assert len(games) == 3
        assert games[0]["game_id"] == 1
        assert games[0]["winner"] == 1
        assert games[0]["winner_display"] == "黑胜"
        assert games[0]["is_draw"] is False
        assert games[0]["temp_side"] == "both"
        assert games[0]["temp_side_display"] == "双方采样"
        assert games[0]["total_steps"] == 2

        assert games[1]["game_id"] == 2
        assert games[1]["winner"] == 2
        assert games[1]["winner_display"] == "白胜"
        assert games[1]["temp_side"] == "black"
        assert games[1]["total_steps"] == 2

        assert games[2]["game_id"] == 3
        assert games[2]["winner"] == 3
        assert games[2]["winner_display"] == "平局"
        assert games[2]["is_draw"] is True
        assert games[2]["temp_side"] == "none"
        assert games[2]["temp_side_display"] == "无(argmax)"

    def test_corrupted_json_skipped(self, round_dir):
        """T2: record.json 损坏 → scan_round_dir 跳过不报错"""
        _write_record(round_dir, 1, _make_record(
            [_make_step()], winner=1))
        # 写入损坏的 JSON 文件
        game_dir = os.path.join(round_dir, "2")
        os.makedirs(game_dir, exist_ok=True)
        with open(os.path.join(game_dir, "record.json"), "w") as f:
            f.write("INVALID JSON{{{")

        games = scan_round_dir(round_dir)
        assert len(games) == 1
        assert games[0]["game_id"] == 1

    def test_missing_record_json_skipped(self, round_dir):
        """round_dir 中空目录（无 record.json）→ 跳过"""
        game_dir = os.path.join(round_dir, "99")
        os.makedirs(game_dir, exist_ok=True)
        # 不写 record.json

        games = scan_round_dir(round_dir)
        assert len(games) == 0

    def test_nonexistent_round_dir(self):
        """round_dir 不存在 → 抛 FileNotFoundError"""
        with pytest.raises(FileNotFoundError):
            scan_round_dir("/nonexistent/path")


# ------------------------------------------------------------------
# T3: parse_game_steps 空 history
# ------------------------------------------------------------------

class TestParseGameSteps:
    """测试 parse_game_steps 函数"""

    def test_empty_history(self, round_dir):
        """T3: 空 history → total_steps=0, steps=[]"""
        _write_record(round_dir, 1, _make_record([], winner=1))
        result = parse_game_steps(round_dir, 1)
        assert result["game_id"] == 1
        assert result["winner"] == 1
        assert result["is_draw"] is False
        assert result["steps"] == []

    def test_normal_parse(self, round_dir, tmp_path):
        """T10+T12（backend设计）：步数据完整"""
        sid1 = "aaa-111"
        sid2 = "bbb-222"
        h = [
            _make_step(step_id=sid1, strategy="sampling",
                       sampling_probs=[0.02]*225),
            _make_step(step_id=sid2, strategy="argmax"),
        ]
        _write_record(round_dir, 1, _make_record(h, winner=1))
        sample_path = _write_samples(tmp_path, [
            _make_sample(step_id=sid1, output_vec=[0.5]*225),
        ])
        result = parse_game_steps(round_dir, 1, sample_path=sample_path)

        assert len(result["steps"]) == 2
        s0 = result["steps"][0]
        assert s0["step_index"] == 0
        assert s0["player"] == 1
        assert s0["move_position"] == [7, 7]
        assert s0["raw_output"] is not None
        assert s0["sampling_probs"] is not None  # strategy="sampling"
        assert s0["strategy"] == "sampling"
        assert s0["sample_output"] is not None  # step_id 匹配成功

        s1 = result["steps"][1]
        assert s1["step_index"] == 1
        assert s1["sampling_probs"] is None  # strategy="argmax"
        assert s1["strategy"] == "argmax"
        assert s1["sample_output"] is None  # 无匹配样本

    def test_record_not_found(self, round_dir):
        """game_id 不存在 → 抛 FileNotFoundError"""
        with pytest.raises(FileNotFoundError):
            parse_game_steps(round_dir, 999)

    def test_corrupted_record(self, round_dir):
        """record.json 损坏 → 抛 JSONDecodeError"""
        game_dir = os.path.join(round_dir, "1")
        os.makedirs(game_dir, exist_ok=True)
        with open(os.path.join(game_dir, "record.json"), "w") as f:
            f.write("BAD JSON")
        with pytest.raises(json.JSONDecodeError):
            parse_game_steps(round_dir, 1)


# ------------------------------------------------------------------
# T4-T7, T11: match_samples
# ------------------------------------------------------------------

class TestMatchSamples:
    """测试 match_samples 函数"""

    def test_winning_move_match(self):
        """T4: winning_move 样本匹配 → 仅最后步匹配"""
        sid_last = "last-step-id"
        history = [
            _make_step(step_id="sid-1"),
            _make_step(step_id="sid-2"),
            _make_step(step_id=sid_last),
        ]
        samples = [
            _make_sample(step_id=sid_last, output_vec=[1.0]*225),
        ]
        result = match_samples(history, samples)
        assert 2 in result  # 仅最后步（索引2）匹配
        assert 0 not in result
        assert 1 not in result

    def test_winner_path_match(self):
        """T5: winner_path 样本匹配 → 仅获胜方步匹配"""
        # 黑胜(winner=1)：偶数索引步 [0, 2, 4, ...]
        sid0 = "sid-even-0"
        sid1 = "sid-odd-1"
        sid2 = "sid-even-2"
        history = [
            _make_step(step_id=sid0),
            _make_step(step_id=sid1),
            _make_step(step_id=sid2),
        ]
        samples = [
            _make_sample(step_id=sid0, output_vec=[0.5]*225),
            _make_sample(step_id=sid2, output_vec=[0.5]*225),
        ]
        result = match_samples(history, samples)
        assert 0 in result  # 黑方步（偶数索引）
        assert 2 in result  # 黑方步
        assert 1 not in result  # 白方步无样本

    def test_no_sample_file(self):
        """T6: 样本文件不存在 → 所有步 sample_output=null"""
        history = [
            _make_step(step_id="sid-1"),
            _make_step(step_id="sid-2"),
        ]
        result = match_samples(history, [])
        assert len(result) == 0  # 无匹配

    def test_draw_game_no_samples(self):
        """T7: 平局对局 → 无样本匹配"""
        # 平局时 SampleCreator 跳过，样本列表为空
        history = [
            _make_step(step_id="draw-sid-1"),
        ]
        result = match_samples(history, [])
        assert len(result) == 0

    def test_step_id_matching(self):
        """T11: step_id 匹配 → O(1) 无误匹配"""
        sid_a = "uuid-aaaa"
        sid_b = "uuid-bbbb"
        history = [
            _make_step(step_id=sid_a),
            _make_step(step_id=sid_b),
            _make_step(step_id="sid-c"),
        ]
        samples = [
            _make_sample(step_id=sid_a, output_vec=[0.1]*225),
            _make_sample(step_id=sid_b, output_vec=[0.2]*225),
        ]
        result = match_samples(history, samples)
        assert result[0] == [0.1]*225  # sid_a 匹配
        assert result[1] == [0.2]*225  # sid_b 匹配
        assert 2 not in result  # "sid-c" 无匹配样本

    def test_old_format_no_step_id(self):
        """旧格式无 step_id → 全部跳过匹配"""
        history = [
            _make_step(),  # 无 step_id
            _make_step(),  # 无 step_id
        ]
        samples = [
            _make_sample(),  # 无 step_id
        ]
        result = match_samples(history, samples)
        assert len(result) == 0  # 无匹配

    def test_d4_augmentation_same_step_id(self):
        """D4 增强：8 个同 step_id 变体，仅首次写入（恒等变换 tid=0）"""
        sid = "d4-shared-id"
        history = [_make_step(step_id=sid)]
        # D4 增强产生 8 个同 step_id 的样本
        samples = [
            _make_sample(step_id=sid, output_vec=[0.1]*225),
            _make_sample(step_id=sid, output_vec=[0.2]*225),  # 变体
            _make_sample(step_id=sid, output_vec=[0.3]*225),  # 变体
        ]
        result = match_samples(history, samples)
        assert len(result) == 1
        # 仅首次写入（恒等变换的 output）
        assert result[0] == [0.1]*225


# ------------------------------------------------------------------
# T8-T10: infer_temp_side
# ------------------------------------------------------------------

class TestInferTempSide:
    """测试 infer_temp_side 函数"""

    def test_both_sides(self):
        """T9: 双方 → 'both'"""
        history = [
            _make_step(strategy="sampling", input_vec=[0]*225+[1]),
            _make_step(strategy="sampling", input_vec=[0]*225+[2]),
        ]
        assert infer_temp_side(history) == "both"

    def test_black_only(self):
        """黑方采样 → 'black'"""
        history = [
            _make_step(strategy="sampling", input_vec=[0]*225+[1]),
            _make_step(strategy="argmax", input_vec=[0]*225+[2]),
        ]
        assert infer_temp_side(history) == "black"

    def test_white_only(self):
        """白方采样 → 'white'"""
        history = [
            _make_step(strategy="argmax", input_vec=[0]*225+[1]),
            _make_step(strategy="sampling", input_vec=[0]*225+[2]),
        ]
        assert infer_temp_side(history) == "white"

    def test_none_argmax(self):
        """T10: 全 argmax → 'none'"""
        history = [
            _make_step(strategy="argmax"),
            _make_step(strategy="argmax"),
        ]
        assert infer_temp_side(history) == "none"

    def test_old_format_no_strategy(self):
        """T8: 旧格式无 strategy → 'none'"""
        history = [
            _make_step(),  # 无 strategy
            _make_step(),  # 无 strategy
        ]
        assert infer_temp_side(history) == "none"