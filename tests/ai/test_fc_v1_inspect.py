"""SampleInspector 测试用例

按照设计文档 section 12.5 编写：
- primary: 主流程测试
- edge: 边缘/分支测试
"""

import json
import os
import sys
import tempfile
import unittest

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from ai.models.fc_v1.model import FCv1Model
from ai.models.fc_v1.sample_inspector import SampleInspector


def _make_sample_json(tmpdir: str, steps: int = 3) -> str:
    """创建用于测试的临时 sample.json。

    最后一步的第0个空位 reward=+1000000，其余空位=0，已占位=-1000000。
    """
    data = []
    for i in range(steps):
        occupied = i  # 前 i 个位置已落子
        inp = [1] * occupied + [0] * (225 - occupied)  # 棋盘（占位用1代替）
        inp.append(1)  # 当前落子方=黑

        out = [-1000000] * occupied + [0] * (225 - occupied)
        # 最后一步的第一个空位设置为 +1000000（胜利信号）
        if i == steps - 1 and occupied < 225:
            out[occupied] = 1000000

        mask = [1] * 225
        data.append({"input": inp, "output": out, "mask": mask})

    sample_path = os.path.join(tmpdir, "sample.json")
    with open(sample_path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    return sample_path


class TestSampleInspectorPrimary(unittest.TestCase):
    """主流程测试"""

    def setUp(self):
        # 每个测试用新模型（避免加载已有 pth 影响结果）
        self.model = FCv1Model(fresh=True)
        self.inspector = SampleInspector(self.model)
        self.tmpdir = tempfile.mkdtemp()
        self.sample_path = _make_sample_json(self.tmpdir, steps=5)

    def test_inspect_returns_correct_step_count(self):
        """[primary] inspect 返回列表长度等于 sample 步数"""
        results = self.inspector.inspect(self.sample_path)
        self.assertEqual(len(results), 5)

    def test_step_result_has_required_fields(self):
        """[primary] 每个 step_result 包含必要字段"""
        required_fields = {
            "step", "is_last", "player",
            "occupied_count", "empty_count",
            "model_output", "target",
            "loss", "win_pos", "win_model_output",
        }
        results = self.inspector.inspect(self.sample_path)
        for r in results:
            for field in required_fields:
                self.assertIn(field, r, f"缺少字段: {field}")

    def test_is_last_only_on_last_step(self):
        """[primary] is_last 仅在最后一步为 True"""
        results = self.inspector.inspect(self.sample_path)
        for i, r in enumerate(results):
            if i == len(results) - 1:
                self.assertTrue(r["is_last"])
            else:
                self.assertFalse(r["is_last"])

    def test_loss_is_non_negative_float(self):
        """[primary] loss 为非负浮点数"""
        results = self.inspector.inspect(self.sample_path)
        for r in results:
            self.assertIsInstance(r["loss"], float)
            self.assertGreaterEqual(r["loss"], 0.0)

    def test_format_step_returns_nonempty_string(self):
        """[primary] format_step 返回非空字符串"""
        results = self.inspector.inspect(self.sample_path)
        for r in results:
            text = self.inspector.format_step(r)
            self.assertIsInstance(text, str)
            self.assertGreater(len(text.strip()), 0)


class TestSampleInspectorEdge(unittest.TestCase):
    """边缘测试"""

    def setUp(self):
        self.model = FCv1Model(fresh=True)
        self.inspector = SampleInspector(self.model)
        self.tmpdir = tempfile.mkdtemp()

    def test_win_pos_detected_in_last_step(self):
        """[edge] win_pos 在有 +1e6 的步骤（最后一步）中不为 None"""
        sample_path = _make_sample_json(self.tmpdir, steps=3)
        results = self.inspector.inspect(sample_path)
        last = results[-1]
        self.assertIsNotNone(last["win_pos"])
        self.assertIsNotNone(last["win_model_output"])

    def test_win_pos_none_in_non_last_step(self):
        """[edge] win_pos 在无 +1e6 的步骤中为 None"""
        sample_path = _make_sample_json(self.tmpdir, steps=3)
        results = self.inspector.inspect(sample_path)
        for r in results[:-1]:
            self.assertIsNone(r["win_pos"])
            self.assertIsNone(r["win_model_output"])

    def test_model_output_length_is_225(self):
        """[edge] model_output 长度为 225"""
        sample_path = _make_sample_json(self.tmpdir, steps=2)
        results = self.inspector.inspect(sample_path)
        for r in results:
            self.assertEqual(len(r["model_output"]), 225)

    def test_target_equals_sample_output(self):
        """[edge] target 与 sample.json 中的 output 原始值一致（未归一化）"""
        sample_path = _make_sample_json(self.tmpdir, steps=2)
        # 读取原始数据
        with open(sample_path, encoding="utf-8") as f:
            raw = json.load(f)
        results = self.inspector.inspect(sample_path)
        for i, r in enumerate(results):
            self.assertEqual(r["target"], raw[i]["output"])

    def test_occupied_and_empty_count_sum_to_225(self):
        """[edge] occupied_count + empty_count == 225"""
        sample_path = _make_sample_json(self.tmpdir, steps=4)
        results = self.inspector.inspect(sample_path)
        for r in results:
            self.assertEqual(r["occupied_count"] + r["empty_count"], 225)


if __name__ == "__main__":
    unittest.main()
