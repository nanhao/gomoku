"""OffPolicyTrainer 样本加载单元测试（设计 58 TC-OPT-01~10）

验证 _load_samples 的三种加载规则（manifest+chunk / sample.json / 目录扫描）
与容错行为（损坏 JSON、缺字段、维度不符、空目录）。

测试均在临时目录中构造数据，禁止触碰真实样本文件。
"""

import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
import unittest

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


class TestOffPolicyTrainerLoadSamples(unittest.TestCase):
    """OffPolicyTrainer._load_samples 加载与容错测试（TC-OPT-01~10）"""

    @classmethod
    def setUpClass(cls):
        try:
            from ai.off_policy_trainer import OffPolicyTrainer
            cls.OffPolicyTrainer = OffPolicyTrainer
        except ImportError:
            cls.OffPolicyTrainer = None

    def setUp(self):
        if self.OffPolicyTrainer is None:
            self.skipTest("OffPolicyTrainer 尚未实现")
        self._tmpdir = tempfile.mkdtemp(prefix="off_policy_opt_")

    def tearDown(self):
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    # ---------- 样本构造辅助（类内静态方法，无全局函数） ----------

    @staticmethod
    def _make_sample(idx, state_len=226, pi_len=225, drop_field=None):
        """构造单条样本记录

        Args:
            idx: 样本序号（写入 state[0] 便于内容断言）
            state_len: state 维度（默认 226）
            pi_len: pi 维度（默认 225）
            drop_field: 要删除的字段名（None 表示保留全部字段）

        Returns:
            样本 dict {state, pi, z}
        """
        state = [0] * state_len
        state[0] = idx + 1
        pi = [1.0 / pi_len] * pi_len
        z = 1.0 if idx % 2 == 0 else -1.0
        sample = {"state": state, "pi": pi, "z": z}
        if drop_field is not None:
            del sample[drop_field]
        return sample

    @staticmethod
    def _write_json(path, data):
        """写入 JSON 文件"""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    @staticmethod
    def _write_manifest(data_dir, chunks):
        """写入 manifest.json（chunks 列表格式与 renjunet_samples 一致）"""
        manifest = {
            "max_size": 1000000000000,
            "chunk_games": 10,
            "total_samples": sum(c["sample_count"] for c in chunks),
            "chunks": chunks,
        }
        with open(os.path.join(data_dir, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f)

    # ---------- 测试用例 ----------

    def test_empty_dir_returns_empty_lists(self):
        """TC-OPT-01：空目录返回三个空列表"""
        states, pis, zs = self.OffPolicyTrainer._load_samples([self._tmpdir])
        self.assertEqual(states, [])
        self.assertEqual(pis, [])
        self.assertEqual(zs, [])

    def test_manifest_chunk_loading(self):
        """TC-OPT-02：manifest + 2 个 chunk 共 50 条全部加载"""
        chunk1 = [self._make_sample(i) for i in range(25)]
        chunk2 = [self._make_sample(i + 100) for i in range(25)]
        self._write_json(os.path.join(self._tmpdir, "chunk_000001_000001.json"), chunk1)
        self._write_json(os.path.join(self._tmpdir, "chunk_000002_000002.json"), chunk2)
        self._write_manifest(self._tmpdir, [
            {"game_start": 1, "game_end": 1, "sample_count": 25,
             "file": "chunk_000001_000001.json"},
            {"game_start": 2, "game_end": 2, "sample_count": 25,
             "file": "chunk_000002_000002.json"},
        ])

        states, pis, zs = self.OffPolicyTrainer._load_samples([self._tmpdir])

        self.assertEqual(len(states), 50)
        self.assertEqual(len(pis), 50)
        self.assertEqual(len(zs), 50)
        # 维度全部正确
        for s, p in zip(states, pis):
            self.assertEqual(len(s), 226)
            self.assertEqual(len(p), 225)
        # 内容与 chunk 一致：前 25 条来自 chunk1（state[0] = idx+1）
        self.assertEqual(states[0][0], 1)
        self.assertEqual(states[24][0], 25)
        self.assertEqual(states[25][0], 101)
        self.assertEqual(states[49][0], 125)

    def test_corrupted_chunk_skipped_with_warning(self):
        """TC-OPT-03：损坏 chunk 跳过并打印警告，不抛异常"""
        good = [self._make_sample(i) for i in range(30)]
        self._write_json(os.path.join(self._tmpdir, "chunk_ok.json"), good)
        with open(os.path.join(self._tmpdir, "chunk_bad.json"), "w", encoding="utf-8") as f:
            f.write("{invalid json!!")

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            states, pis, zs = self.OffPolicyTrainer._load_samples([self._tmpdir])

        self.assertEqual(len(states), 30)
        self.assertEqual(len(pis), 30)
        self.assertEqual(len(zs), 30)
        self.assertIn("警告", buf.getvalue())
        self.assertIn("chunk_bad.json", buf.getvalue())

    def test_scan_json_files_without_manifest(self):
        """TC-OPT-04：无 manifest，目录扫描 3 个 chunk_*.json 共 30 条"""
        for i in range(3):
            samples = [self._make_sample(i * 10 + j) for j in range(10)]
            self._write_json(os.path.join(self._tmpdir, f"chunk_00{i + 1}.json"), samples)

        states, pis, zs = self.OffPolicyTrainer._load_samples([self._tmpdir])

        self.assertEqual(len(states), 30)
        self.assertEqual(len(pis), 30)
        self.assertEqual(len(zs), 30)

    def test_sample_json_compat(self):
        """TC-OPT-05：无 manifest，sample.json 兼容加载 20 条"""
        samples = [self._make_sample(i) for i in range(20)]
        self._write_json(os.path.join(self._tmpdir, "sample.json"), samples)

        states, pis, zs = self.OffPolicyTrainer._load_samples([self._tmpdir])

        self.assertEqual(len(states), 20)
        self.assertEqual(len(pis), 20)
        self.assertEqual(len(zs), 20)
        self.assertEqual(states[0][0], 1)

    def test_manifest_priority_over_sample_json(self):
        """TC-OPT-06：manifest 与 sample.json 共存时只加载 manifest 指向的 chunk"""
        chunk = [self._make_sample(i) for i in range(40)]
        self._write_json(os.path.join(self._tmpdir, "chunk_main.json"), chunk)
        self._write_manifest(self._tmpdir, [
            {"game_start": 1, "game_end": 1, "sample_count": 40, "file": "chunk_main.json"},
        ])
        extra = [self._make_sample(i + 1000) for i in range(20)]
        self._write_json(os.path.join(self._tmpdir, "sample.json"), extra)

        states, _, _ = self.OffPolicyTrainer._load_samples([self._tmpdir])

        self.assertEqual(len(states), 40)
        # 内容来自 chunk（state[0]=1）而非 sample.json（state[0]=1001）
        self.assertEqual(states[0][0], 1)

    def test_missing_fields_skipped_with_count(self):
        """TC-OPT-07：缺字段样本跳过并计数、打印提示"""
        samples = [self._make_sample(i) for i in range(10)]
        no_z = self._make_sample(100, drop_field="z")
        no_state = self._make_sample(200, drop_field="state")
        samples += [no_z, no_z, no_state]
        self._write_json(os.path.join(self._tmpdir, "chunk_mix.json"), samples)

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            states, pis, zs = self.OffPolicyTrainer._load_samples([self._tmpdir])

        self.assertEqual(len(states), 10)
        self.assertEqual(len(pis), 10)
        self.assertEqual(len(zs), 10)
        self.assertIn("跳过 3 条", buf.getvalue())

    def test_dimension_mismatch_skipped(self):
        """TC-OPT-08：state/pi 维度不符样本跳过并计数"""
        samples = [self._make_sample(i) for i in range(10)]
        bad_state = self._make_sample(100, state_len=225)
        bad_pi = self._make_sample(200, pi_len=224)
        samples += [bad_state, bad_state, bad_pi]
        self._write_json(os.path.join(self._tmpdir, "chunk_dim.json"), samples)

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            states, pis, zs = self.OffPolicyTrainer._load_samples([self._tmpdir])

        self.assertEqual(len(states), 10)
        self.assertEqual(len(pis), 10)
        self.assertEqual(len(zs), 10)
        self.assertIn("跳过 3 条", buf.getvalue())

    def test_manifest_not_parsed_as_samples(self):
        """TC-OPT-09：manifest 优先时只加载其 chunks 指向的文件"""
        chunk = [self._make_sample(i) for i in range(8)]
        self._write_json(os.path.join(self._tmpdir, "only_chunk.json"), chunk)
        self._write_manifest(self._tmpdir, [
            {"game_start": 1, "game_end": 1, "sample_count": 8, "file": "only_chunk.json"},
        ])
        # 目录中还有另一个未被 manifest 引用的 json，不应被加载
        other = [self._make_sample(i + 500) for i in range(5)]
        self._write_json(os.path.join(self._tmpdir, "other.json"), other)

        states, _, _ = self.OffPolicyTrainer._load_samples([self._tmpdir])

        self.assertEqual(len(states), 8)
        self.assertEqual(states[0][0], 1)

    def test_empty_dir_via_train_epochs(self):
        """TC-OPT-10：空目录经 train_epochs 调用不抛异常、history 为空"""
        from ai.off_policy_trainer import OffPolicyTrainer
        from tests.unit.test_dual_head_trainer import MockDualHeadModel

        trainer = OffPolicyTrainer(model=MockDualHeadModel())
        history = trainer.train_epochs(
            sample_dirs=[self._tmpdir], epochs=2, batch_size=4
        )
        self.assertEqual(history, [])


if __name__ == "__main__":
    unittest.main()
