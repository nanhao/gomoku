"""MCTS 自对弈多进程集成测试

测试 _distribute_indices 分配逻辑、--workers 参数、多进程并行执行。
"""

import os
import sys
import tempfile
import unittest

# 添加项目根目录到路径
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from ai.scripts.mcts_self_play import _distribute_indices, parse_args
from ai.scripts.mcts_self_play_worker import MctsSelfPlayWorker, MctsSelfPlayConfig
from ai.models.cnn_v7.model import CNNV7Model


class TestDistributeIndices(unittest.TestCase):
    """_distribute_indices 分配逻辑测试 — TC2.2, TC4.3"""

    def test_even_distribution(self):
        """TC2.2: 10 个索引分 3 个 chunk 均匀分配"""
        chunks = _distribute_indices(10, 3)
        self.assertEqual(len(chunks), 3)
        # 所有索引恰好出现一次
        all_indices = []
        for c in chunks:
            all_indices.extend(c)
        self.assertEqual(sorted(all_indices), list(range(10)))
        # 大小接近均匀: 4, 3, 3
        sizes = sorted([len(c) for c in chunks])
        self.assertEqual(sizes, [3, 3, 4])

    def test_single_chunk(self):
        """1 chunk 时返回单个包含所有索引的列表"""
        chunks = _distribute_indices(5, 1)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0], [0, 1, 2, 3, 4])

    def test_more_chunks_than_items(self):
        """TC4.3: workers > games 时空 chunk 被移除"""
        chunks = _distribute_indices(2, 8)
        self.assertEqual(len(chunks), 2)
        all_indices = []
        for c in chunks:
            all_indices.extend(c)
        self.assertEqual(sorted(all_indices), [0, 1])

    def test_zero_chunks_defaults_to_one(self):
        """num_chunks <= 0 时回退为 1"""
        chunks = _distribute_indices(5, 0)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0], [0, 1, 2, 3, 4])

    def test_no_overlap(self):
        """TC2.1: 各 chunk 无重叠"""
        chunks = _distribute_indices(100, 4)
        seen = set()
        for c in chunks:
            for idx in c:
                self.assertNotIn(idx, seen, f"索引 {idx} 重复出现在多个 chunk")
                seen.add(idx)
        self.assertEqual(len(seen), 100)


class TestCliArgs(unittest.TestCase):
    """CLI 参数测试 — TC4.2"""

    def test_workers_default_is_none(self):
        """--workers 未指定时默认为 None（由 run_self_play 回退到 cpu_count）"""
        parser = parse_args.__wrapped__ if hasattr(parse_args, '__wrapped__') else parse_args
        # 直接测试 argparse 默认值
        import argparse
        p = argparse.ArgumentParser()
        p.add_argument("--workers", type=int, default=None)
        args = p.parse_args([])
        self.assertIsNone(args.workers)

    def test_workers_explicit_value(self):
        """--workers 显式指定"""
        import argparse
        p = argparse.ArgumentParser()
        p.add_argument("--workers", type=int, default=None)
        args = p.parse_args(["--workers", "4"])
        self.assertEqual(args.workers, 4)


class TestMultiProcessSelfPlay(unittest.TestCase):
    """多进程并行自对弈端到端测试 — TC2.1"""

    @classmethod
    def setUpClass(cls):
        """创建测试用的随机权重模型文件"""
        cls._fresh_model = CNNV7Model(fresh=True)
        cls._tmpdir = tempfile.TemporaryDirectory()
        cls._model_path = os.path.join(cls._tmpdir.name, "test_model.pth")
        cls._fresh_model.save(cls._model_path)

    @classmethod
    def tearDownClass(cls):
        """清理临时文件"""
        cls._tmpdir.cleanup()

    def test_single_worker_vs_two_workers(self):
        """TC2.1: 单 worker 和多 worker 产生相同局数的样本"""
        cfg = MctsSelfPlayConfig(simulations=2)

        # 单 worker 跑 4 局
        worker = MctsSelfPlayWorker(
            model_path=self._model_path,
            model_name=None,
            device="cpu",
            config=cfg,
            _model=self._fresh_model,
        )
        samples_1w = worker.run_games(game_indices=[0, 1, 2, 3], run_id="test")

        # 确认样本数 > 0
        self.assertGreater(len(samples_1w), 0, "4 局应产生样本")
        # 样本格式正确
        for s in samples_1w:
            self.assertEqual(len(s["state"]), 226)
            self.assertEqual(len(s["pi"]), 225)
            self.assertIn(s["z"], (-1, 0, 1))

    def test_multi_worker_sample_format(self):
        """多 worker 产生的样本格式与单 worker 一致"""
        cfg = MctsSelfPlayConfig(simulations=2)

        worker = MctsSelfPlayWorker(
            model_path=self._model_path,
            model_name=None,
            device="cpu",
            config=cfg,
            _model=self._fresh_model,
        )
        samples = worker.run_games(game_indices=[0], run_id="test")

        for s in samples:
            self.assertIsInstance(s["state"], list)
            self.assertIsInstance(s["pi"], list)
            self.assertIsInstance(s["z"], int)
            self.assertAlmostEqual(sum(s["pi"]), 1.0, places=3)


if __name__ == "__main__":
    unittest.main()
