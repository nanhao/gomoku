"""train_main CLI 参数测试

测试已移除的参数（--self-play、--merge-output）和 --train-from-dir 的行为。
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from ai.train_main import main
from ai.runners import discover_sample_dirs


class TestSelfPlayParamRemoved(unittest.TestCase):
    """验证 --self-play 参数已移除"""

    def test_self_play_param_removed(self):
        """--self-play 参数已不存在，解析时报错"""
        with patch('sys.argv', ['train_main.py', '--model', 'fc_v1', '--self-play']):
            with self.assertRaises(SystemExit):
                main()

    def test_merge_output_param_removed(self):
        """--merge-output 参数已不存在"""
        with patch('sys.argv', ['train_main.py', '--model', 'fc_v1', '--merge-output', '/tmp/out.json']):
            with self.assertRaises(SystemExit):
                main()


class TestRemovedCycleParam(unittest.TestCase):
    """验证 --cycle 参数已移除"""

    def test_cycle_param_removed(self):
        """--cycle 参数已不存在，解析时报错"""
        with patch('sys.argv', ['train_main.py', '--model', 'fc_v1', '--cycle']):
            with self.assertRaises(SystemExit):
                main()


class TestDiscoverSampleDirs(unittest.TestCase):
    """测试 discover_sample_dirs 辅助函数"""

    def test_discover_finds_all_sample_dirs(self):
        """正确发现所有包含 sample.json 的子目录"""
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            # 创建子目录 1 和 2，包含 sample.json
            for i in [1, 2]:
                sub_dir = os.path.join(tmpdir, str(i))
                os.makedirs(sub_dir)
                with open(os.path.join(sub_dir, "sample.json"), 'w') as f:
                    f.write("{}")

            result = discover_sample_dirs(tmpdir)
            self.assertEqual(len(result), 2)

    def test_discover_skips_dirs_without_sample(self):
        """跳过没有 sample.json 的子目录"""
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            # 子目录 1 包含 sample.json
            sub_dir1 = os.path.join(tmpdir, "1")
            os.makedirs(sub_dir1)
            with open(os.path.join(sub_dir1, "sample.json"), 'w') as f:
                f.write("{}")

            # 子目录 2 不包含 sample.json
            sub_dir2 = os.path.join(tmpdir, "2")
            os.makedirs(sub_dir2)

            result = discover_sample_dirs(tmpdir)
            self.assertEqual(len(result), 1)
            self.assertTrue(result[0].endswith(os.sep + "1"))

    def test_discover_returns_sorted_dirs(self):
        """返回按数字排序的目录列表"""
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            # 创建子目录 3, 1, 2（乱序）
            for i in [3, 1, 2]:
                sub_dir = os.path.join(tmpdir, str(i))
                os.makedirs(sub_dir)
                with open(os.path.join(sub_dir, "sample.json"), 'w') as f:
                    f.write("{}")

            result = discover_sample_dirs(tmpdir)
            # 验证按数字排序
            basenames = [os.path.basename(d) for d in result]
            self.assertEqual(basenames, ["1", "2", "3"])

    def test_discover_empty_dir_returns_empty(self):
        """空目录返回空列表"""
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            result = discover_sample_dirs(tmpdir)
            self.assertEqual(result, [])

    def test_discover_nonexistent_dir_returns_empty(self):
        """不存在的目录返回空列表"""
        result = discover_sample_dirs("/nonexistent/path/12345")
        self.assertEqual(result, [])


if __name__ == '__main__':
    unittest.main()
