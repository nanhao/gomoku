"""merge_samples.py 测试

测试样本合并模块的发现、合并和 CLI 功能。
"""

import json
import os
import sys
import tempfile
import unittest

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from ai.merge_samples import discover_sample_files, merge_sample_files, merge_directories


class TestDiscoverSampleFiles(unittest.TestCase):
    """测试 discover_sample_files 函数"""

    def test_discover_sample_files_finds_all(self):
        """递归发现所有 sample.json，包括子目录"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # 创建层级目录结构
            dir1 = os.path.join(tmp_dir, "a", "1")
            dir2 = os.path.join(tmp_dir, "b", "2")
            os.makedirs(dir1)
            os.makedirs(dir2)

            # 创建 sample.json
            for d in [dir1, dir2]:
                with open(os.path.join(d, "sample.json"), 'w', encoding='utf-8') as f:
                    json.dump([], f)

            result = discover_sample_files(tmp_dir)
            self.assertEqual(len(result), 2)
            # 结果按路径排序
            self.assertTrue(result[0].endswith("sample.json"))
            self.assertTrue(result[1].endswith("sample.json"))

    def test_discover_sample_files_ignores_merge(self):
        """不收集 merge_*.json 文件"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # 创建普通 sample.json
            sample_dir = os.path.join(tmp_dir, "data")
            os.makedirs(sample_dir)
            with open(os.path.join(sample_dir, "sample.json"), 'w', encoding='utf-8') as f:
                json.dump([], f)

            # 创建 merge 文件
            with open(os.path.join(tmp_dir, "merge_cnn_v2.json"), 'w', encoding='utf-8') as f:
                json.dump([], f)

            result = discover_sample_files(tmp_dir)
            self.assertEqual(len(result), 1)
            self.assertTrue(result[0].endswith(os.path.join("data", "sample.json")))

    def test_discover_sample_files_empty_dir(self):
        """空目录返回空列表"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            result = discover_sample_files(tmp_dir)
            self.assertEqual(result, [])

    def test_discover_sample_files_sorted(self):
        """返回结果按路径排序"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # 按反向顺序创建目录，验证返回时按字母排序
            for name in ["z", "a", "m"]:
                d = os.path.join(tmp_dir, name)
                os.makedirs(d)
                with open(os.path.join(d, "sample.json"), 'w', encoding='utf-8') as f:
                    json.dump([], f)

            result = discover_sample_files(tmp_dir)
            basenames = [os.path.dirname(p).split(os.sep)[-1] for p in result]
            self.assertEqual(basenames, ["a", "m", "z"])


class TestMergeSampleFiles(unittest.TestCase):
    """测试 merge_sample_files 函数"""

    def test_merge_sample_files_correct_count(self):
        """合并后样本数等于各文件样本数之和"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # 创建两个样本文件
            file1 = os.path.join(tmp_dir, "sample1.json")
            file2 = os.path.join(tmp_dir, "sample2.json")

            with open(file1, 'w', encoding='utf-8') as f:
                json.dump([{"input": [1.0] * 226, "output": [0.0] * 225, "mask": [1] * 225}], f)
            with open(file2, 'w', encoding='utf-8') as f:
                json.dump([
                    {"input": [2.0] * 226, "output": [0.0] * 225, "mask": [1] * 225},
                    {"input": [3.0] * 226, "output": [0.0] * 225, "mask": [1] * 225},
                ], f)

            output = os.path.join(tmp_dir, "merged.json")
            result = merge_sample_files([file1, file2], output)

            self.assertEqual(result["total_files"], 2)
            self.assertEqual(result["total_samples"], 3)
            self.assertEqual(result["output_path"], output)

    def test_merge_sample_files_output_valid(self):
        """输出文件可被 json.load 正确读取"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            file1 = os.path.join(tmp_dir, "sample1.json")
            with open(file1, 'w', encoding='utf-8') as f:
                json.dump([{"input": [1.0] * 226, "output": [0.0] * 225, "mask": [1] * 225}], f)

            output = os.path.join(tmp_dir, "merged.json")
            merge_sample_files([file1], output)

            with open(output, 'r', encoding='utf-8') as f:
                data = json.load(f)

            self.assertIsInstance(data, list)
            self.assertEqual(len(data), 1)
            self.assertEqual(data[0]["input"], [1.0] * 226)

    def test_merge_sample_files_creates_dir(self):
        """输出目录不存在时自动创建"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            file1 = os.path.join(tmp_dir, "sample1.json")
            with open(file1, 'w', encoding='utf-8') as f:
                json.dump([], f)

            # 输出目录尚不存在
            output = os.path.join(tmp_dir, "subdir", "nested", "merged.json")
            self.assertFalse(os.path.exists(os.path.dirname(output)))

            merge_sample_files([file1], output)

            self.assertTrue(os.path.exists(output))

    def test_merge_sample_files_empty_list(self):
        """空文件列表合并后结果为空列表"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            output = os.path.join(tmp_dir, "merged.json")
            result = merge_sample_files([], output)

            self.assertEqual(result["total_files"], 0)
            self.assertEqual(result["total_samples"], 0)

            with open(output, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.assertEqual(data, [])

    def test_merge_sample_files_compact_format(self):
        """输出文件应为紧凑格式（无换行缩进）"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            file1 = os.path.join(tmp_dir, "sample1.json")
            with open(file1, 'w', encoding='utf-8') as f:
                json.dump([{"input": [1.0] * 226, "output": [0.0] * 225, "mask": [1] * 225}], f)

            output = os.path.join(tmp_dir, "merged.json")
            merge_sample_files([file1], output)

            with open(output, 'r', encoding='utf-8') as f:
                content = f.read()

            # 紧凑格式不包含换行符
            self.assertNotIn('\n', content)


class TestMergeDirectories(unittest.TestCase):
    """测试 merge_directories 函数"""

    def test_merge_directories_multi_dir(self):
        """多目录合并样本数正确"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            dir1 = os.path.join(tmp_dir, "dir1")
            dir2 = os.path.join(tmp_dir, "dir2")
            os.makedirs(dir1)
            os.makedirs(dir2)

            # dir1 中创建一个 sample.json（2 条样本）
            with open(os.path.join(dir1, "sample.json"), 'w', encoding='utf-8') as f:
                json.dump([
                    {"input": [1.0] * 226, "output": [0.0] * 225, "mask": [1] * 225},
                    {"input": [2.0] * 226, "output": [0.0] * 225, "mask": [1] * 225},
                ], f)

            # dir2 中创建子目录和 sample.json（1 条样本）
            sub = os.path.join(dir2, "sub")
            os.makedirs(sub)
            with open(os.path.join(sub, "sample.json"), 'w', encoding='utf-8') as f:
                json.dump([{"input": [3.0] * 226, "output": [0.0] * 225, "mask": [1] * 225}], f)

            output = os.path.join(tmp_dir, "merged.json")
            result = merge_directories([dir1, dir2], output)

            self.assertEqual(result["total_files"], 2)
            self.assertEqual(result["total_samples"], 3)


class TestMergeSamplesCLI(unittest.TestCase):
    """测试 CLI 入口"""

    def test_cli_parses_args(self):
        """CLI 解析位置参数和 -o 参数"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            data_dir = os.path.join(tmp_dir, "data")
            os.makedirs(data_dir)
            with open(os.path.join(data_dir, "sample.json"), 'w', encoding='utf-8') as f:
                json.dump([{"input": [1.0] * 226, "output": [0.0] * 225, "mask": [1] * 225}], f)

            output = os.path.join(tmp_dir, "out.json")

            # 模拟命令行调用
            old_argv = sys.argv
            try:
                sys.argv = ["merge_samples.py", data_dir, "-o", output]
                from ai import merge_samples
                merge_samples.main()
            finally:
                sys.argv = old_argv

            self.assertTrue(os.path.exists(output))
            with open(output, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.assertEqual(len(data), 1)

    def test_cli_requires_output(self):
        """CLI 不指定 -o 时报错退出"""
        old_argv = sys.argv
        try:
            sys.argv = ["merge_samples.py", "/tmp/data"]
            from ai import merge_samples
            with self.assertRaises(SystemExit):
                merge_samples.main()
        finally:
            sys.argv = old_argv

    def test_cli_requires_directory(self):
        """CLI 不指定目录时报错退出"""
        old_argv = sys.argv
        try:
            sys.argv = ["merge_samples.py", "-o", "/tmp/out.json"]
            from ai import merge_samples
            with self.assertRaises(SystemExit):
                merge_samples.main()
        finally:
            sys.argv = old_argv


if __name__ == '__main__':
    unittest.main()
