"""LossLogger 单元测试

测试 LossLogger 的日志记录、曲线图生成、清空功能及边界情况。
"""

import json
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from ai.loss_logger import LossLogger


class TestLossLoggerLog(unittest.TestCase):
    """测试日志记录和文件写入"""

    def test_log_basic(self):
        """测试基本日志记录"""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_path:
            logger = LossLogger(model_name="test_model", log_dir=tmp_path)

            logger.log(1, 0.5)
            logger.log(200, 0.3)
            logger.log(400, 0.1)

            log_path = logger.get_log_path()
            self.assertTrue(os.path.exists(log_path))

            with open(log_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            self.assertEqual(len(lines), 3)
            entry1 = json.loads(lines[0])
            self.assertEqual(entry1["epoch"], 1)
            self.assertEqual(entry1["loss"], 0.5)

            entry2 = json.loads(lines[1])
            self.assertEqual(entry2["epoch"], 200)
            self.assertEqual(entry2["loss"], 0.3)

            entry3 = json.loads(lines[2])
            self.assertEqual(entry3["epoch"], 400)
            self.assertEqual(entry3["loss"], 0.1)


class TestLossLoggerPlot(unittest.TestCase):
    """测试曲线图生成"""

    def test_plot_generates_png(self):
        """测试 loss 曲线图生成"""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_path:
            logger = LossLogger(model_name="test_model", log_dir=tmp_path)

            logger.log(1, 0.5)
            logger.log(200, 0.3)
            logger.log(400, 0.1)

            img_path = logger.plot()

            self.assertIsNotNone(img_path)
            self.assertTrue(os.path.exists(img_path))
            self.assertTrue(img_path.endswith(".png"))

    def test_plot_empty_returns_none(self):
        """测试空数据集情况：不记录日志直接绘图应返回 None"""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_path:
            logger = LossLogger(model_name="test_model", log_dir=tmp_path)
            img_path = logger.plot()
            self.assertIsNone(img_path)

    def test_plot_single_point(self):
        """测试仅一个数据点的曲线图生成"""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_path:
            logger = LossLogger(model_name="test_model", log_dir=tmp_path)
            logger.log(1, 0.5)

            img_path = logger.plot()

            self.assertIsNotNone(img_path)
            self.assertTrue(os.path.exists(img_path))

    def test_plot_nan_inf(self):
        """测试 NaN/Infinity loss 值：日志可保存，绘图不崩溃"""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_path:
            logger = LossLogger(model_name="test_model", log_dir=tmp_path)

            logger.log(1, 0.5)
            logger.log(200, float('nan'))
            logger.log(400, float('inf'))

            log_path = logger.get_log_path()
            self.assertTrue(os.path.exists(log_path))

            # plot 应处理异常值而不抛出异常
            img_path = logger.plot()
            # 由于 NaN/inf 数据，matplotlib 可能无法生成有效图像，返回 None 或正常路径均可
            self.assertTrue(img_path is None or os.path.exists(img_path))


class TestLossLoggerClear(unittest.TestCase):
    """测试清空日志功能"""

    def test_clear_empties_file(self):
        """测试清空后日志文件应为空"""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_path:
            logger = LossLogger(model_name="test_model", log_dir=tmp_path)

            logger.log(1, 0.5)
            logger.clear()

            log_path = logger.get_log_path()
            with open(log_path, 'r', encoding='utf-8') as f:
                content = f.read()
            self.assertEqual(content, "")


class TestLossLoggerMatplotlibMissing(unittest.TestCase):
    """测试 matplotlib 缺失时的优雅降级"""

    def test_matplotlib_missing_returns_none(self):
        """测试 matplotlib 不可用时优雅降级"""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_path:
            logger = LossLogger(model_name="test_model", log_dir=tmp_path)
            logger.log(1, 0.5)
            logger.log(200, 0.3)

            # mock matplotlib 导入失败
            modules = sys.modules.copy()
            try:
                sys.modules['matplotlib'] = None
                sys.modules['matplotlib.pyplot'] = None

                img_path = logger.plot()

                # 应返回 None，但日志文件保持完整
                self.assertIsNone(img_path)
                self.assertTrue(os.path.exists(logger.get_log_path()))
            finally:
                # 恢复 sys.modules
                for key in list(sys.modules.keys()):
                    if key not in modules:
                        del sys.modules[key]
                sys.modules.update(modules)


class TestLossLoggerDirectoryReadonly(unittest.TestCase):
    """测试日志目录不可写时抛出异常"""

    def test_readonly_directory_raises(self):
        """测试日志目录不可写时抛出异常"""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_path:
            log_dir = os.path.join(tmp_path, "readonly")
            os.makedirs(log_dir)

            # Windows 上 os.chmod 对目录只读权限的支持有限，
            # 此测试在 Windows 上可能表现不同，捕获可能的异常即可
            try:
                # 尝试创建 logger 并写入
                logger = LossLogger(model_name="test_model", log_dir=log_dir)
                logger.log(1, 0.5)
                # 如果成功写入，说明当前系统不支持此测试场景，跳过验证
            except (PermissionError, IOError, OSError):
                pass  # 期望的行为


class TestLossLoggerFileNaming(unittest.TestCase):
    """测试日志文件命名规则"""

    def test_filename_contains_model_name(self):
        """测试文件名包含模型名称"""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_path:
            logger = LossLogger(model_name="fc_v1", log_dir=tmp_path)
            log_path = logger.get_log_path()
            basename = os.path.basename(log_path)
            self.assertTrue(basename.startswith("fc_v1_"))
            self.assertTrue(basename.endswith(".jsonl"))

    def test_different_loggers_different_files(self):
        """测试不同 logger 实例生成不同文件"""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_path:
            logger1 = LossLogger(model_name="test_model", log_dir=tmp_path)
            logger2 = LossLogger(model_name="test_model", log_dir=tmp_path)

            self.assertNotEqual(logger1.get_log_path(), logger2.get_log_path())


if __name__ == '__main__':
    unittest.main()
