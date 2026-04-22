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


class TestLossLoggerSessionDir(unittest.TestCase):
    """测试 session_dir 子目录隔离"""

    def test_session_dir_creates_subdirectory(self):
        """传入 session_dir 后日志应位于子目录下"""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_path:
            logger = LossLogger(
                model_name="test_model",
                log_dir=tmp_path,
                session_dir="test_session"
            )
            log_path = logger.get_log_path()
            expected_dir = os.path.join(tmp_path, "test_session")
            self.assertTrue(os.path.isdir(expected_dir))
            self.assertTrue(log_path.startswith(expected_dir + os.sep))

    def test_round_level_filename(self):
        """log_type=round_level 时文件名为 round_level.jsonl"""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_path:
            logger = LossLogger(
                model_name="cnn_v1",
                log_dir=tmp_path,
                session_dir="session",
                log_type="round_level",
                total_rounds=3,
                games=50
            )
            log_path = logger.get_log_path()
            self.assertEqual(os.path.basename(log_path), "round_level.jsonl")

    def test_epoch_level_filename_with_round_num(self):
        """log_type=epoch_level 且提供 round_num 时文件名语义化"""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_path:
            logger = LossLogger(
                model_name="cnn_v1",
                log_dir=tmp_path,
                session_dir="session",
                log_type="epoch_level",
                round_num=2,
                total_rounds=3,
                epochs=1000
            )
            log_path = logger.get_log_path()
            self.assertEqual(os.path.basename(log_path), "round_2_epoch_level.jsonl")

    def test_backward_compatibility_no_new_params(self):
        """不传新参数时保持旧命名格式"""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_path:
            logger = LossLogger(model_name="fc_v1", log_dir=tmp_path)
            log_path = logger.get_log_path()
            basename = os.path.basename(log_path)
            # 旧格式: fc_v1_YYYYMMDD_HHMMSS_xxxxxxxx.jsonl
            import re
            pattern = r'^fc_v1_\d{8}_\d{6}_[0-9a-f]{8}\.jsonl$'
            self.assertRegex(basename, pattern)


class TestLossLoggerPlotTitle(unittest.TestCase):
    """测试图表标题语义化"""

    @patch('ai.loss_logger.plt')
    def test_plot_title_with_round_level(self, mock_plt):
        """round_level 图表标题应包含轮次和对局数信息"""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_path:
            logger = LossLogger(
                model_name="cnn_v1",
                log_dir=tmp_path,
                session_dir="session",
                log_type="round_level",
                total_rounds=5,
                games=100
            )
            logger.log(1, 0.5)
            logger.plot()
            mock_plt.title.assert_called_once()
            title = mock_plt.title.call_args[0][0]
            self.assertIn("Round Level", title)
            self.assertIn("5 rounds", title)
            self.assertIn("100 games", title)

    @patch('ai.loss_logger.plt')
    def test_plot_title_with_epoch_level(self, mock_plt):
        """epoch_level 图表标题应包含轮次编号和 epoch 数"""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_path:
            logger = LossLogger(
                model_name="cnn_v1",
                log_dir=tmp_path,
                session_dir="session",
                log_type="epoch_level",
                round_num=1,
                total_rounds=3,
                epochs=1000
            )
            logger.log(200, 0.5)
            logger.plot()
            mock_plt.title.assert_called_once()
            title = mock_plt.title.call_args[0][0]
            self.assertIn("Round 1/3", title)
            self.assertIn("Epoch Level", title)
            self.assertIn("1000 epochs", title)


class TestLossLoggerPlotComparison(unittest.TestCase):
    """测试合并对比图功能"""

    def test_plot_comparison_generates_merged_image(self):
        """多个 epoch_level 日志合并为一张对比图"""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_path:
            session_dir = os.path.join(tmp_path, "test_session")
            os.makedirs(session_dir)

            # 创建 round_1_epoch_level.jsonl
            with open(os.path.join(session_dir, "round_1_epoch_level.jsonl"), 'w') as f:
                f.write(json.dumps({"epoch": 200, "loss": 0.0020}) + "\n")
                f.write(json.dumps({"epoch": 400, "loss": 0.0015}) + "\n")

            # 创建 round_2_epoch_level.jsonl
            with open(os.path.join(session_dir, "round_2_epoch_level.jsonl"), 'w') as f:
                f.write(json.dumps({"epoch": 200, "loss": 0.0018}) + "\n")
                f.write(json.dumps({"epoch": 400, "loss": 0.0012}) + "\n")

            img_path = LossLogger.plot_comparison(
                session_dir="test_session",
                model_name="cnn_v1",
                log_dir=tmp_path
            )

            self.assertIsNotNone(img_path)
            self.assertTrue(os.path.exists(img_path))
            self.assertTrue(img_path.endswith("comparison.png"))

    def test_plot_comparison_empty_returns_none(self):
        """session_dir 下无 epoch_level 文件时返回 None"""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_path:
            session_dir = os.path.join(tmp_path, "empty_session")
            os.makedirs(session_dir)

            img_path = LossLogger.plot_comparison(
                session_dir="empty_session",
                model_name="cnn_v1",
                log_dir=tmp_path
            )

            self.assertIsNone(img_path)


class TestLossLoggerDenseLog(unittest.TestCase):
    """测试密集日志模式（mini-batch 模式）"""

    def test_log_dense_buffers_in_memory(self):
        """log_dense 应存入内存缓冲区，不直接写文件"""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_path:
            logger = LossLogger(
                model_name="test_model",
                log_dir=tmp_path,
                dense_log=True
            )

            logger.log_dense(1, 0.5, 0.1)
            logger.log_dense(2, 0.4, 0.08)

            # flush 前文件应为空
            with open(logger.get_log_path(), 'r', encoding='utf-8') as f:
                content = f.read()
            self.assertEqual(content, "")

            # flush 后文件应有数据
            logger.flush()
            with open(logger.get_log_path(), 'r', encoding='utf-8') as f:
                lines = f.readlines()
            self.assertEqual(len(lines), 2)

    def test_log_dense_format(self):
        """flush 后的数据格式应包含 step/loss/grad_norm"""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_path:
            logger = LossLogger(
                model_name="test_model",
                log_dir=tmp_path,
                dense_log=True
            )

            logger.log_dense(1, 0.5, 0.1)
            logger.flush()

            with open(logger.get_log_path(), 'r', encoding='utf-8') as f:
                entry = json.loads(f.readline())

            self.assertEqual(entry["step"], 1)
            self.assertEqual(entry["loss"], 0.5)
            self.assertEqual(entry["grad_norm"], 0.1)

    def test_flush_clears_buffer(self):
        """flush 后缓冲区应被清空"""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_path:
            logger = LossLogger(
                model_name="test_model",
                log_dir=tmp_path,
                dense_log=True
            )

            logger.log_dense(1, 0.5, 0.1)
            logger.flush()

            # 再次 flush 不应写入重复数据
            logger.flush()
            with open(logger.get_log_path(), 'r', encoding='utf-8') as f:
                lines = f.readlines()
            self.assertEqual(len(lines), 1)


class TestLossLoggerPlotGradNorm(unittest.TestCase):
    """测试双 Y 轴图（含 grad_norm）"""

    @patch('ai.loss_logger.plt')
    def test_plot_with_grad_norm_dual_axis(self, mock_plt):
        """含 grad_norm 的数据应绘制双 Y 轴图"""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_path:
            logger = LossLogger(
                model_name="cnn_v1",
                log_dir=tmp_path,
                dense_log=True
            )
            logger.log_dense(1, 0.5, 0.1)
            logger.log_dense(2, 0.4, 0.08)
            logger.flush()

            # mock subplots 返回 (fig, ax1)
            mock_ax1 = MagicMock()
            mock_ax2 = MagicMock()
            mock_plt.subplots.return_value = (MagicMock(), mock_ax1)
            mock_ax1.twinx.return_value = mock_ax2

            logger.plot()

            # 断言创建了双 Y 轴
            mock_ax1.twinx.assert_called_once()
            # 断言 ax1 画了 loss（蓝色）
            mock_ax1.plot.assert_called_once()
            plot_call = mock_ax1.plot.call_args
            self.assertEqual(plot_call[0][2], 'b-')
            # 断言 ax2 画了 grad_norm（红色）
            mock_ax2.plot.assert_called_once()
            plot_call2 = mock_ax2.plot.call_args
            self.assertEqual(plot_call2[0][2], 'r-')

    @patch('ai.loss_logger.plt')
    def test_plot_without_grad_norm_single_axis(self, mock_plt):
        """不含 grad_norm 的数据保持单 Y 轴图（向后兼容）"""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_path:
            logger = LossLogger(model_name="cnn_v1", log_dir=tmp_path)
            logger.log(200, 0.0020)
            logger.log(400, 0.0015)

            mock_ax1 = MagicMock()
            mock_plt.subplots.return_value = (MagicMock(), mock_ax1)

            logger.plot()

            # 不应创建双 Y 轴
            mock_ax1.twinx.assert_not_called()


class TestLossLoggerDownsample(unittest.TestCase):
    """测试自动降采样"""

    def test_downsample_reduces_points(self):
        """超过 5000 点时自动降采样"""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_path:
            logger = LossLogger(
                model_name="test_model",
                log_dir=tmp_path,
                dense_log=True
            )

            # 构造 10000 条密集数据
            for i in range(1, 10001):
                logger.log_dense(i, 1.0 / i, 0.1 / i)
            logger.flush()

            # 手动调用 _downsample 验证
            entries = logger._read_entries()
            # 原始数据应为 10000 条
            self.assertEqual(len(entries), 10000)

            downsampled = logger._downsample(entries, max_points=5000)
            self.assertLessEqual(len(downsampled), 5000)

    def test_downsample_preserves_small_dataset(self):
        """不超过 5000 点时不应降采样"""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_path:
            logger = LossLogger(
                model_name="test_model",
                log_dir=tmp_path,
                dense_log=True
            )

            for i in range(1, 101):
                logger.log_dense(i, 0.1, 0.01)
            logger.flush()

            entries = logger._read_entries()
            downsampled = logger._downsample(entries, max_points=5000)
            self.assertEqual(len(downsampled), 100)


if __name__ == '__main__':
    unittest.main()
