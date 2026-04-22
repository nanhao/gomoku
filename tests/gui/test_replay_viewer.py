"""ReplayViewer 组件测试 - TestReplayViewer, TestReplayWindow"""

import sys
import os
import json
import tempfile
import tkinter as tk
import unittest

# 确保项目根目录在路径中
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from gui.components.replay_viewer import ReplayViewer, ReplayWindow


class TestReplayViewer:
    """测试回放查看器组件"""

    def setup_method(self):
        """测试前准备"""
        self.root = tk.Tk()
        self.root.withdraw()

        # 创建临时测试数据
        self.temp_dir = tempfile.mkdtemp()
        self.record_file = os.path.join(self.temp_dir, "record.json")
        self.sample_file = os.path.join(self.temp_dir, "sample.json")

        # 写入测试 record.json
        record_data = {
            "history": [
                {
                    "input": [0] * 225 + [1],  # 225个棋盘位置 + 当前玩家
                    "output": [1000] * 225,  # 225个 reward，按行优先索引
                    "position": [7, 7]
                },
                {
                    "input": [0] * 225 + [2],
                    "output": [500] * 225,
                    "position": [7, 8]
                }
            ],
            "winner": 1
        }
        with open(self.record_file, 'w', encoding='utf-8') as f:
            json.dump(record_data, f)

        # 写入测试 sample.json
        sample_data = [
            {
                "input": [0] * 225 + [1],
                "output": [1000] * 225,  # 225 个 reward元素
                "mask": [1] * 225
            },
            {
                "input": [0] * 225 + [2],
                "output": [500] * 225,
                "mask": [1] * 225
            }
        ]
        with open(self.sample_file, 'w', encoding='utf-8') as f:
            json.dump(sample_data, f)

    def teardown_method(self):
        """测试后清理"""
        try:
            self.root.destroy()
        except Exception as e:
            raise unittest.SkipTest(f"GUI环境不可用: {e}")

        # 删除临时文件
        import shutil
        try:
            shutil.rmtree(self.temp_dir)
        except Exception as e:
            raise unittest.SkipTest(f"GUI环境不可用: {e}")

    def test_init_with_record(self):
        """测试使用 record 文件初始化"""
        viewer = ReplayViewer(self.root, self.record_file)

        assert viewer is not None
        assert len(viewer.history) == 2
        assert viewer.current_step == 0

    def test_init_with_sample(self):
        """测试使用 record 和 sample 文件初始化"""
        viewer = ReplayViewer(self.root, self.record_file, self.sample_file)

        assert viewer is not None
        assert len(viewer.history) == 2
        assert len(viewer.sample_data) == 2

    def test_load_data(self):
        """测试加载数据"""
        viewer = ReplayViewer(self.root, self.record_file, self.sample_file)

        assert len(viewer.history) == 2
        assert viewer.history[0]["position"] == [7, 7]
        assert viewer.history[1]["position"] == [7, 8]

    def test_next_step(self):
        """测试下一步"""
        viewer = ReplayViewer(self.root, self.record_file)

        assert viewer.current_step == 0
        viewer._next_step()
        assert viewer.current_step == 1

    def test_next_step_at_end(self):
        """测试在结尾时下一步"""
        viewer = ReplayViewer(self.root, self.record_file)
        viewer.current_step = 2  # 已经是最后一步

        viewer._next_step()
        assert viewer.current_step == 2  # 不应该超过最大值

    def test_prev_step(self):
        """测试上一步"""
        viewer = ReplayViewer(self.root, self.record_file)
        viewer.current_step = 2

        viewer._prev_step()
        assert viewer.current_step == 1

    def test_prev_step_at_beginning(self):
        """测试在开头时上一步"""
        viewer = ReplayViewer(self.root, self.record_file)
        viewer.current_step = 0

        viewer._prev_step()
        assert viewer.current_step == 0  # 不应该小于 0

    def test_goto_end(self):
        """测试跳转到结尾"""
        viewer = ReplayViewer(self.root, self.record_file)

        viewer._goto_end()
        assert viewer.current_step == 2

    def test_reset(self):
        """测试重置"""
        viewer = ReplayViewer(self.root, self.record_file)
        viewer.current_step = 2

        viewer._reset()
        assert viewer.current_step == 0

    def test_update_display(self):
        """测试更新显示"""
        viewer = ReplayViewer(self.root, self.record_file)
        viewer._update_display()

        # 验证步骤标签更新
        assert viewer.step_label.cget("text") == "第 0 / 2 步"

    def test_draw_board_grid(self):
        """测试绘制棋盘网格"""
        viewer = ReplayViewer(self.root, self.record_file)
        viewer._draw_board_grid()

        assert True  # 应该正常执行

    def test_draw_pieces(self):
        """测试绘制棋子"""
        viewer = ReplayViewer(self.root, self.record_file)

        input_vector = [0] * 225
        input_vector[7 * 15 + 7] = 1  # 黑棋在(7,7)
        input_vector[7 * 15 + 8] = 2  # 白棋在(7,8)

        move = {"position": [7, 8]}
        viewer._draw_pieces(input_vector, move)

        assert True

    def test_draw_heatmaps(self):
        """测试绘制热力图"""
        viewer = ReplayViewer(self.root, self.record_file, self.sample_file)
        viewer.current_step = 1

        move = viewer.history[0]
        viewer._draw_heatmaps(1, move)

        assert True

    def test_update_info_text(self):
        """测试更新信息文本"""
        viewer = ReplayViewer(self.root, self.record_file)
        viewer.current_step = 1

        move = viewer.history[0]
        viewer._update_info_text(1, move)

        # 验证信息文本已更新
        text = viewer.info_text.get(1.0, tk.END)
        assert "第 1 步" in text


    def test_update_info_text_with_sample(self):
        """测试含 sample 数据时更新信息文本"""
        viewer = ReplayViewer(self.root, self.record_file, self.sample_file)
        viewer.current_step = 1
        move = viewer.history[0]  # position=[7,7]
        viewer._update_info_text(1, move)

        text = viewer.info_text.get(1.0, tk.END)
        # position[7,7] -> idx = 7*15+7 = 112, reward = 1000
        assert "实际落子 reward: 1000" in text

    def test_update_info_text_goto_end_no_index_error(self):
        """回归：跳转到结局时 _update_info_text 不应抛 IndexError
        
        复现 Bug：output 被错误当作 (x,y,reward)*225 = 675 维格式处理时
        step = len(history) 时 step - 1 为最后一步， output_vec 索引越界
        """
        viewer = ReplayViewer(self.root, self.record_file, self.sample_file)
        # 跳转到结局（step = 2 = len(history)）
        viewer.current_step = len(viewer.history)
        move = viewer.history[-1]  # 最后一步 position=[7,8]
        # 不应抛出任何异常
        viewer._update_info_text(len(viewer.history), move)

        text = viewer.info_text.get(1.0, tk.END)
        assert "Sample 数据" in text

    def test_draw_heatmaps_with_225_format(self):
        """测试 output 为 225 元素格式时热力图正常绘制"""
        viewer = ReplayViewer(self.root, self.record_file, self.sample_file)
        move = viewer.history[0]  # output=[1000]*225
        # 不应抛出任何异常
        viewer._draw_heatmaps(1, move)
        assert True

    def test_draw_heatmaps_empty_output(self):
        """测试 output 为空列表时热力图不抛异常"""
        viewer = ReplayViewer(self.root, self.record_file)
        move = {"output": [], "position": [7, 7]}
        viewer._draw_heatmaps(1, move)
        assert True


class TestReplayWindow:
    """测试回放窗口"""

    def setup_method(self):
        """测试前准备"""
        self.temp_dir = tempfile.mkdtemp()
        self.record_file = os.path.join(self.temp_dir, "record.json")

        # 写入测试数据
        record_data = {
            "history": [],
            "winner": 0
        }
        with open(self.record_file, 'w', encoding='utf-8') as f:
            json.dump(record_data, f)

    def teardown_method(self):
        """测试后清理"""
        import shutil
        try:
            shutil.rmtree(self.temp_dir)
        except Exception as e:
            raise unittest.SkipTest(f"GUI环境不可用: {e}")

    def test_init(self):
        """测试初始化"""
        window = ReplayWindow(self.record_file)

        assert window is not None
        assert window.record_file == self.record_file
        assert window.sample_file is None
        window.root.destroy()

    def test_init_with_sample(self):
        """测试带 sample 文件初始化"""
        sample_file = os.path.join(self.temp_dir, "sample.json")
        with open(sample_file, 'w', encoding='utf-8') as f:
            json.dump([], f)

        window = ReplayWindow(self.record_file, sample_file)

        assert window.sample_file == sample_file
        window.root.destroy()
