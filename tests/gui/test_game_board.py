"""GameBoard 组件测试 - TestGameBoard"""

import sys
import os
import tkinter as tk

# 确保项目根目录在路径中
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from gui.components.game_board import GameBoard


class TestGameBoard:
    """测试棋盘组件"""

    def setup_method(self):
        """测试前准备"""
        self.root = tk.Tk()
        self.root.withdraw()  # 隐藏窗口

    def teardown_method(self):
        """测试后清理"""
        try:
            self.root.destroy()
        except Exception:
            pass

    def test_init(self):
        """测试初始化"""
        board = GameBoard(self.root)
        assert board is not None
        assert board.BOARD_SIZE == 15
        assert board.CELL_SIZE == 42
        assert board.BOARD_MARGIN == 50

    def test_init_with_callback(self):
        """测试带回调函数初始化"""
        callback_called = [False]

        def on_click(row, col):
            callback_called[0] = True

        board = GameBoard(self.root, on_click_callback=on_click)
        assert board.on_click_callback is not None

    def test_draw_board(self):
        """测试绘制棋盘"""
        board = GameBoard(self.root)
        board.draw_board()

        # 验证画布存在
        assert board.canvas is not None

    def test_draw_pieces_empty(self):
        """测试绘制空棋盘"""
        board = GameBoard(self.root)
        board_state = [[0] * 15 for _ in range(15)]

        board.draw_pieces(board_state)
        assert True

    def test_draw_pieces_with_black(self):
        """测试绘制黑棋"""
        board = GameBoard(self.root)
        board_state = [[0] * 15 for _ in range(15)]
        board_state[7][7] = 1  # 黑棋在中心

        board.draw_pieces(board_state)
        assert True

    def test_draw_pieces_with_white(self):
        """测试绘制白棋"""
        board = GameBoard(self.root)
        board_state = [[0] * 15 for _ in range(15)]
        board_state[7][7] = 2  # 白棋在中心

        board.draw_pieces(board_state)
        assert True

    def test_draw_pieces_with_last_move(self):
        """测试绘制最后落子标记"""
        board = GameBoard(self.root)
        board_state = [[0] * 15 for _ in range(15)]
        board_state[7][7] = 1

        board.draw_pieces(board_state, last_move=(7, 7))
        assert True

    def test_draw_pieces_multiple(self):
        """测试绘制多个棋子"""
        board = GameBoard(self.root)
        board_state = [[0] * 15 for _ in range(15)]
        board_state[7][7] = 1  # 黑
        board_state[7][8] = 2  # 白
        board_state[8][7] = 1  # 黑

        board.draw_pieces(board_state, last_move=(8, 7))
        assert True

    def test_constants(self):
        """测试常量定义"""
        board = GameBoard(self.root)

        assert hasattr(board, 'COLOR_BOARD')
        assert hasattr(board, 'COLOR_LINE')
        assert hasattr(board, 'COLOR_BLACK')
        assert hasattr(board, 'COLOR_WHITE')
        assert hasattr(board, 'COLOR_LAST_MOVE')

    def test_get_canvas(self):
        """测试获取画布"""
        board = GameBoard(self.root)
        canvas = board.get_canvas()

        assert canvas is not None
        assert isinstance(canvas, tk.Canvas)

    def test_canvas_size(self):
        """测试画布尺寸计算"""
        board = GameBoard(self.root)
        expected_size = board.BOARD_MARGIN * 2 + board.CELL_SIZE * (board.BOARD_SIZE - 1)

        assert board.canvas_size == expected_size

    def test_click_callback(self):
        """测试点击回调"""
        clicked_pos = [None]

        def on_click(row, col):
            clicked_pos[0] = (row, col)

        board = GameBoard(self.root, on_click_callback=on_click)

        # 模拟点击中心位置
        center_x = board.BOARD_MARGIN + 7 * board.CELL_SIZE
        center_y = board.BOARD_MARGIN + 7 * board.CELL_SIZE

        event = tk.Event()
        event.x = center_x
        event.y = center_y

        board._on_click(event)

        assert clicked_pos[0] is not None
        assert clicked_pos[0] == (7, 7)

    def test_click_out_of_bounds(self):
        """测试点击边界外不触发回调"""
        clicked_pos = [None]

        def on_click(row, col):
            clicked_pos[0] = (row, col)

        board = GameBoard(self.root, on_click_callback=on_click)

        # 点击右下角边界外（第 15 格以外）
        # BOARD_MARGIN=50, CELL_SIZE=42
        # 要让 col >= 15：(event.x - 50 + 21) / 42 >= 15
        # event.x >= 50 - 21 + 15*42 = 659
        event = tk.Event()
        event.x = 700
        event.y = 700

        board._on_click(event)

        # 边界外点击不应该触发回调
        assert clicked_pos[0] is None
