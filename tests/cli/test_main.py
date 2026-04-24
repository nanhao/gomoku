"""CLI 主程序命令解析测试

测试 cli/main.py 的命令解析和执行逻辑。
使用 mock 模拟用户输入和 controller 调用。
"""

import sys
import os
import tempfile
import shutil
from unittest.mock import MagicMock, patch, call

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from cli.main import parse_command, handle_new, handle_put, handle_list
from cli.main import handle_print, handle_show, handle_short, handle_help
from game import GomokuController
from game.save import GomokuSave


class TestParseCommand:
    """测试 parse_command 命令解析"""

    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp(prefix="test_cli_")
        save_dir = os.path.join(self.temp_dir, "saves")
        self.controller = GomokuController()
        self.controller.save = GomokuSave(save_dir)

    def teardown_method(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_quit_returns_quit(self):
        """quit 命令返回 'quit'"""
        assert parse_command(self.controller, "quit") == "quit"

    def test_exit_returns_quit(self):
        """exit 命令返回 'quit'"""
        assert parse_command(self.controller, "exit") == "quit"

    def test_empty_input_returns_none(self):
        """空输入返回 None"""
        assert parse_command(self.controller, "") is None

    def test_whitespace_input_returns_none(self):
        """纯空格输入返回 None"""
        assert parse_command(self.controller, "   ") is None

    def test_unknown_command_returns_none(self):
        """未知命令返回 None（并打印提示）"""
        with patch('builtins.print') as mock_print:
            result = parse_command(self.controller, "xyz")
        assert result is None

    def test_new_command_dispatches(self):
        """new 命令正确分派到 handle_new"""
        with patch('cli.main.handle_new') as mock_new:
            parse_command(self.controller, "new test_game")
            mock_new.assert_called_once()

    def test_put_command_dispatches(self):
        """put 命令正确分派到 handle_put"""
        with patch('cli.main.handle_put') as mock_put:
            parse_command(self.controller, "put 7 7")
            mock_put.assert_called_once()

    def test_list_command_dispatches(self):
        """list 命令正确分派到 handle_list"""
        with patch('cli.main.handle_list') as mock_list:
            parse_command(self.controller, "list")
            mock_list.assert_called_once()

    def test_print_command_dispatches(self):
        """print 命令正确分派到 handle_print"""
        with patch('cli.main.handle_print') as mock_print:
            parse_command(self.controller, "print")
            mock_print.assert_called_once()

    def test_show_command_dispatches(self):
        """show 命令正确分派到 handle_show"""
        with patch('cli.main.handle_show') as mock_show:
            parse_command(self.controller, "show mygame")
            mock_show.assert_called_once()

    def test_short_command_dispatches(self):
        """short 命令正确分派到 handle_short"""
        with patch('cli.main.handle_short') as mock_short:
            parse_command(self.controller, "short mygame")
            mock_short.assert_called_once()

    def test_help_command_dispatches(self):
        """help 命令正确分派到 handle_help"""
        with patch('cli.main.handle_help') as mock_help:
            parse_command(self.controller, "help")
            mock_help.assert_called_once()

    def test_case_insensitive(self):
        """命令不区分大小写"""
        assert parse_command(self.controller, "QUIT") == "quit"
        assert parse_command(self.controller, "Exit") == "quit"


class TestHandleNew:
    """测试 handle_new 命令"""

    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp(prefix="test_cli_new_")
        save_dir = os.path.join(self.temp_dir, "saves")
        self.controller = GomokuController()
        self.controller.save = GomokuSave(save_dir)

    def teardown_method(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_new_game_success(self):
        """新建游戏成功，打印相关信息"""
        with patch('builtins.print') as mock_print:
            handle_new(self.controller, ["测试游戏"])

        assert self.controller.game is not None
        printed = "\n".join(str(c) for c in mock_print.call_args_list)
        assert "游戏已创建" in printed

    def test_new_game_no_args(self):
        """不传存档名称时使用默认名称"""
        with patch('builtins.print') as mock_print:
            handle_new(self.controller, [])

        assert self.controller.game is not None

    def test_new_game_duplicate_name(self):
        """重复名称应打印错误"""
        handle_new(self.controller, ["dup_game"])
        with patch('builtins.print') as mock_print:
            handle_new(self.controller, ["dup_game"])

        printed = "\n".join(str(c) for c in mock_print.call_args_list)
        assert "错误" in printed


class TestHandlePut:
    """测试 handle_put 命令"""

    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp(prefix="test_cli_put_")
        save_dir = os.path.join(self.temp_dir, "saves")
        self.controller = GomokuController()
        self.controller.save = GomokuSave(save_dir)

    def teardown_method(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_put_valid_move(self):
        """有效落子应打印落子信息"""
        self.controller.new_game("put_test")
        with patch('builtins.print') as mock_print:
            handle_put(self.controller, ["7", "7"])

        printed = "\n".join(str(c) for c in mock_print.call_args_list)
        assert "落子" in printed

    def test_put_wrong_arg_count(self):
        """参数数量不对应打印错误"""
        with patch('builtins.print') as mock_print:
            handle_put(self.controller, ["7"])

        printed = "\n".join(str(c) for c in mock_print.call_args_list)
        assert "用法" in printed

    def test_put_non_integer_coords(self):
        """非整数坐标应打印错误"""
        with patch('builtins.print') as mock_print:
            handle_put(self.controller, ["abc", "def"])

        printed = "\n".join(str(c) for c in mock_print.call_args_list)
        assert "整数" in printed


class TestHandleList:
    """测试 handle_list 命令"""

    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp(prefix="test_cli_list_")
        save_dir = os.path.join(self.temp_dir, "saves")
        self.controller = GomokuController()
        self.controller.save = GomokuSave(save_dir)

    def teardown_method(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_list_empty(self):
        """空存档列表应打印提示"""
        with patch('builtins.print') as mock_print:
            handle_list(self.controller, [])

        printed = "\n".join(str(c) for c in mock_print.call_args_list)
        assert "暂无存档" in printed

    def test_list_with_saves(self):
        """有存档时应列出存档名"""
        self.controller.new_game("list_test")
        with patch('builtins.print') as mock_print:
            handle_list(self.controller, [])

        printed = "\n".join(str(c) for c in mock_print.call_args_list)
        assert "list_test" in printed


class TestHandlePrint:
    """测试 handle_print 命令"""

    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp(prefix="test_cli_print_")
        save_dir = os.path.join(self.temp_dir, "saves")
        self.controller = GomokuController()
        self.controller.save = GomokuSave(save_dir)

    def teardown_method(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_print_no_game(self):
        """无游戏时应打印错误"""
        with patch('builtins.print') as mock_print:
            handle_print(self.controller, [])

        printed = "\n".join(str(c) for c in mock_print.call_args_list)
        assert "请先创建或加载游戏" in printed

    def test_print_with_game(self):
        """有游戏时应打印当前存档名"""
        self.controller.new_game("print_test")
        with patch('builtins.print') as mock_print:
            handle_print(self.controller, [])

        printed = "\n".join(str(c) for c in mock_print.call_args_list)
        assert "print_test" in printed


class TestHandleShow:
    """测试 handle_show 命令"""

    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp(prefix="test_cli_show_")
        save_dir = os.path.join(self.temp_dir, "saves")
        self.controller = GomokuController()
        self.controller.save = GomokuSave(save_dir)

    def teardown_method(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_show_no_args(self):
        """不传存档名称应打印错误"""
        with patch('builtins.print') as mock_print:
            handle_show(self.controller, [])

        printed = "\n".join(str(c) for c in mock_print.call_args_list)
        assert "用法" in printed

    def test_show_nonexistent_save(self):
        """加载不存在的存档应打印错误"""
        with patch('builtins.print') as mock_print:
            handle_show(self.controller, ["nonexistent"])

        printed = "\n".join(str(c) for c in mock_print.call_args_list)
        assert "错误" in printed


class TestHandleShort:
    """测试 handle_short 命令"""

    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp(prefix="test_cli_short_")
        save_dir = os.path.join(self.temp_dir, "saves")
        self.controller = GomokuController()
        self.controller.save = GomokuSave(save_dir)

    def teardown_method(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_short_no_args(self):
        """不传存档名称应打印错误"""
        with patch('builtins.print') as mock_print:
            handle_short(self.controller, [])

        printed = "\n".join(str(c) for c in mock_print.call_args_list)
        assert "用法" in printed

    def test_short_existing_save(self):
        """加载已存在存档应打印 JSON"""
        self.controller.new_game("short_test")
        with patch('builtins.print') as mock_print:
            handle_short(self.controller, ["short_test"])

        # 应输出 JSON 内容（至少有打印调用）
        assert len(mock_print.call_args_list) > 0


class TestHandleHelp:
    """测试 handle_help 命令"""

    def test_help_prints_commands(self):
        """help 命令应打印所有命令列表"""
        controller = MagicMock()
        with patch('builtins.print') as mock_print:
            handle_help(controller, [])

        printed = "\n".join(str(c) for c in mock_print.call_args_list)
        assert "new" in printed
        assert "put" in printed
        assert "list" in printed
        assert "quit" in printed


class TestMainREPL:
    """测试 main() REPL 循环"""

    def test_main_quit_on_quit(self):
        """输入 quit 时主循环退出"""
        from cli.main import main
        with patch('builtins.input', side_effect=["quit"]):
            with patch('builtins.print'):
                main()

    def test_main_exit_on_exit(self):
        """输入 exit 时主循环退出"""
        from cli.main import main
        with patch('builtins.input', side_effect=["exit"]):
            with patch('builtins.print'):
                main()

    def test_main_keyboard_interrupt(self):
        """KeyboardInterrupt 应正常退出"""
        from cli.main import main
        with patch('builtins.input', side_effect=KeyboardInterrupt):
            with patch('builtins.print'):
                main()

    def test_main_eof(self):
        """EOFError 应正常退出"""
        from cli.main import main
        with patch('builtins.input', side_effect=EOFError):
            with patch('builtins.print'):
                main()

    def test_main_empty_input_continues(self):
        """空输入不应退出，继续循环"""
        from cli.main import main
        with patch('builtins.input', side_effect=["", "quit"]):
            with patch('builtins.print'):
                main()
