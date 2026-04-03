"""
回放浏览器测试 - TestRoundSelector, TestGameSelector, TestReplayBrowser

测试回放选择器和浏览器的核心功能。
使用临时目录结构进行测试。
"""

import sys
import os
import json
import shutil
import tempfile
import tkinter as tk

# 确保项目根目录在路径中
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from gui.components.replay_browser import RoundSelector, GameSelector, ReplayBrowser


def _create_temp_ai_data():
    """创建临时 ai_data 目录结构，返回根目录路径"""
    tmp_dir = tempfile.mkdtemp()
    # 创建两个 round 目录，每个 round 下有一个 game 目录和 record.json
    for round_name in ["random_vs_random_1", "random_vs_random_2"]:
        round_dir = os.path.join(tmp_dir, round_name)
        os.makedirs(round_dir)
        for game_id in ["1"]:
            game_dir = os.path.join(round_dir, game_id)
            os.makedirs(game_dir)
            record_data = {
                "winner": 1,
                "history": [
                    {"position": [7, 7], "input": [0] * 225, "output": [0.0] * 675}
                ]
            }
            with open(os.path.join(game_dir, "record.json"), 'w', encoding='utf-8') as f:
                json.dump(record_data, f)
    return tmp_dir


class TestRoundSelector:
    """测试轮次选择器"""

    def setup_method(self):
        """创建临时数据目录"""
        self.tmp_dir = _create_temp_ai_data()

    def _cleanup(self):
        """清理临时目录"""
        try:
            shutil.rmtree(self.tmp_dir)
        except Exception:
            pass

    def _create_selector(self, on_select=None):
        """创建 RoundSelector 实例（无父窗口，使用 tk.Tk()）"""
        root = tk.Tk()
        root.withdraw()
        selector = RoundSelector.__new__(RoundSelector)
        selector.parent = None
        selector.ai_data_dir = self.tmp_dir
        selector.on_select = on_select
        selector.root = root
        return selector, root

    def test_get_rounds_returns_list(self):
        """_get_rounds 应返回列表"""
        selector, root = self._create_selector()
        rounds = selector._get_rounds()
        assert isinstance(rounds, list)
        root.destroy()
        self._cleanup()

    def test_get_rounds_finds_directories(self):
        """_get_rounds 应找到两个轮次目录"""
        selector, root = self._create_selector()
        rounds = selector._get_rounds()
        assert len(rounds) == 2
        root.destroy()
        self._cleanup()

    def test_get_rounds_has_name_path_display(self):
        """每个轮次应包含 name, path, display 字段"""
        selector, root = self._create_selector()
        rounds = selector._get_rounds()
        for r in rounds:
            assert "name" in r
            assert "path" in r
            assert "display" in r
        root.destroy()
        self._cleanup()

    def test_get_rounds_empty_dir(self):
        """空目录应返回空列表"""
        empty_dir = tempfile.mkdtemp()
        selector, root = self._create_selector()
        selector.ai_data_dir = empty_dir
        rounds = selector._get_rounds()
        assert rounds == []
        root.destroy()
        shutil.rmtree(empty_dir)
        self._cleanup()

    def test_get_rounds_nonexistent_dir(self):
        """不存在的目录应返回空列表"""
        selector, root = self._create_selector()
        selector.ai_data_dir = "/nonexistent/path"
        rounds = selector._get_rounds()
        assert rounds == []
        root.destroy()
        self._cleanup()


class TestGameSelector:
    """测试对局选择器"""

    def setup_method(self):
        """创建临时数据目录"""
        self.tmp_dir = _create_temp_ai_data()

    def _cleanup(self):
        """清理临时目录"""
        try:
            shutil.rmtree(self.tmp_dir)
        except Exception:
            pass

    def _create_selector(self, round_dir=None, on_select=None):
        """创建 GameSelector 实例"""
        root = tk.Tk()
        root.withdraw()
        if round_dir is None:
            round_dir = os.path.join(self.tmp_dir, "random_vs_random_1")
        selector = GameSelector.__new__(GameSelector)
        selector.parent = None
        selector.round_dir = round_dir
        selector.on_select = on_select
        selector.root = root
        return selector, root

    def test_get_games_returns_list(self):
        """_get_games 应返回列表"""
        selector, root = self._create_selector()
        games = selector._get_games()
        assert isinstance(games, list)
        root.destroy()
        self._cleanup()

    def test_get_games_finds_game(self):
        """_get_games 应找到一个对局"""
        selector, root = self._create_selector()
        games = selector._get_games()
        assert len(games) == 1
        root.destroy()
        self._cleanup()

    def test_get_games_has_required_fields(self):
        """每个对局应包含 id, path, record_file, display 字段"""
        selector, root = self._create_selector()
        games = selector._get_games()
        for g in games:
            assert "id" in g
            assert "path" in g
            assert "record_file" in g
            assert "display" in g
        root.destroy()
        self._cleanup()

    def test_get_games_empty_dir(self):
        """没有对局的目录应返回空列表"""
        empty_dir = tempfile.mkdtemp()
        selector, root = self._create_selector(round_dir=empty_dir)
        games = selector._get_games()
        assert games == []
        root.destroy()
        shutil.rmtree(empty_dir)
        self._cleanup()

    def test_get_games_nonexistent_dir(self):
        """不存在的目录应返回空列表"""
        selector, root = self._create_selector(round_dir="/nonexistent/path")
        games = selector._get_games()
        assert games == []
        root.destroy()
        self._cleanup()


class TestReplayBrowser:
    """测试回放浏览器（静态方法）"""

    def test_select_round_is_static(self):
        """select_round 应是静态方法"""
        assert callable(ReplayBrowser.select_round)

    def test_select_game_is_static(self):
        """select_game 应是静态方法"""
        assert callable(ReplayBrowser.select_game)

    def test_browse_is_static(self):
        """browse 应是静态方法"""
        assert callable(ReplayBrowser.browse)
