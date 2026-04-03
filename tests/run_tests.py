"""测试运行器 - 无需 pytest"""

import sys
import os
import traceback

# 获取项目根目录（tests 的父目录）
current_dir = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.dirname(current_dir)

# 将项目根目录添加到 sys.path 最前面
if parent_dir in sys.path:
    sys.path.remove(parent_dir)
sys.path.insert(0, parent_dir)

# 移除空字符串（当前目录）避免冲突
if '' in sys.path:
    sys.path.remove('')
if '.' in sys.path:
    sys.path.remove('.')

# 导入 GUI 测试类
from tests.gui.test_mode_selector import TestGameModeSelector
from tests.gui.test_game_board import TestGameBoard
from tests.gui.test_heatmap import TestHeatmapCanvas, TestHeatmapLegend
from tests.gui.test_replay_viewer import TestReplayViewer, TestReplayWindow
from tests.gui.test_replay_browser import TestRoundSelector, TestGameSelector, TestReplayBrowser
from tests.gui.test_ai_battle import TestAIBattleConfigDialog, TestAIBattleProgressWindow, TestAIBattleManager
from tests.gui.test_gomoku_gui import TestGomokuGUIOnBoardClick, TestGomokuGUIPveAiPlayer2, TestGomokuGUIPveAiPlayer1

def run_test_method(test_class, method_name):
    """运行单个测试方法"""
    try:
        instance = test_class()
        method = getattr(instance, method_name)
        
        # 如果存在 setup_method，先调用
        if hasattr(instance, 'setup_method'):
            instance.setup_method()
        
        method()
        print(f"  PASS {method_name}")
        return True
    except Exception as e:
        print(f"  FAIL {method_name}")
        print(f"    Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def run_all_tests():
    """运行所有测试"""
    print("=" * 60)
    print("运行五子棋游戏测试")
    print("=" * 60)
    
    # 导入测试类
    try:
        from tests.test_save import TestGomokuSave
        from tests.test_game import TestGomokuGame
        from tests.test_controller import TestGomokuController
    except ImportError as e:
        print(f"导入测试模块失败: {e}")
        return False
    
    # 导入 GUI 测试类
    # 确保项目根目录在 sys.path 最前面
    if parent_dir in sys.path:
        sys.path.remove(parent_dir)
    sys.path.insert(0, parent_dir)
    
    # 移除空字符串（当前目录）避免冲突
    if '' in sys.path:
        sys.path.remove('')
    if '.' in sys.path:
        sys.path.remove('.')
    
    gui_tests_available = True
    
    test_modules = [
        ("GomokuSave 测试", TestGomokuSave),
        ("GomokuGame 测试", TestGomokuGame),
        ("GomokuController 测试", TestGomokuController),
    ]
    
    # 添加 GUI 测试
    if gui_tests_available:
        test_modules.extend([
            ("GUI - GameModeSelector 测试", TestGameModeSelector),
            ("GUI - GameBoard 测试", TestGameBoard),
            ("GUI - HeatmapCanvas 测试", TestHeatmapCanvas),
            ("GUI - HeatmapLegend 测试", TestHeatmapLegend),
            ("GUI - ReplayViewer 测试", TestReplayViewer),
            ("GUI - ReplayWindow 测试", TestReplayWindow),
            ("GUI - RoundSelector 测试", TestRoundSelector),
            ("GUI - GameSelector 测试", TestGameSelector),
            ("GUI - ReplayBrowser 测试", TestReplayBrowser),
            ("GUI - AIBattleConfigDialog 测试", TestAIBattleConfigDialog),
            ("GUI - AIBattleProgressWindow 测试", TestAIBattleProgressWindow),
            ("GUI - AIBattleManager 测试", TestAIBattleManager),
            ("GUI - GomokuGUI 棋盘点击测试", TestGomokuGUIOnBoardClick),
            ("GUI - GomokuGUI 人机对战(AI执白)测试", TestGomokuGUIPveAiPlayer2),
            ("GUI - GomokuGUI 人机对战(AI执黑)测试", TestGomokuGUIPveAiPlayer1),
        ])
    
    total_tests = 0
    passed_tests = 0
    
    for module_name, test_class in test_modules:
        print(f"\n{module_name}")
        print("-" * 60)
        
        # 获取所有测试方法
        test_methods = [m for m in dir(test_class) if m.startswith('test_')]
        
        for method_name in test_methods:
            total_tests += 1
            if run_test_method(test_class, method_name):
                passed_tests += 1
    
    print("\n" + "=" * 60)
    print(f"测试结果: {passed_tests}/{total_tests} 通过")
    print("=" * 60)
    
    if passed_tests == total_tests:
        print("SUCCESS: 所有测试通过！")
        return True
    else:
        print(f"FAILED: {total_tests - passed_tests} 个测试失败")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
