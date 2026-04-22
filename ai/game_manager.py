"""AI 游戏管理器模块

管理人机对战和 AI 内战流程。
重要原则：不改动 game 包，所有 AI 逻辑都在此模块中实现。
通过调用 controller 的公开接口操作游戏。
"""

from typing import List, Tuple, Optional
from game import GomokuController
from ai.engine import AIEngine
from ai.data_recorder import DataRecorder


class AIGameManager:
    """AI 游戏管理器
    
    管理人机对战（PvE）流程：
    - 启动游戏
    - 处理玩家落子
    - 自动触发 AI 落子
    
    重要：本类不修改 game 包，仅通过 controller 的公开接口操作游戏。
    """
    
    def __init__(self, controller: GomokuController, engine: AIEngine, ai_player: int):
        """
        初始化 AI 游戏管理器
        
        Args:
            controller: GomokuController 实例
            engine: AIEngine 实例
            ai_player: AI 执棋颜色（1=黑, 2=白）
        """
        self.controller = controller
        self.engine = engine
        self.ai_player = ai_player
    
    def start_pve_game(self, save_name: str = None) -> dict:
        """
        启动人机对战游戏
        
        如果 AI 执黑（先手），会自动执行第一步。
        
        Args:
            save_name: 存档名称，None 则使用默认名称
            
        Returns:
            controller.new_game() 或 ai_move() 的返回结果
        """
        result = self.controller.new_game(save_name)
        
        # 如果 AI 执黑，自动执行第一步
        if result.get("status") == "success" and self.ai_player == 1:
            return self.ai_move()
        
        return result
    
    def player_move(self, x: int, y: int) -> dict:
        """
        玩家落子，如轮到 AI 则自动触发
        
        流程：
        1. 执行玩家落子
        2. 检查游戏是否结束
        3. 如轮到 AI，自动执行 AI 落子
        
        Args:
            x: 落子行坐标（0-14）
            y: 落子列坐标（0-14）
            
        Returns:
            {
                "player_result": <玩家落子结果>,
                "ai_result": <AI 落子结果>  # 仅当 AI 落子时包含
            }
        """
        # 玩家落子
        player_result = self.controller.make_move(x, y)
        if "error" in player_result:
            return {"player_result": player_result}
        
        # 检查游戏是否结束
        if self.controller.is_over():
            return {"player_result": player_result}
        
        # 检查是否轮到 AI
        current_player = self.controller.get_current_player()
        if current_player != self.ai_player:
            return {"player_result": player_result}
        
        # 触发 AI 落子
        ai_result = self.ai_move()
        return {
            "player_result": player_result,
            "ai_result": ai_result
        }
    
    def ai_move(self) -> dict:
        """
        AI 执行落子
        
        流程：
        1. 获取当前棋盘状态
        2. 调用引擎计算落子位置
        3. 执行落子
        
        Returns:
            controller.make_move() 的返回结果
        """
        # 获取当前棋盘状态
        state = self.controller.get_state()
        board = state["board"]
        current_player = state["current_player"]
        
        # AI 计算落子位置
        x, y = self.engine.get_move(board, current_player)
        
        # 执行落子
        return self.controller.make_move(x, y)
    
    @staticmethod
    def start_ai_vs_ai(
        engine1: AIEngine,
        engine2: AIEngine,
        num_games: int,
        recorder: DataRecorder,
        project_root: str = None,
        on_progress=None,
    ) -> List[dict]:
        """
        执行 AI 内战
        
        engine1 执黑，engine2 执白。
        每局游戏独立创建 controller，数据通过 recorder 记录。
        
        Args:
            engine1: 执黑 AI 引擎
            engine2: 执白 AI 引擎
            num_games: 对战局数
            recorder: 数据记录器
            project_root: 项目根目录，用于创建 controller
            on_progress: 可选进度回调，每局完成后调用，参数为 (current: int, total: int)
            
        Returns:
            对战结果列表，每项为 {"game": 局数, "winner": 获胜方}
        """
        import os
        import time
        results = []
        
        for game_num in range(1, num_games + 1):
            controller = GomokuController()
            
            # 新建游戏，使用唯一名称避免冲突（使用毫秒时间戳保证唯一性）
            timestamp = int(time.time() * 1000)
            unique_name = f"ai_vs_ai_{game_num}_{timestamp}"
            result = controller.new_game(unique_name)
            
            # 检查游戏是否创建成功
            if result.get("error"):
                raise RuntimeError(f"创建游戏失败: {result['error']}")
            
            # 开始记录
            recorder.new_game()
            current_game_id = recorder.current_game_id
            
            # 游戏循环
            while not controller.is_over():
                state = controller.get_state()
                board = state["board"]
                player = state["current_player"]
                
                # 选择当前 AI
                engine = engine1 if player == 1 else engine2
                
                # 获取 AI 输入输出（用于记录）
                board_vector = AIEngine.board_to_vector(board, player)
                rewards = engine.model.predict(board_vector)
                
                # AI 落子
                x, y = engine.get_move(board, player)
                
                # 记录数据
                recorder.record_move(current_game_id, board_vector, rewards, (x, y))
                
                # 执行落子
                controller.make_move(x, y)
            
            # 保存结果
            winner = controller.get_winner()
            end_state = controller.get_state()
            end_board = AIEngine.board_to_vector(end_state["board"], 0)
            recorder.save_game_result(current_game_id, winner, end_board)
            
            results.append({
                "game": game_num,
                "winner": winner
            })

            if on_progress is not None:
                on_progress(game_num, num_games)
        
        return results

    @staticmethod
    def start_ai_vs_ai_random(
        engine1: AIEngine,
        engine2: AIEngine,
        num_games: int,
        recorder: DataRecorder
    ) -> List[dict]:
        """
        执行 AI 内战（每局执方随机）
        
        每局随机决定 engine1/engine2 各执黑棋还是白棋。
        数据通过 recorder 记录。
        
        Args:
            engine1: AI 引擎 1
            engine2: AI 引擎 2
            num_games: 对战局数
            recorder: 数据记录器
            
        Returns:
            对战结果列表，每项格式：
            {
                "game": 局数（从1开始）,
                "winner": 获胜方（1=黑, 2=白, 3=平局）,
                "ai1_color": engine1执棋颜色（1=黑, 2=白）,
                "ai2_color": engine2执棋颜色（1=黑, 2=白）,
                "ai1_won": bool,
                "ai2_won": bool
            }
        """
        import random as _random
        import time
        results = []
        
        for game_num in range(1, num_games + 1):
            # 随机决定 engine1 执黑还是白
            ai1_color = _random.choice([1, 2])
            ai2_color = 3 - ai1_color  # 互为黑白
            
            # 根据颜色分配引擎：黑方先手
            engine_black = engine1 if ai1_color == 1 else engine2
            engine_white = engine2 if ai1_color == 1 else engine1
            
            controller = GomokuController()
            
            # 新建游戏，使用唯一名称
            timestamp = int(time.time() * 1000)
            unique_name = f"ai_vs_ai_rnd_{game_num}_{timestamp}"
            result = controller.new_game(unique_name)
            
            if result.get("error"):
                raise RuntimeError(f"创建游戏失败: {result['error']}")
            
            # 开始记录
            recorder.new_game()
            current_game_id = recorder.current_game_id
            
            # 游戏循环
            while not controller.is_over():
                state = controller.get_state()
                board = state["board"]
                player = state["current_player"]
                
                # 根据当前落子方选择对应引擎
                engine = engine_black if player == 1 else engine_white
                
                # 获取 AI 输入输出（用于记录）
                board_vector = AIEngine.board_to_vector(board, player)
                rewards = engine.model.predict(board_vector)
                
                # AI 落子
                x, y = engine.get_move(board, player)
                
                # 记录数据
                recorder.record_move(current_game_id, board_vector, rewards, (x, y))
                
                # 执行落子
                controller.make_move(x, y)
            
            # 保存结果
            winner = controller.get_winner()
            end_state = controller.get_state()
            end_board = AIEngine.board_to_vector(end_state["board"], 0)
            recorder.save_game_result(current_game_id, winner, end_board)
            
            # 判断各方是否获胜（平局时两方都不算赢）
            ai1_won = (winner == ai1_color)
            ai2_won = (winner == ai2_color)
            
            results.append({
                "game": game_num,
                "winner": winner,
                "ai1_color": ai1_color,
                "ai2_color": ai2_color,
                "ai1_won": ai1_won,
                "ai2_won": ai2_won
            })
        
        return results
