"""AI 游戏管理器模块

管理人机对战和 AI 内战流程。
重要原则：不改动 game 包，所有 AI 逻辑都在此模块中实现。
通过调用 controller 的公开接口操作游戏。
"""

import random as _stdrandom
from typing import List, NamedTuple, Optional, Tuple
from game import GomokuController
from ai.engine import AIEngine, STRATEGY_ARGMAX, STRATEGY_SAMPLING
from ai.data_recorder import DataRecorder
from ai.move_sampler import TempConfig, resolve_temp_side, should_sample


class SingleGameResult(NamedTuple):
    """单局对弈结果

    Attributes:
        winner: 获胜方（0=未结束, 1=黑, 2=白, 3=平局）
        temp_side_this_game: 该局实际采样方
            - "black" / "white"：单方采样
            - "both"：双方采样
            - None：未启用温度采样
    """
    winner: int
    temp_side_this_game: Optional[str]


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
    def _play_single_game(
        engine_black: AIEngine,
        engine_white: AIEngine,
        recorder: DataRecorder,
        game_num: int,
        name_prefix: str = "ai_vs_ai",
        temp_config: Optional[TempConfig] = None,
    ) -> SingleGameResult:
        """执行单局 AI 内战

        封装完整的单局流程：创建 controller、new_game、落子循环、
        记录每步数据、保存游戏结果。

        Args:
            engine_black: 执黑 AI 引擎
            engine_white: 执白 AI 引擎
            recorder: 数据记录器
            game_num: 当前局号（用于存档命名和日志）
            name_prefix: 存档名称前缀（"ai_vs_ai" 或 "ai_vs_ai_rnd"）
            temp_config: 可选的温度采样配置。None 表示全程 argmax；
                非 None 时按 move_count / temp_side 规则切换到 sampling。

        Returns:
            SingleGameResult: 包含 winner 和 temp_side_this_game 的命名元组
        """
        import time

        controller = GomokuController()

        # 新建游戏，使用唯一名称避免冲突（毫秒时间戳保证唯一性）
        timestamp = int(time.time() * 1000)
        unique_name = f"{name_prefix}_{game_num}_{timestamp}"
        result = controller.new_game(unique_name)

        # 检查游戏是否创建成功
        if result.get("error"):
            raise RuntimeError(f"创建游戏失败: {result['error']}")

        # 开始记录，获取当前 game_id
        recorder.new_game()
        current_game_id = recorder.current_game_id

        # 温度采样编排：按局初始化 rng 和启用方
        rng: Optional[_stdrandom.Random] = None
        temp_side_this_game: Optional[str] = None
        if temp_config is not None:
            if temp_config.seed is not None:
                # 以 seed + game_num 派生按局独立的 rng，保证可复现
                rng = _stdrandom.Random(temp_config.seed + game_num)
            else:
                rng = _stdrandom.Random()
            temp_side_this_game = resolve_temp_side(temp_config.temp_side, rng)

        # 游戏循环：交替落子直到游戏结束
        move_count = 0
        while not controller.is_over():
            state = controller.get_state()
            board = state["board"]
            player = state["current_player"]

            # 根据当前落子方选择对应引擎
            engine = engine_black if player == 1 else engine_white

            # 获取 AI 输入输出（用于记录和落子）
            # 性能优化：只调用一次 model.predict，结果同时用于记录和落子
            board_vector = AIEngine.board_to_vector(board, player)
            probs = engine.model.predict(board_vector)

            # 判定本步是否启用温度采样
            use_temp = (
                temp_config is not None
                and should_sample(
                    move_count, player, temp_config, temp_side_this_game
                )
            )

            try:
                if use_temp:
                    x, y = engine.get_move(
                        board, player, probs=probs,
                        strategy=STRATEGY_SAMPLING,
                        temperature=temp_config.temperature,
                        top_k=temp_config.top_k,
                        rng=rng,
                    )
                else:
                    x, y = engine.get_move(board, player, probs=probs)
            except ValueError as e:
                # 棋盘已满，无法落子，结束当前对局
                print(f"AI对局 {game_num} 棋盘已满: {e}")
                break

            # 记录数据
            # 第四十一次追加：按策略传递温度采样信息
            if use_temp and temp_config is not None:
                # 构建 225 维 sampling_probs：对全部 225 个位置统一做温度变换。
                # 不区分空位/已落子位——sampling_probs 仅用于可视化，
                # 对全部位置一视同仁可确保 T=1.0 时与 AI output 归一化尺度一致，
                # T≠1.0 时也能自洽地反映温度变换效果。
                sampling_probs = [0.0] * 225
                for i in range(225):
                    p = probs[i]
                    sampling_probs[i] = (
                        (p ** (1.0 / temp_config.temperature)) if p > 0 else 0.0
                    )
                recorder.record_move(
                    current_game_id, board_vector, probs, (x, y),
                    strategy=STRATEGY_SAMPLING,
                    temperature=temp_config.temperature,
                    top_k=temp_config.top_k,
                    sampling_probs=sampling_probs,
                )
            else:
                recorder.record_move(
                    current_game_id, board_vector, probs, (x, y),
                    strategy=STRATEGY_ARGMAX,
                )

            # 执行落子
            controller.make_move(x, y)

            move_count += 1

        # 保存结果：获取获胜方和终局棋盘
        winner = controller.get_winner()
        end_state = controller.get_state()
        end_board = AIEngine.board_to_vector(end_state["board"], 0)
        # 始终传入 temp_side_this_game（含 None 和 "both"）
        recorder.save_game_result(current_game_id, winner, end_board,
                                  temp_side_this_game=temp_side_this_game)

        return SingleGameResult(winner=winner, temp_side_this_game=temp_side_this_game)

    @staticmethod
    def start_ai_vs_ai(
        engine1: AIEngine,
        engine2: AIEngine,
        num_games: int,
        recorder: DataRecorder,
        project_root: str = None,
        on_progress=None,
        temp_config: Optional[TempConfig] = None,
    ) -> List[dict]:
        """执行 AI 内战
    
        engine1 执黑，engine2 执白。
        每局游戏独立创建 controller，数据通过 recorder 记录。
    
        Args:
            engine1: 执黑 AI 引擎
            engine2: 执白 AI 引擎
            num_games: 对战局数
            recorder: 数据记录器
            project_root: 项目根目录（保留兼容，内部不使用）
            on_progress: 可选进度回调，每局完成后调用，参数为 (current: int, total: int)
            temp_config: 可选温度采样配置；None 时保持原有 argmax 行为
    
        Returns:
            对战结果列表，每项为 {"game": 局数, "winner": 获胜方, "temp_side_this_game": 采样方}
        """
        results = []
    
        for game_num in range(1, num_games + 1):
            # 固定执方：engine1 执黑，engine2 执白
            result = AIGameManager._play_single_game(
                engine1, engine2, recorder, game_num,
                name_prefix="ai_vs_ai",
                temp_config=temp_config,
            )
            results.append({
                "game": game_num,
                "winner": result.winner,
                "temp_side_this_game": result.temp_side_this_game,
            })
    
            # 跨局进度回调
            if on_progress is not None:
                on_progress(game_num, num_games)
    
        return results
    
    @staticmethod
    def start_ai_vs_ai_random(
        engine1: AIEngine,
        engine2: AIEngine,
        num_games: int,
        recorder: DataRecorder,
        temp_config: Optional[TempConfig] = None,
    ) -> List[dict]:
        """执行 AI 内战（每局执方随机）
    
        每局随机决定 engine1/engine2 各执黑棋还是白棋。
        数据通过 recorder 记录。
    
        Args:
            engine1: AI 引擎 1
            engine2: AI 引擎 2
            num_games: 对战局数
            recorder: 数据记录器
            temp_config: 可选温度采样配置；None 时保持原有 argmax 行为
    
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
        results = []
    
        for game_num in range(1, num_games + 1):
            # 随机决定 engine1 执黑还是白
            ai1_color = _random.choice([1, 2])
            ai2_color = 3 - ai1_color  # 互为黑白
    
            # 根据颜色分配引擎：黑方先手
            engine_black = engine1 if ai1_color == 1 else engine2
            engine_white = engine2 if ai1_color == 1 else engine1
    
            result = AIGameManager._play_single_game(
                engine_black, engine_white, recorder, game_num,
                name_prefix="ai_vs_ai_rnd",
                temp_config=temp_config,
            )
    
            # 判断各方是否获胜（平局时两方都不算赢）
            ai1_won = (result.winner == ai1_color)
            ai2_won = (result.winner == ai2_color)
    
            results.append({
                "game": game_num,
                "winner": result.winner,
                "temp_side_this_game": result.temp_side_this_game,
                "ai1_color": ai1_color,
                "ai2_color": ai2_color,
                "ai1_won": ai1_won,
                "ai2_won": ai2_won,
            })
    
        return results
