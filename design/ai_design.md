# AI 系统设计文档

## 1. 概述

本文档描述五子棋 AI 系统的设计，包括 AI 引擎、模型接口、游戏模式扩展和数据记录。

## 2. 目录结构

```
gomoku/
├── ai/                          # AI 模块根目录
│   ├── __init__.py              # 模块初始化，导出主要类
│   ├── engine.py                # AI 引擎
│   ├── base_model.py            # 模型基类/接口
│   ├── random/                  # Random AI 模型
│   │   ├── __init__.py
│   │   └── model.py             # RandomModel 实现
│   └── data_recorder.py         # AI 对战数据记录器
├── ai_data/                     # AI 对战数据存储目录（运行时创建）
├── game/
│   ├── controller.py            # 需要扩展支持 AI 模式
│   └── ...
└── gui/
    └── main.py                  # 需要扩展启动界面
```

## 3. 类设计

### 3.1 BaseModel（模型基类）

**文件**: `ai/base_model.py`

**职责**: 定义所有 AI 模型必须实现的接口

**公共方法**:

| 方法 | 参数 | 返回值 | 说明 |
|------|------|--------|------|
| `predict` | `board_vector: List[int]` | `List[int]` | 接收 226 维向量，返回 225 维 reward 列表 |
| `name` | 无 | `str` | 返回模型名称 |

**代码框架**:
```python
from abc import ABC, abstractmethod
from typing import List


class BaseModel(ABC):
    """AI 模型基类"""
    
    @abstractmethod
    def predict(self, board_vector: List[int]) -> List[int]:
        """
        预测每个落子位置的 reward
        
        Args:
            board_vector: 226 维向量，前 225 位是棋盘（按行展开），最后 1 位是当前落子方
            
        Returns:
            225 维列表，格式为 [reward(0,0), reward(0,1), ..., reward(14,14)]
            元素索引 i 对应棋盘位置 (i//15, i%15)，reward 范围：-1000000 <= reward <= 1000000
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """模型名称"""
        pass
```

### 3.2 RandomModel（随机 AI 模型）

**文件**: `ai/random/model.py`

**职责**: 第一版 AI，所有位置返回 reward=0，使引擎随机选择空位

**公共方法**: 实现 BaseModel 接口

**代码框架**:
```python
from typing import List
from ai.base_model import BaseModel


class RandomModel(BaseModel):
    """随机 AI 模型 - 所有位置返回相同 reward，实现随机落子"""
    
    @property
    def name(self) -> str:
        return "random"
    
    def predict(self, board_vector: List[int]) -> List[int]:
        """
        无论输入如何，返回所有位置 reward=0
        
        Returns:
            225 维列表，所有 reward 为 0，索引 i 对应位置 (i//15, i%15)
        """
        return [0] * 225
```

### 3.3 AIEngine（AI 引擎）

**文件**: `ai/engine.py`

**职责**: 连接 UI 和模型，处理输入输出，选择落子位置

**公共方法**:

| 方法 | 参数 | 返回值 | 说明 |
|------|------|--------|------|
| `__init__` | `model: BaseModel` | - | 初始化引擎，绑定模型 |
| `get_move` | `board: List[List[int]], current_player: int` | `Tuple[int, int]` | 获取 AI 落子位置 |
| `board_to_vector` | `board: List[List[int]], current_player: int` | `List[int]` | 将棋盘转换为 226 维向量（静态方法） |

**代码框架**:
```python
import random
from typing import List, Tuple
from ai.base_model import BaseModel


class AIEngine:
    """AI 引擎 - 连接 UI 和 AI 模型"""
    
    def __init__(self, model: BaseModel):
        """
        初始化 AI 引擎
        
        Args:
            model: AI 模型实例
        """
        self.model = model
    
    def get_move(self, board: List[List[int]], current_player: int) -> Tuple[int, int]:
        """
        获取 AI 落子位置
        
        Args:
            board: 15x15 棋盘，0=空, 1=黑, 2=白
            current_player: 当前落子方，1=黑, 2=白
            
        Returns:
            (x, y) 落子位置
            
        Raises:
            ValueError: 棋盘已满
        """
        # 1. 转换棋盘为向量
        board_vector = self.board_to_vector(board, current_player)
        
        # 2. 获取模型预测
        rewards = self.model.predict(board_vector)
        
        # 3. 解析 225 维 reward 列表，索引 i 对应位置 (i//15, i%15)
        position_rewards = []
        for i in range(225):
            x = i // 15
            y = i % 15
            position_rewards.append((x, y, rewards[i]))
        
        # 4. 过滤掉已有棋子的位置
        empty_positions = [(x, y, r) for x, y, r in position_rewards if board[x][y] == 0]
        
        if not empty_positions:
            raise ValueError("棋盘已满")
        
        # 5. 找出最大 reward
        max_reward = max(r for _, _, r in empty_positions)
        
        # 6. 找出所有 reward 最大的位置
        best_positions = [(x, y) for x, y, r in empty_positions if r == max_reward]
        
        # 7. 随机选择一个
        return random.choice(best_positions)
    
    @staticmethod
    def board_to_vector(board: List[List[int]], current_player: int) -> List[int]:
        """
        将棋盘转换为 226 维向量
        
        Args:
            board: 15x15 棋盘
            current_player: 当前落子方
            
        Returns:
            226 维向量
        """
        vector = []
        for row in board:
            vector.extend(row)
        vector.append(current_player)
        return vector
```

### 3.4 DataRecorder（数据记录器）

**文件**: `ai/data_recorder.py`

**职责**: 记录 AI 对战的每一轮数据

**公共方法**:

| 方法 | 参数 | 返回值 | 说明 |
|------|------|--------|------|
| `__init__` | `ai_name: str, ai_data_dir: str = "ai_data"` | - | 初始化记录器，创建轮次目录 |
| `new_game` | 无 | `int` | 开始新一局游戏，返回局数编号 |
| `record_move` | `game_id: int, input_vector: List[int], output_vector: List[int], position: Tuple[int, int]` | - | 记录一次落子 |
| `save_game_result` | `game_id: int, winner: int, end_board: List[int]` | - | 保存游戏结果 |

**代码框架**:
```python
import os
import json
from typing import List, Tuple
from datetime import datetime


class DataRecorder:
    """AI 对战数据记录器"""
    
    def __init__(self, ai_name: str, ai_data_dir: str = "ai_data"):
        """
        初始化数据记录器
        
        Args:
            ai_name: AI 名称，用于命名轮次目录
            ai_data_dir: 数据存储根目录
        """
        self.ai_name = ai_name
        self.ai_data_dir = ai_data_dir
        self.round_dir = self._create_round_dir()
        self.current_game_id = 0
        self.current_game_data = None
    
    def _create_round_dir(self) -> str:
        """创建轮次目录，返回路径"""
        # 查找下一个可用的轮次编号
        round_num = 1
        while True:
            dir_name = f"{self.ai_name}_{round_num}"
            dir_path = os.path.join(self.ai_data_dir, dir_name)
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
                return dir_path
            round_num += 1
    
    def new_game(self) -> int:
        """
        开始新一局游戏
        
        Returns:
            游戏编号（从 1 开始）
        """
        self.current_game_id += 1
        self.current_game_data = {
            "history": [],
            "winner": 0,
            "end_board": []
        }
        return self.current_game_id
    
    def record_move(self, game_id: int, input_vector: List[int], 
                    output_vector: List[int], position: Tuple[int, int]):
        """
        记录一次落子
        
        Args:
            game_id: 游戏编号
            input_vector: 226 维输入向量
            output_vector: 675 维输出向量
            position: 落子位置 (x, y)
        """
        if game_id != self.current_game_id:
            raise ValueError(f"游戏编号不匹配: {game_id} != {self.current_game_id}")
        
        self.current_game_data["history"].append({
            "input": input_vector,
            "output": output_vector,
            "position": list(position)
        })
    
    def save_game_result(self, game_id: int, winner: int, end_board: List[int]):
        """
        保存游戏结果到文件
        
        Args:
            game_id: 游戏编号
            winner: 获胜方（1=黑, 2=白, 3=平局, 0=未结束）
            end_board: 最终棋盘状态（226 维向量）
        """
        if game_id != self.current_game_id:
            raise ValueError(f"游戏编号不匹配: {game_id} != {self.current_game_id}")
        
        self.current_game_data["winner"] = winner
        self.current_game_data["end_board"] = end_board
        
        # 创建局数子目录
        game_dir = os.path.join(self.round_dir, str(game_id))
        os.makedirs(game_dir, exist_ok=True)
        
        # 保存到文件
        file_path = os.path.join(game_dir, "record.json")
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.current_game_data, f, ensure_ascii=False, indent=2)
```

### 3.5 AIGameManager（AI 游戏管理器）

**文件**: `ai/game_manager.py`

**重要原则**: **不改动 game 包**，所有 AI 逻辑都在 ai 包中实现

**职责**: 管理 AI 对战流程，通过调用 controller 的公开接口实现

**公共方法**:

| 方法 | 参数 | 返回值 | 说明 |
|------|------|--------|------|
| `__init__` | `controller: GomokuController, engine: AIEngine, ai_player: int` | - | 初始化，绑定 controller 和引擎 |
| `start_pve_game` | `save_name: str` | `dict` | 启动人机对战游戏 |
| `player_move` | `x: int, y: int` | `dict` | 玩家落子，如轮到 AI 则自动触发 |
| `ai_move` | 无 | `dict` | AI 执行落子 |
| `start_ai_vs_ai` | `engine1: AIEngine, engine2: AIEngine, num_games: int, recorder: DataRecorder` | `list` | 执行 AI 内战 |

**代码框架**:
```python
class AIGameManager:
    """AI 游戏管理器 - 管理人机对战和 AI 内战流程"""
    
    def __init__(self, controller, engine, ai_player):
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
    
    def start_pve_game(self, save_name=None):
        """启动人机对战游戏"""
        result = self.controller.new_game(save_name)
        
        # 如果 AI 执黑，自动执行第一步
        if result.get("status") == "success" and self.ai_player == 1:
            return self.ai_move()
        
        return result
    
    def player_move(self, x, y):
        """
        玩家落子，如轮到 AI 则自动触发
        
        Returns:
            {"player_result": ..., "ai_result": ...} 或 {"player_result": ...}
        """
        # 玩家落子
        result = self.controller.make_move(x, y)
        if "error" in result:
            return {"player_result": result}
        
        # 检查游戏是否结束或轮到 AI
        if (self.controller.is_over() or 
            self.controller.get_current_player() != self.ai_player):
            return {"player_result": result}
        
        # 触发 AI 落子
        ai_result = self.ai_move()
        return {
            "player_result": result,
            "ai_result": ai_result
        }
    
    def ai_move(self):
        """AI 执行落子"""
        # 获取当前棋盘状态
        state = self.controller.get_state()
        board = state["board"]
        current_player = state["current_player"]
        
        # AI 计算落子位置
        x, y = self.engine.get_move(board, current_player)
        
        # 执行落子
        return self.controller.make_move(x, y)
    
    @staticmethod
    def start_ai_vs_ai(engine1, engine2, num_games, recorder):
        """
        执行 AI 内战
        
        Args:
            engine1: 执黑 AI 引擎
            engine2: 执白 AI 引擎
            num_games: 对战局数
            recorder: 数据记录器
            
        Returns:
            对战结果列表
        """
        results = []
        
        for game_num in range(1, num_games + 1):
            # 创建独立 controller
            controller = GomokuController()
            controller.new_game(f"ai_vs_ai_{game_num}")
            
            # 开始记录
            recorder.new_game()
            
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
                recorder.record_move(game_num, board_vector, rewards, (x, y))
                
                # 执行落子
                controller.make_move(x, y)
            
            # 保存结果
            winner = controller.get_winner()
            end_state = controller.get_state()
            end_board = AIEngine.board_to_vector(end_state["board"], 0)
            recorder.save_game_result(game_num, winner, end_board)
            
            results.append({
                "game": game_num,
                "winner": winner
            })
        
        return results
```

## 4. 流程设计

### 4.1 人机对战流程

```
1. GUI 启动 -> 选择"人机对战"模式 -> 选择执黑/白
2. 创建 AIGameManager(controller, engine, ai_player)
3. 调用 manager.start_pve_game(save_name)
4. 如果 AI 执黑，manager 自动执行 AI 第一步
5. 玩家点击棋盘 -> manager.player_move(x, y)
6. manager 内部逻辑：
   a. 先执行玩家落子：controller.make_move(x, y)
   b. 检查游戏是否结束
   c. 如轮到 AI，自动调用 ai_move()：
      - 获取棋盘状态
      - engine.get_move() 计算落子位置
      - controller.make_move() 执行 AI 落子
7. 返回结果给 GUI，更新界面
8. 重复 5-7 直到游戏结束
```

### 4.2 AI 内战流程

```
1. GUI 启动 -> 选择"AI 内战"模式 -> 设置对战局数
2. 创建两个 AIEngine 实例（engine1 执黑，engine2 执白）
3. 创建 DataRecorder 实例
4. 调用 AIGameManager.start_ai_vs_ai(engine1, engine2, num_games, recorder)
5. 对每一局游戏：
   a. 创建独立 controller
   b. recorder.new_game()
   c. 循环直到游戏结束：
      i.   获取当前棋盘和落子方
      ii.  调用 AI 模型 predict() 获取 reward
      iii. recorder.record_move() 记录数据
      iv.  engine.get_move() 选择落子位置
      v.   controller.make_move() 执行落子
   d. recorder.save_game_result() 保存结果
6. 所有对局完成，返回结果给 GUI
```

## 5. GUI 启动界面设计

**文件**: `gui/main.py` 修改

**新增启动选择界面**:

```python
class GameModeSelector:
    """游戏模式选择器 - 启动时显示"""
    
    def __init__(self, root, on_select):
        self.window = tk.Toplevel(root)
        self.window.title("选择游戏模式")
        self.window.geometry("400x500")
        self.window.transient(root)
        self.window.grab_set()
        
        # 模式选择按钮
        # - 人人对战
        # - 人机对战（选择后弹出执黑/白选择）
        # - AI 内战（选择后设置局数）
```

**启动流程修改**:
```python
def main():
    root = tk.Tk()
    
    # 先显示模式选择
    mode_selector = GameModeSelector(root, on_mode_selected)
    
    # 选择后创建主界面
    # ...
```

## 6. 测试用例

### 6.1 单元测试

**文件**: `tests/test_ai_engine.py`

```python
class TestAIEngine(unittest.TestCase):
    """AI 引擎测试"""
    
    def test_board_to_vector(self):
        """测试棋盘转向量"""
        board = [[0]*15 for _ in range(15)]
        vector = AIEngine.board_to_vector(board, 1)
        self.assertEqual(len(vector), 226)
        self.assertEqual(vector[-1], 1)
    
    def test_random_model_predict(self):
        """测试 RandomModel 输出格式"""
        model = RandomModel()
        result = model.predict([0]*226)
        self.assertEqual(len(result), 225)
        self.assertTrue(all(r == 0 for r in result))  # 所有 reward 为 0
    
    def test_engine_get_move(self):
        """测试引擎落子选择"""
        model = RandomModel()
        engine = AIEngine(model)
        board = [[0]*15 for _ in range(15)]
        x, y = engine.get_move(board, 1)
        self.assertTrue(0 <= x < 15 and 0 <= y < 15)
        self.assertEqual(board[x][y], 0)
```

**文件**: `tests/test_data_recorder.py`

```python
class TestDataRecorder(unittest.TestCase):
    """数据记录器测试"""
    
    def test_create_round_dir(self):
        """测试轮次目录创建"""
        recorder = DataRecorder("random", "test_ai_data")
        self.assertTrue(os.path.exists(recorder.round_dir))
    
    def test_record_and_save(self):
        """测试记录和保存"""
        recorder = DataRecorder("random", "test_ai_data")
        game_id = recorder.new_game()
        recorder.record_move(game_id, [0]*226, [0]*225, (7, 7))
        recorder.save_game_result(game_id, 1, [0]*226)
        # 验证文件存在
```

### 6.2 集成测试

**文件**: `tests/test_ai_integration.py`

```python
class TestAIIntegration(unittest.TestCase):
    """AI 集成测试"""
    
    def test_pve_game(self):
        """测试人机对战完整流程"""
        from ai.game_manager import AIGameManager
        from ai.engine import AIEngine
        from ai.random.model import RandomModel
        
        controller = GomokuController()
        engine = AIEngine(RandomModel())
        # 创建人机对战管理器，AI 执白
        manager = AIGameManager(controller, engine, ai_player=2)
        # 启动游戏
        manager.start_pve_game("test_pve")
        # 玩家落子
        manager.player_move(7, 7)
        # 验证 AI 已自动落子
        self.assertEqual(len(controller.game.history), 2)
    
    def test_ai_vs_ai_game(self):
        """测试 AI 内战完整流程"""
        from ai.game_manager import AIGameManager
        from ai.engine import AIEngine
        from ai.random.model import RandomModel
        from ai.data_recorder import DataRecorder
        
        engine1 = AIEngine(RandomModel())
        engine2 = AIEngine(RandomModel())
        recorder = DataRecorder("random", "test_ai_data")
        
        # 执行 AI 内战，1 局
        results = AIGameManager.start_ai_vs_ai(engine1, engine2, 1, recorder)
        
        # 验证结果
        self.assertEqual(len(results), 1)
        # 验证数据文件已创建
```

## 7. 实现顺序

1. **基础接口**: `ai/base_model.py`
2. **Random 模型**: `ai/random/model.py`
3. **AI 引擎**: `ai/engine.py`
4. **数据记录器**: `ai/data_recorder.py`
5. **AI 游戏管理器**: `ai/game_manager.py`
6. **单元测试**: `tests/test_ai_engine.py`, `tests/test_data_recorder.py`, `tests/test_game_manager.py`
7. **集成测试**: `tests/test_ai_integration.py`
8. **GUI 启动界面**: 修改 `gui/main.py`（调用 ai 包接口）
9. **AI 对战脚本**: `ai/battle.py`

## 8. AI 对战脚本设计

### 8.1 功能描述

独立运行的命令行脚本，用于启动两个 AI 对战。支持通过参数指定双方 AI 及对战局数。**每一局执方随机**（黑白随机分配），对战结束后输出每局结果及双方胜率。

**文件**: `ai/battle.py`

### 8.2 命令格式

```bash
# 基本用法
python ai/battle.py --ai1 random --ai2 fc_v1 --games 10

# 自定义参数
python ai/battle.py --ai1 random --ai2 random --games 50
```

### 8.3 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--ai1` | str | 必填 | 第一方 AI 名称（目录名） |
| `--ai2` | str | 必填 | 第二方 AI 名称（目录名） |
| `--games` | int | 10 | 对战局数 |

### 8.4 输出格式

```
开始 AI 对战: random vs fc_v1, 共 10 局

第 1 局: ai1(白) vs ai2(黑) -> 黑方胜
第 2 局: ai1(黑) vs ai2(白) -> 白方胜
...
第 10 局: ai1(白) vs ai2(黑) -> 平局

=== 对战结果 ===
总局数: 10
ai1 胜: 3 局 (30.0%)
ai2 胜: 6 局 (60.0%)
平局: 1 局 (10.0%)
```

### 8.5 AIGameManager 扩展

在 `ai/game_manager.py` 中新增静态方法 `start_ai_vs_ai_random`：

**方法签名**：
```python
@staticmethod
def start_ai_vs_ai_random(
    engine1: AIEngine,
    engine2: AIEngine,
    num_games: int,
    recorder: DataRecorder
) -> List[dict]
```

**返回值**：每局结果列表，每项格式：
```python
{
    "game": 局数,          # 从1开始
    "winner": 获胜方,       # 1=黑, 2=白, 3=平局
    "ai1_color": ai1执棋颜色,  # 1=黑, 2=白
    "ai2_color": ai2执棋颜色,  # 1=黑, 2=白
    "ai1_won": bool,       # ai1是否获胜
    "ai2_won": bool        # ai2是否获胜
}
```

**主流程伪代码**：
```
for game_num in 1..num_games:
    # 随机决定执方
    ai1_color = random.choice([1, 2])
    ai2_color = 3 - ai1_color  # 互为黑白
    
    # 根据颜色分配引擎
    engine_black = engine1 if ai1_color == 1 else engine2
    engine_white = engine2 if ai1_color == 1 else engine1
    
    # 进行一局对战（与 start_ai_vs_ai 逻辑相同）
    run_one_game(engine_black, engine_white, game_num, recorder)
    
    # 计算 ai1/ai2 是否获胜
    ai1_won = (winner == ai1_color)
    ai2_won = (winner == ai2_color)
    
    results.append({...})

return results
```

### 8.6 主流程测试用例

```python
class TestBattle(unittest.TestCase):
    def test_random_side_assignment(self):
        """每局执方随机分配"""
        # 运行多局，验证执方分配是随机的
    
    def test_win_rate_calculation(self):
        """胜率计算正确"""
        # 模拟已知结果，验证胜率计算
    
    def test_full_battle_flow(self):
        """完整对战流程：指定 ai 名称，运行对战，输出结果"""
```
