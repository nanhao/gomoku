# 评价器双向棋型检测法设计文档

**版本**: v1.0  
**创建日期**: 2026-04-04  
**需求来源**: key_points.md 第十三次追加

## 1. 概述

将 Evaluator 的 reward 计算算法从"五子连线反向传播"改为"双向棋型检测法"。

### 1.1 核心思想

对棋盘上每个空位进行双向检测：
1. **我方视角**：如果当前落子方在此位置落子，会形成什么棋型？reward 是多少？
2. **敌方视角**：如果对手方在此位置落子，会形成什么棋型？reward 是多少？
3. **最终 reward** = max(我方 reward, 敌方 reward)

### 1.2 关键变更

| 项目 | 旧算法 | 新算法 |
|------|--------|--------|
| reward 计算方式 | 五子连线反向传播 | 双向棋型检测 |
| 已落子位置 | reward = MIN_REWARD | reward = 0, mask = 0 |
| mask 初始化 | 全部为 1 | 已落子为 0，空位为 1 |

---

## 2. 棋型定义

### 2.1 棋型枚举

```python
class ShapeType:
    """棋型类型枚举"""
    FIVE = "连五"
    LIVE_FOUR = "活四"
    MULTI_FOUR = "多冲四"
    FOUR = "冲四"
    MULTI_LIVE_THREE = "多活三"
    LIVE_THREE = "活三"
    SLEEP_THREE = "眠三"
    LIVE_TWO = "活二"
    SLEEP_TWO = "眠二"
    SINGLE = "独一"
```

### 2.2 棋型判断规则

| 棋型 | 定义 |
|------|------|
| 连五 | 任意方向上五子相连 |
| 活四 | 任意方向上四子相连，且两端无对方棋子或棋盘边缘阻挡 |
| 多冲四 | 两个或以上方向上四子相连，且两端有且仅有一端被阻挡 |
| 冲四 | 仅一个方向上四子相连，且两端有且仅有一端被阻挡 |
| 多活三 | 两个或以上方向上三子相连，且两端无阻挡 |
| 活三 | 仅一个方向上三子相连，且两端无阻挡 |
| 眠三 | 任意方向上三子相连，且两端有且仅有一端被阻挡 |
| 活二 | 任意方向上二子相连，且两端无阻挡 |
| 眠二 | 任意方向上二子相连，且两端有且仅有一端被阻挡 |
| 独一 | 单独一子，所有方向上无己方子相连 |

### 2.3 方向定义

```python
DIRECTIONS = [
    (0, 1),   # 横向
    (1, 0),   # 纵向
    (1, 1),   # 主对角线
    (1, -1),  # 副对角线
]
```

---

## 3. 数据结构

### 3.1 棋型信息

```python
# detect_shapes 返回的数据结构
shape_info = {
    "shape": "多活三",           # 棋型类型
    "directions": ["横向", "纵向"]  # 该棋型出现的方向列表
}

# count_directions = len(shape_info["directions"])
```

### 3.2 棋型返回示例

```python
# 一个位置可能同时形成多种棋型
shapes = [
    {"shape": "多活三", "directions": ["横向", "纵向"]},
    {"shape": "活二", "directions": ["主对角线"]}
]
```

---

## 4. Reward 映射表

| 棋型 | 我方 | 敌方 |
|------|------|------|
| 连五 | MAX_REWARD | MAX_REWARD × 0.95 |
| 活四 | 0.9 × MAX_REWARD | 0.9 × MAX_REWARD × 0.95 |
| 多冲四 | 0.9 × MAX_REWARD | 0.9 × MAX_REWARD × 0.95 |
| 冲四 | 递归计算 | 递归计算 × 0.95 |
| 多活三 | 0.8 × MAX_REWARD | 0.8 × MAX_REWARD × 0.95 |
| 活三 | 0.5 × MAX_REWARD | 0.5 × MAX_REWARD × 0.95 |
| 眠三 | (0.1 + count_directions) × MAX_REWARD | (0.1 + count_directions) × MAX_REWARD × 0.95 |
| 活二 | (0.3 + count_directions) × MAX_REWARD | (0.3 + count_directions) × MAX_REWARD × 0.95 |
| 眠二 | 0 | 0 |
| 独一 | 0 | 0 |

**注**：MAX_REWARD = 1,000,000

---

## 5. 冲四递归处理

### 5.1 我方冲四

```
1. 模拟我方在 position 落子，形成冲四
2. 找到唯一的防守位置（冲四的另一端空位）
3. 模拟敌方在防守位置落子
4. 检测我方在防守位置落子后能形成的棋型
5. 返回那个棋型的 reward
```

### 5.2 敌方冲四

```
1. 模拟敌方在 position 落子，形成冲四
2. 找到唯一的防守位置
3. 模拟我方在防守位置落子
4. 检测敌方在防守位置落子后能形成的棋型
5. 用敌方列映射计算 reward，最后 × 0.95
```

---

## 6. 主流程伪代码

### 6.1 evaluate_record 主流程

```python
def evaluate_record(record_file: str) -> str:
    # 读取记录
    record = json.load(record_file)
    history = record["history"]
    winner = record["winner"]
    
    # 初始化 sample_dict
    sample_dict = []
    for step in history:
        board = step["input"][:225]
        player = step["input"][225]  # 当前落子方
        
        # 初始化 mask 和 output
        mask = [0 if board[i] != 0 else 1 for i in range(225)]
        output = [0] * 225
        
        # 对每个空位计算 reward
        for pos in range(225):
            if board[pos] == 0:  # 空位
                x, y = pos // 15, pos % 15
                output[pos] = _compute_position_reward(board, x, y, player)
        
        sample_dict.append({
            "input": step["input"],
            "output": output,
            "mask": mask
        })
    
    # 保存 sample.json
    save_sample(sample_dict, sample_file)
    return sample_file
```

### 6.2 位置 reward 计算

```python
def _compute_position_reward(board: List[int], x: int, y: int, current_player: int) -> int:
    opponent = 3 - current_player  # 1->2, 2->1
    
    # 检测当前落子方在此位置落子后的棋型
    shapes1 = detect_shapes(board, (x, y), current_player)
    reward1 = max([shape_to_reward(board, shape, current_player, True) for shape in shapes1])
    
    # 检测对手方在此位置落子后的棋型
    shapes2 = detect_shapes(board, (x, y), opponent)
    reward2 = max([shape_to_reward(board, shape, opponent, False) for shape in shapes2])
    
    return max(reward1, reward2)
```

### 6.3 棋型检测

```python
def detect_shapes(board: List[int], position: Tuple[int, int], player: int) -> List[Dict]:
    """检测落子后形成的所有棋型"""
    x, y = position
    shapes_result = []
    
    # 模拟落子
    simulated_board = board.copy()
    simulated_board[x * 15 + y] = player
    
    # 按方向检测
    for dx, dy in DIRECTIONS:
        shape = _detect_shape_in_direction(simulated_board, x, y, player, dx, dy)
        if shape:
            # 记录该方向的棋型
            _add_shape_to_result(shapes_result, shape, direction_name(dx, dy))
    
    # 合并相同棋型的方向
    return _merge_shape_directions(shapes_result)
```

### 6.4 单方向棋型检测

```python
def _detect_shape_in_direction(board: List[int], x: int, y: int, player: int, dx: int, dy: int) -> str:
    """检测单个方向上的棋型"""
    opponent = 3 - player
    
    # 统计该方向上连续己方棋子数
    count = 1  # 包含当前位置
    
    # 正方向计数
    nx, ny = x + dx, y + dy
    while 0 <= nx < 15 and 0 <= ny < 15 and board[nx * 15 + ny] == player:
        count += 1
        nx += dx
        ny += dy
    # 检查正方向是否被阻挡
    blocked_positive = (not (0 <= nx < 15 and 0 <= ny < 15)) or (board[nx * 15 + ny] == opponent)
    
    # 反方向计数
    nx, ny = x - dx, y - dy
    while 0 <= nx < 15 and 0 <= ny < 15 and board[nx * 15 + ny] == player:
        count += 1
        nx -= dx
        ny -= dy
    # 检查反方向是否被阻挡
    blocked_negative = (not (0 <= nx < 15 and 0 <= ny < 15)) or (board[nx * 15 + ny] == opponent)
    
    # 判断棋型
    if count >= 5:
        return "连五"
    
    blocked_count = int(blocked_positive) + int(blocked_negative)
    
    if count == 4:
        if blocked_count == 0:
            return "活四"
        elif blocked_count == 1:
            return "冲四"
        # blocked_count == 2: 眠四（不构成威胁，忽略）
    elif count == 3:
        if blocked_count == 0:
            return "活三"
        elif blocked_count == 1:
            return "眠三"
    elif count == 2:
        if blocked_count == 0:
            return "活二"
        elif blocked_count == 1:
            return "眠二"
    elif count == 1:
        return "独一"
    
    return None
```

### 6.5 棋型转 Reward

```python
def shape_to_reward(board: List[int], shape_info: Dict, player: int, is_current_player: bool) -> int:
    """将棋型转换为 reward 值"""
    shape_type = shape_info["shape"]
    directions = shape_info["directions"]
    count_directions = len(directions)
    
    # 敌方系数
    factor = 1.0 if is_current_player else 0.95
    
    if shape_type == "连五":
        return int(MAX_REWARD * factor)
    
    elif shape_type == "活四":
        return int(0.9 * MAX_REWARD * factor)
    
    elif shape_type == "多冲四":
        return int(0.9 * MAX_REWARD * factor)
    
    elif shape_type == "冲四":
        # 递归计算
        return _compute_four_reward(board, shape_info, player, is_current_player)
    
    elif shape_type == "多活三":
        return int(0.8 * MAX_REWARD * factor)
    
    elif shape_type == "活三":
        return int(0.5 * MAX_REWARD * factor)
    
    elif shape_type == "眠三":
        return int((0.1 + count_directions) * MAX_REWARD * factor)
    
    elif shape_type == "活二":
        return int((0.3 + count_directions) * MAX_REWARD * factor)
    
    else:  # 眠二, 独一
        return 0
```

---

## 7. 测试用例

### 7.1 主流程测试（primary）

```python
class TestEvaluatorShapeDetection(unittest.TestCase):
    """双向棋型检测法测试"""
    
    def test_live_five(self):
        """测试连五：五子相连"""
        # 构造棋盘：当前方在 (7,7) 落子形成连五
        # reward 应为 MAX_REWARD
        pass
    
    def test_live_four(self):
        """测试活四：四子相连，两端无阻挡"""
        # reward 应为 0.9 * MAX_REWARD
        pass
    
    def test_multi_four(self):
        """测试多冲四：两个方向形成冲四"""
        # reward 应为 0.9 * MAX_REWARD
        pass
    
    def test_four_recursive(self):
        """测试冲四递归计算"""
        # 冲四 -> 模拟防守 -> 检测后续棋型 -> 计算 reward
        pass
    
    def test_multi_live_three(self):
        """测试多活三：两个方向形成活三"""
        # reward 应为 0.8 * MAX_REWARD
        pass
    
    def test_live_three(self):
        """测试活三：单方向三子相连，两端无阻挡"""
        # reward 应为 0.5 * MAX_REWARD
        pass
    
    def test_sleep_three(self):
        """测试眠三：三子相连，一端阻挡"""
        # reward 应为 (0.1 + count_directions) * MAX_REWARD
        pass
    
    def test_live_two(self):
        """测试活二：二子相连，两端无阻挡"""
        # reward 应为 (0.3 + count_directions) * MAX_REWARD
        pass
    
    def test_enemy_factor(self):
        """测试敌方系数 0.95"""
        # 同样棋型，敌方 reward = 我方 reward * 0.95
        pass
    
    def test_mask_for_occupied(self):
        """测试已落子位置的 mask 为 0"""
        pass
    
    def test_reward_for_occupied(self):
        """测试已落子位置的 reward 为 0"""
        pass
```

### 7.2 边缘测试（edge）

```python
class TestEvaluatorShapeDetectionEdge(unittest.TestCase):
    """边缘情况测试"""
    
    def test_empty_board(self):
        """测试空棋盘：所有位置都是独一"""
        pass
    
    def test_board_edge_block(self):
        """测试棋盘边缘阻挡"""
        # 棋子到达棋盘边缘，视为被阻挡
        pass
    
    def test_double_direction_same_shape(self):
        """测试同一棋型在多个方向出现"""
        # 眠三在两个方向出现，count_directions = 2
        pass
    
    def test_max_reward_value(self):
        """测试 reward 值不超过 MAX_REWARD"""
        pass
```

---

## 8. 实现计划

1. **修改 `_initialize_sample`**
   - 已落子位置 mask = 0（原为 1）
   - 已落子位置 reward = 0（原为 MIN_REWARD）

2. **修改 `_compute_rewards`**
   - 删除旧的五子连线反向传播逻辑
   - 改为对每个空位调用 `_compute_position_reward`

3. **新增 `_compute_position_reward`**
   - 双向检测：我方视角 + 敌方视角
   - 返回 max(reward1, reward2)

4. **新增 `detect_shapes`**
   - 检测四个方向的棋型
   - 返回棋型列表（含方向信息）

5. **新增 `_detect_shape_in_direction`**
   - 单方向棋型检测

6. **新增 `shape_to_reward`**
   - 棋型到 reward 的映射

7. **新增 `_compute_four_reward`**
   - 冲四递归计算

---

## 9. 版本历史

| 版本 | 日期 | 变更内容 |
|------|------|----------|
| v1.0 | 2026-04-04 | 初始版本，双向棋型检测法设计 |
