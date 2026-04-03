# 修复AI训练存档名唯一性缺陷

## 1. 需求内容

修复 `ai/game_manager.py` 中 `start_ai_vs_ai` 方法的存档名生成算法缺陷。该缺陷导致训练周期（train_cycle）在多轮训练时，因存档名称冲突而失败。

错误信息：
```
训练失败: 创建游戏失败: 存档 'ai_vs_ai_30_2098160943856' 已存在
```

## 2. 解决思路

**问题根源分析**：
- 原代码使用 `id(controller)` 生成存档名称的一部分
- `id(controller)` 返回 Python 对象的内存地址，该地址在对象销毁后可能被重用
- 训练周期每轮都会创建新的 controller，不同轮次可能分配到相同的 id，导致存档名冲突

**解决方案**：
- 将 `id(controller)` 替换为毫秒时间戳 `int(time.time() * 1000)`
- 时间戳具有单调递增特性，能确保每局游戏的存档名称唯一

## 3. 执行过程

### 步骤1：定位问题
- 文件：`ai/game_manager.py`
- 方法：`start_ai_vs_ai`
- 问题代码：`unique_name = f"ai_vs_ai_{game_num}_{id(controller)}"`

### 步骤2：修复代码
```python
# 修改前
unique_name = f"ai_vs_ai_{game_num}_{id(controller)}"

# 修改后
import time
timestamp = int(time.time() * 1000)
unique_name = f"ai_vs_ai_{game_num}_{timestamp}"
```

### 步骤3：添加回归测试
- 文件：`tests/ai/test_game_manager.py`
- 新增测试方法：`test_start_ai_vs_ai_unique_save_names`
- 测试内容：模拟训练周期，连续两次调用 `start_ai_vs_ai`，验证不会产生存档名冲突

### 步骤4：运行回归测试
```bash
python -m unittest tests.ai.test_game_manager -v
```

**测试结果**：10个测试用例全部通过
- test_ai_move
- test_init
- test_player_move_error
- test_player_move_game_over
- test_player_move_no_ai_turn
- test_start_pve_game_ai_black
- test_start_pve_game_ai_white
- test_start_ai_vs_ai_multiple_games
- test_start_ai_vs_ai_single_game
- test_start_ai_vs_ai_unique_save_names（新增）

## 4. 涉及文件

| 文件路径 | 操作类型 | 说明 |
|---------|---------|------|
| `ai/game_manager.py` | 修改 | 修复存档名生成算法，使用毫秒时间戳替代 id(controller) |
| `tests/ai/test_game_manager.py` | 修改 | 添加回归测试用例 `test_start_ai_vs_ai_unique_save_names` |

## 5. 验证结果

修复后，训练周期可以正常运行多轮而不会出现存档名冲突。回归测试验证了多次调用 `start_ai_vs_ai` 不会产生冲突。
