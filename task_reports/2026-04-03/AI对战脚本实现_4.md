# AI 对战脚本实现

## 1. 需求内容

在 `requirements/key_points.md` 第 384 行新增需求：

> 增加一个脚本可以启动两个 AI 内战，支持用参数指定双方 AI 以及对战局数。每一局执方随机。结束后输出结果和双方胜率。

## 2. 解决思路

新增 `ai/battle.py` 独立脚本，复用已有的 `AIGameManager`、`AIEngine`、`DataRecorder` 组件：

1. **扩展 `AIGameManager`**：新增 `start_ai_vs_ai_random` 静态方法，在每局开始时随机决定双方执黑/白。
2. **新建 `ai/battle.py`**：命令行入口脚本，接受 `--ai1`、`--ai2`、`--games` 参数，动态加载模型，调用 `start_ai_vs_ai_random` 执行对战，逐局输出结果并统计胜率。
3. **设计文档更新**：在 `design/ai_design.md` 中新增第 8 节"AI 对战脚本设计"，补充接口说明和测试用例。

## 3. 执行过程

### 步骤1：更新设计文档

在 `design/ai_design.md` 中追加第 8 节，包含：
- 功能描述、命令格式、参数说明、输出格式
- `AIGameManager.start_ai_vs_ai_random` 方法签名和伪代码
- 主流程测试用例

### 步骤2：扩展 AIGameManager

在 `ai/game_manager.py` 中新增 `start_ai_vs_ai_random` 静态方法：
- 每局随机 `random.choice([1, 2])` 决定 engine1 执黑还是白
- 根据颜色分配引擎后，复用与 `start_ai_vs_ai` 相同的游戏循环逻辑
- 结果中增加 `ai1_color`、`ai2_color`、`ai1_won`、`ai2_won` 字段

### 步骤3：创建 ai/battle.py

实现包含：
- `load_model(ai_name)`: 根据名称动态导入模型类（支持 `random`、`fc_v1`）
- `get_color_name(color)`: 颜色数字转名称
- `get_winner_name(winner)`: 获胜方转描述
- `print_battle_results(ai1, ai2, results)`: 打印胜率统计
- `run_battle(ai1, ai2, games)`: 主对战流程
- `main()`: CLI 入口，使用 `argparse` 解析参数

### 步骤4：编写测试用例

`tests/ai/test_battle.py` 包含 17 个测试：

**主流程测试（TestStartAiVsAiRandom）**：
- 返回结果数量等于对战局数
- 每个结果包含必要字段
- 局数编号从1开始递增
- ai1_color 和 ai2_color 互为黑白（和为3）
- winner 只能是 1/2/3
- ai1_won/ai2_won 与 winner/color 一致
- 多局后执方具有随机性
- 对战后记录文件被正确创建

**边缘测试（TestStartAiVsAiRandomEdge）**：
- 只对战 1 局
- 平局时两方都不算获胜

**脚本模块测试（TestBattleModule）**：
- 加载 random 模型
- 加载未知 AI 抛出 ValueError
- 颜色名称转换正确
- 获胜方描述转换正确
- 胜率统计输出正确
- 空结果不出错
- `run_battle` 完整流程

### 步骤5：运行测试

- 新增测试：17/17 通过
- 全量测试：148/148 通过（无回归）

## 4. 执行结果

功能完整实现：

```bash
# 使用示例
python ai/battle.py --ai1 random --ai2 random --games 10

# 输出示例：
开始 AI 对战: random vs random, 共 10 局

第 1 局: random(黑) vs random(白) -> 黑方胜
第 2 局: random(白) vs random(黑) -> 黑方胜
...
第 10 局: random(白) vs random(黑) -> 平局

=== 对战结果 ===
总局数: 10
random 胜: 4 局 (40.0%)
random 胜: 5 局 (50.0%)
平局: 1 局 (10.0%)
```

## 5. 涉及文件

| 文件 | 操作 | 说明 |
|------|------|------|
| `design/ai_design.md` | 修改 | 新增第 8 节 AI 对战脚本设计 |
| `ai/game_manager.py` | 修改 | 新增 `start_ai_vs_ai_random` 静态方法 |
| `ai/battle.py` | 新建 | AI 对战命令行入口脚本 |
| `tests/ai/test_battle.py` | 新建 | AI 对战功能测试（17个测试用例） |
