# 实现 CNN 模型并支持模型参数指定

## 1. 需求内容

在 `ai/models/cnn` 目录下实现一个 CNN 网络，然后确保 ai 目录下的脚本，例如 `battle.py`、`data_recorder.py` 等脚本都支持参数指定模型。

## 2. 解决思路

1. **分析现有模型结构**：参考 `fc_v1` 模型的实现方式，创建 CNN 模型
2. **设计 CNN 网络结构**：使用卷积层提取棋盘特征，全连接层输出 reward
3. **输入编码**：将 226 维向量转换为 3 通道 15x15 张量（黑棋、白棋、当前方）
4. **更新相关脚本**：在 `battle.py`、`engine.py` 中添加 CNN 模型支持
5. **编写测试用例**：确保新模型功能正确
6. **运行全量测试**：验证所有功能正常

## 3. 执行过程

### 步骤 1：创建 CNN 模型目录和文件

创建 `ai/models/cnn/` 目录，包含：
- `__init__.py`：模块初始化文件
- `model.py`：CNN 模型实现

### 步骤 2：实现 CNN 网络结构

设计 CNN 网络结构如下：
- **输入层**：3 通道 15x15 张量
  - 通道 1：黑棋位置（1 表示黑棋，0 表示其他）
  - 通道 2：白棋位置（1 表示白棋，0 表示其他）
  - 通道 3：当前落子方（全 1 表示黑方，全 0 表示白方）
- **卷积层**：4 层卷积（3→64→128→256→256），使用 BatchNorm 和 ReLU
- **全连接层**：256\*15\*15 → 1024 → 512 → 225
- **输出层**：225 维 reward 预测

### 步骤 3：更新模型映射

在以下文件中添加 CNN 模型映射：
- `ai/models/__init__.py`：添加 `CNNModel` 导出
- `ai/battle.py`：在 `_AI_MODEL_MAP` 中添加 `cnn` 映射
- `ai/engine.py`：在 `create_engine` 的 `model_map` 中添加 `cnn` 映射

### 步骤 4：分析其他脚本

检查 `data_recorder.py`、`evaluator.py`、`game_manager.py`，确认这些脚本通过传入的 `AIEngine` 实例工作，不需要直接修改模型选择逻辑。

### 步骤 5：编写测试用例

创建 `tests/ai/test_cnn_model.py`，测试内容包括：
- 网络结构测试
- 模型预测测试
- 模型保存/加载测试
- 向量转张量测试
- 集成测试
- 默认加载行为测试

更新现有测试文件：
- `tests/ai/test_battle.py`：添加 `test_load_model_cnn` 测试
- `tests/ai/test_engine.py`：添加 `test_create_engine_cnn` 测试

### 步骤 6：运行测试验证

所有测试通过：
- CNN 模型测试：21 passed
- Battle 测试：19 passed
- Engine 测试：14 passed
- AI 模块全量测试：158 passed
- 项目全量测试：340 passed, 1 error（Tcl 环境问题，非代码问题）

## 4. 执行结果

成功实现 CNN 模型并集成到现有系统中：

1. **新增文件**：
   - `ai/models/cnn/__init__.py`
   - `ai/models/cnn/model.py`
   - `tests/ai/test_cnn_model.py`

2. **修改文件**：
   - `ai/models/__init__.py`：添加 CNNModel 导出
   - `ai/battle.py`：添加 cnn 模型支持，更新帮助文档
   - `ai/engine.py`：添加 cnn 模型支持
   - `tests/ai/test_battle.py`：添加 CNN 模型加载测试
   - `tests/ai/test_engine.py`：添加 CNN 引擎创建测试

3. **功能验证**：
   - CNN 模型可以正常创建、预测、保存、加载
   - `battle.py` 支持 `--ai1 cnn --ai2 fc_v1` 参数
   - `create_engine("cnn")` 可以正确创建 CNN 引擎

## 5. 涉及文件

### 新增文件
- `ai/models/cnn/__init__.py`
- `ai/models/cnn/model.py`
- `tests/ai/test_cnn_model.py`

### 修改文件
- `ai/models/__init__.py`
- `ai/battle.py`
- `ai/engine.py`
- `tests/ai/test_battle.py`
- `tests/ai/test_engine.py`
