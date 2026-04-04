# AI 数据目录路径修正任务报告

## 1. 需求内容

用户反馈回放功能提示"暂无回放数据"，但数据实际存在于 `ai/ai_data/` 和 `ai/train_data/` 目录下。

问题根源：代码中的路径配置指向项目根目录的 `ai_data/` 和 `train_data/`，而实际数据存放在 `ai/` 子目录下。

## 2. 解决思路

修改代码中的路径计算逻辑，使其指向 `ai/` 子目录下的数据目录：
- `ai_data` → `ai/ai_data`
- `train_data` → `ai/train_data`

## 3. 执行过程

### 步骤1：详细设计
- 创建设计文档 `design/ai_data_path_fix_design.md`
- 分析涉及文件和修改方案
- 编写伪代码和测试用例设计

### 步骤2：编码实现
修改以下4个文件：

1. **app/gui_helper.py** - `get_ai_data_dir()` 函数
   - 修改路径计算，指向 `ai/ai_data/`

2. **ai/battle.py** - DataRecorder 初始化路径
   - 修改 `ai_data_dir` 路径为 `ai/ai_data/`

3. **ai/models/fc_v1/trainer.py** - 训练和评价数据目录路径
   - 修改 `ai_data_dir` 和 `train_data_dir` 路径

4. **ai/evaluator.py** - 默认数据目录路径
   - 修改默认参数为 `ai/ai_data` 和 `ai/train_data`

### 步骤3：更新测试用例
- 修改 `tests/app/test_gui_helper.py`
- 添加路径包含 `ai/ai_data` 的验证

### 步骤4：跑通全量测试
- 运行 `tests/run_tests.py`
- 结果：206/206 通过

## 4. 执行结果

- ✅ 所有路径修改完成
- ✅ 全量测试通过（206/206）
- ✅ 回放功能可正确找到数据

## 5. 涉及文件

### 修改的文件
1. `app/gui_helper.py` - 修改 `get_ai_data_dir()` 路径计算
2. `ai/battle.py` - 修改 DataRecorder 路径
3. `ai/models/fc_v1/trainer.py` - 修改训练和评价数据目录路径
4. `ai/evaluator.py` - 修改默认数据目录路径
5. `tests/app/test_gui_helper.py` - 更新测试用例

### 新建的文件
1. `design/ai_data_path_fix_design.md` - 设计文档
2. `task_reports/2026-04-04/AI数据目录路径修正_1.md` - 任务报告

## 6. 验证方式

运行回放功能，应能正确识别 `ai/ai_data/` 目录下的数据。
