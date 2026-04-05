# 任务报告：BaseTrainer 重构与 CNN 训练器实现

## 1. 需求内容

根据 requirements/key_points.md 第十七次追加：
- 在 ai 目录下创建一个 base_trainer.py 文件，作为现在的 ai/models/fc_v1/trainer.py 的父类
- 重构 ai/models/fc_v1/trainer.py 继承该父类，把通用的代码都挪到父类
- 在 cnn 模型目录下也创建并实现一个训练类继承该父类
- train_main.py 支持 cnn_v1 模型的训练，使用方式和 fc_v1 的一样

## 2. 解决思路

通过分析 FCv1Trainer 的代码，识别出通用训练逻辑：
- **通用代码**：数据加载、张量准备、训练循环（train_single/train_batch）、训练周期（train_cycle）、损失计算
- **模型特定代码**：模型创建、输入格式转换（FC 是 226 维向量，CNN 是 4D 张量）

设计抽象基类 BaseTrainer：
- 使用 ABC 抽象基类机制，强制子类实现 `_create_model()` 和 `model_name` 属性
- 提供 `_convert_inputs()` 钩子方法，允许子类覆盖以支持不同的输入格式
- 将通用训练逻辑封装在基类中

## 3. 执行过程

### 步骤 1：编写设计文档
- 创建 design/base_trainer_design.md
- 定义类结构、主流程伪代码、测试用例设计

### 步骤 2：编写测试用例
- tests/ai/test_base_trainer.py：测试抽象基类特性（5 个测试）
- tests/ai/test_cnn_trainer.py：测试 CNN 训练器功能（14 个测试）
- tests/ai/test_train_main.py：测试 train_main.py 的 get_trainer 函数（5 个测试）

### 步骤 3：实现 BaseTrainer
- 创建 ai/base_trainer.py
- 实现抽象基类，包含通用训练逻辑
- 提供 `_convert_inputs()` 默认实现（适用于 FC 模型）

### 步骤 4：重构 FCv1Trainer
- 修改 ai/models/fc_v1/trainer.py
- 移除所有通用代码，只保留 `_create_model()` 和 `model_name` 属性
- 代码从 372 行减少到 42 行

### 步骤 5：创建 CNNTrainer
- 创建 ai/models/cnn_v1/trainer.py
- 继承 BaseTrainer
- 实现 `_create_model()` 返回 CNNModel
- 覆盖 `_convert_inputs()` 以支持 4D 输入格式

### 步骤 6：修改 train_main.py
- 更新 get_trainer() 函数，支持 cnn_v1 模型
- 添加 CNNTrainer 导入和实例化逻辑

### 步骤 7：运行全量测试
- 新测试：25 个全部通过
- FC v1 原有测试：43 个全部通过
- AI 全量测试：214 个全部通过

## 4. 执行结果

### 代码变更

**新建文件**：
- ai/base_trainer.py（403 行）：训练器抽象基类
- ai/models/cnn_v1/trainer.py（67 行）：CNN 训练器
- design/base_trainer_design.md（196 行）：设计文档
- tests/ai/test_base_trainer.py（91 行）：BaseTrainer 测试
- tests/ai/test_cnn_trainer.py（228 行）：CNNTrainer 测试
- tests/ai/test_train_main.py（59 行）：train_main 测试

**修改文件**：
- ai/models/fc_v1/trainer.py：从 372 行减少到 42 行，继承 BaseTrainer
- ai/train_main.py：添加 cnn_v1 支持

### 测试结果

```
tests/ai/test_base_trainer.py: 5 passed
tests/ai/test_cnn_trainer.py: 14 passed  
tests/ai/test_train_main.py: 5 passed
tests/ai/test_fc_v1_trainer.py: 43 passed
AI 全量测试: 214 passed, 5 subtests passed
```

### 功能验证

- FCv1Trainer 重构后功能完全保持
- CNNTrainer 可以正常训练和保存模型
- train_main.py 可以通过 `--model cnn_v1` 参数使用 CNN 训练器
- 训练周期功能对两种模型都可用

## 5. 涉及文件

### 设计文档
- design/base_trainer_design.md

### 源代码
- ai/base_trainer.py（新建）
- ai/models/fc_v1/trainer.py（重构）
- ai/models/cnn_v1/trainer.py（新建）
- ai/train_main.py（修改）

### 测试代码
- tests/ai/test_base_trainer.py（新建）
- tests/ai/test_cnn_trainer.py（新建）
- tests/ai/test_train_main.py（新建）

### 需求文档
- requirements/key_points.md（参考，未修改）
