# 训练样本合并功能设计

## 背景

自我对局数增多后，训练时需要从 100+ 个子目录分别加载 `sample.json` 文件，
逐个打开/解析/关闭的开销显著。将所有 `sample.json` 合并为单个大文件，
训练时只需加载一个文件，可大幅减少 I/O 和 JSON 解析时间。

## 设计原则

- merge 文件与个体 sample.json **目录隔离**：merge 文件通常放在父级目录，
  个体 sample.json 在子目录中，二者物理隔离，加载时直接合并无需去重
- 合并脚本**独立**：可独立运行，也可被 `--self-play` 模式调用
- 向后兼容：`--train-from-dir` 同时支持 merge 文件和个体 sample.json

## API 设计

### 1. 新模块：`ai/merge_samples.py`

```python
def discover_sample_files(data_dir: str) -> List[str]:
    """递归发现目录下所有 sample.json 文件路径

    Args:
        data_dir: 数据目录路径

    Returns:
        sample.json 文件路径列表，按路径排序
    """

def merge_sample_files(sample_files: List[str], output_path: str) -> dict:
    """将多个 sample.json 文件合并为一个文件

    Args:
        sample_files: sample.json 文件路径列表
        output_path: 合并后的输出文件路径

    Returns:
        {"total_files": int, "total_samples": int, "output_path": str}
    """

def merge_directories(directories: List[str], output_path: str) -> dict:
    """将多个目录下的 sample.json 合并为一个文件

    Args:
        directories: 数据目录路径列表
        output_path: 合并后的输出文件路径

    Returns:
        {"total_files": int, "total_samples": int, "output_path": str}
    """
```

CLI 入口：

```
python ai/merge_samples.py <dir1> [dir2] ... -o <output_path>
```

- 位置参数：一个或多个数据目录
- `-o` / `--output`：合并文件输出路径（必需）
- 执行后打印统计：合并文件数、总样本数、输出路径

### 2. 修改：`ai/sample_loader.py`

```python
class SampleLoader:
    # 现有方法不变

    def load_merge_file(self, merge_path: str) -> list:
        """加载合并后的样本文件

        Args:
            merge_path: merge_*.json 文件路径

        Returns:
            sample 数据列表
        """

    def discover_and_load(self, data_dir: str) -> list:
        """从目录发现并加载所有训练数据（merge 文件 + 个体 sample.json）

        扫描指定目录：
        1. 找到所有 merge_*.json 文件 → 加载
        2. 找到所有包含 sample.json 的子目录 → 加载
        3. 合并所有数据返回

        Args:
            data_dir: 训练数据目录路径

        Returns:
            合并后的 sample 数据列表
        """
```

### 3. 修改：`ai/train_main.py`

#### `--self-play` 模式

新增 `--merge-output` 可选参数：

```
python ai/train_main.py --model cnn_v2 --self-play --games 100 --merge-output ai/train_data/merge_cnn_v2.json
```

- 指定时：对局完成后自动调用 `merge_directories` 合并数据
- 不指定时：行为不变，仅输出个体 sample.json

#### `--train-from-dir` 模式

- 当参数指向目录时：调用 `SampleLoader.discover_and_load` 加载
  （自动发现 merge 文件和个体 sample.json）
- 当参数指向 `.json` 文件时：直接加载该文件（支持直接指定 merge 文件）

#### `discover_sample_dirs` 修改

新增 `discover_all_data` 函数替代原有逻辑：

```python
def discover_all_data(train_data_dir: str) -> dict:
    """发现目录下所有训练数据源

    Args:
        train_data_dir: 训练数据目录路径

    Returns:
        {
            "merge_files": List[str],    # merge_*.json 文件路径列表
            "sample_dirs": List[str]     # 包含 sample.json 的子目录列表
        }
    """
```

## 主流程伪代码

### merge_samples.py CLI

```
1. 解析参数：directories (位置参数), -o output_path
2. 校验：output_path 必须指定；directories 至少一个
3. 调用 merge_directories(directories, output_path)
4. 打印统计结果
```

### merge_directories

```
1. 遍历每个 directory：
   a. 调用 discover_sample_files(directory) 找到所有 sample.json
   b. 收集到 sample_files 列表
2. 调用 merge_sample_files(sample_files, output_path)
3. 返回统计结果
```

### merge_sample_files

```
1. all_data = []
2. for each sample_file in sample_files:
   a. json.load(sample_file) → data
   b. all_data.extend(data)
3. 确保输出目录存在
4. json.dump(all_data, output_path, ensure_ascii=False)
5. 返回 {"total_files": len(sample_files), "total_samples": len(all_data), "output_path": output_path}
```

### discover_and_load

```
1. result = discover_all_data(data_dir)
2. all_data = []
3. for merge_file in result["merge_files"]:
   a. all_data.extend(SampleLoader.load_merge_file(merge_file))
4. for sample_dir in result["sample_dirs"]:
   a. all_data.extend(SampleLoader.load(os.path.join(sample_dir, "sample.json")))
5. return all_data
```

### --self-play 模式（修改后）

```
1. 执行自我对弈 + 评价（现有逻辑不变）
2. 打印对局统计和样本数（现有逻辑不变）
3. 如果指定了 --merge-output：
   a. 调用 merge_directories([train_data_dir], merge_output)
   b. 打印合并结果
```

### --train-from-dir 模式（修改后）

```
1. 如果参数是 .json 文件：
   a. 直接加载该文件
2. 如果参数是目录：
   a. 调用 SampleLoader.discover_and_load(data_dir) 加载所有数据
   b. 如果无数据，报错退出
3. 后续训练逻辑不变
```

## 主流程测试用例

### merge_samples 模块测试

| 用例 | 说明 |
|------|------|
| test_discover_sample_files_finds_all | 递归发现所有 sample.json |
| test_discover_sample_files_ignores_merge | 不收集 merge_*.json |
| test_discover_sample_files_empty_dir | 空目录返回空列表 |
| test_merge_sample_files_correct_count | 合并后样本数 = 各文件样本数之和 |
| test_merge_sample_files_output_valid | 输出文件可被 json.load 正确读取 |
| test_merge_sample_files_creates_dir | 输出目录不存在时自动创建 |
| test_merge_directories_multi_dir | 多目录合并样本数正确 |

### SampleLoader 新方法测试

| 用例 | 说明 |
|------|------|
| test_load_merge_file | 加载 merge 文件返回正确数据 |
| test_discover_and_load_merge_only | 目录仅有 merge 文件时正确加载 |
| test_discover_and_load_samples_only | 目录仅有个体 sample.json 时正确加载 |
| test_discover_and_load_both | 目录同时有 merge 文件和个体 sample.json 时全部加载 |
| test_discover_and_load_empty | 空目录返回空列表 |

### CLI 集成测试

| 用例 | 说明 |
|------|------|
| test_self_play_with_merge_output | --self-play --merge-output 自动合并 |
| test_self_play_without_merge_output | --self-play 不指定 merge-output 行为不变 |
| test_train_from_dir_with_merge | --train-from-dir 目录含 merge 文件时正确加载 |
| test_train_from_dir_json_file | --train-from-dir 直接指定 .json 文件 |
