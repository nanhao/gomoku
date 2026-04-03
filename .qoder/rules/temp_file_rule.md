---
trigger: model_decision
description: 生成临时文件时触发
---
1. 将临时脚本、临时文件保存到 {project_root}/temp 目录下
2. 不要删除这些临时文件
3. 文件命名要清晰，能反映其用途（如：simplify_csv_20250331.py）
4. 除非用户明确要求清理，否则保留所有临时文件
5. 在项目目录下创建 temp 目录（如果不存在）