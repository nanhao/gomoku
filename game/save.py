"""存档管理模块"""

import json
import os
from datetime import datetime


class GomokuSave:
    """游戏存档管理类"""
    
    def __init__(self, save_dir="./game/save/"):
        """初始化存档目录"""
        self.save_dir = save_dir
        self._ensure_save_dir()
    
    def save_game(self, game_state, save_name):
        """保存游戏状态
        
        不修改传入的 game_state 字典，创建副本后写入文件。
        """
        # 创建副本避免修改调用者的字典
        state_copy = dict(game_state)
        # 确保存档名称存在
        if "name" not in state_copy:
            state_copy["name"] = save_name
        
        # 生成文件名：存档名称_YYYY-MM-DD.json
        filename = self._generate_filename(save_name)
        filepath = os.path.join(self.save_dir, filename)
        
        # 写入JSON文件
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(state_copy, f, ensure_ascii=False, indent=4)
    
    def load_game(self, save_name):
        """加载游戏状态，返回 dict 格式"""
        # 查找匹配的存档文件
        pattern = f"{save_name}_"
        matching_files = []
        
        for filename in os.listdir(self.save_dir):
            if filename.startswith(pattern) and filename.endswith('.json'):
                matching_files.append(filename)
        
        if not matching_files:
            raise FileNotFoundError(f"存档 '{save_name}' 不存在")
        
        # 选择最新的存档文件（按文件名中的日期）
        # 文件名格式：存档名称_YYYY-MM-DD.json
        # 按字典序排序，日期越新字典序越大
        latest_file = sorted(matching_files)[-1]
        filepath = os.path.join(self.save_dir, latest_file)
        
        # 读取JSON文件
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def list_saves(self):
        """列出所有存档（返回列表）"""
        if not os.path.exists(self.save_dir):
            return []
        
        import re
        saves = set()
        for filename in os.listdir(self.save_dir):
            if filename.endswith('.json'):
                # 提取存档名称（去掉日期时间后缀）
                # 格式：存档名称_YYYY-MM-DD_HHMMSS.json 或 存档名称_YYYY-MM-DD.json（旧格式）
                # 匹配末尾的 _YYYY-MM-DD_HHMMSS 或 _YYYY-MM-DD
                match = re.match(r'(.+)_(\d{4}-\d{2}-\d{2})(_\d{6})?\.json$', filename)
                if match:
                    saves.add(match.group(1))
                else:
                    # 不匹配日期格式的文件，忽略
                    pass
        
        # 返回排序后的列表
        return sorted(list(saves))
    
    def _ensure_save_dir(self):
        """确保存档目录存在"""
        os.makedirs(self.save_dir, exist_ok=True)
    
    def _generate_filename(self, save_name):
        """生成存档文件名
        
        格式：{save_name}_YYYY-MM-DD_HHMMSS.json
        包含时间戳确保同一天多次保存不会覆盖
        """
        now = datetime.now()
        return f"{save_name}_{now.strftime('%Y-%m-%d_%H%M%S')}.json"
