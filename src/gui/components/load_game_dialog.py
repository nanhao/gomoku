"""加载游戏对话框组件

封装存档选择对话框，返回选中的存档名或 None。
"""

import tkinter as tk
from tkinter import messagebox
from typing import List, Optional


class LoadGameDialog:
    """加载游戏对话框"""
    
    @staticmethod
    def show(parent, saves: List[str]) -> Optional[str]:
        """显示存档选择对话框
        
        Args:
            parent: 父窗口
            saves: 存档名称列表
            
        Returns:
            选中的存档名，或 None（用户取消/未选择）
        """
        result = {"save_name": None}
        
        dialog = tk.Toplevel(parent)
        dialog.title("加载游戏")
        dialog.geometry("300x300")
        dialog.transient(parent)
        dialog.grab_set()
        
        tk.Label(dialog, text="选择存档:", font=("微软雅黑", 12)).pack(pady=10)
        
        listbox = tk.Listbox(dialog, height=10, font=("微软雅黑", 11))
        listbox.pack(fill=tk.BOTH, expand=True, padx=20, pady=5)
        
        for save in saves:
            listbox.insert(tk.END, save)
        
        def do_load():
            selection = listbox.curselection()
            if not selection:
                messagebox.showwarning("提示", "请选择一个存档")
                return
            
            result["save_name"] = listbox.get(selection[0])
            dialog.destroy()
        
        tk.Button(dialog, text="加载", command=do_load, width=15).pack(pady=10)
        
        # 等待对话框关闭
        parent.wait_window(dialog)
        
        return result["save_name"]
