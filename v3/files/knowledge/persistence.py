import os
import json
import pickle
from typing import Any, Dict, List

class KnowledgePersistence:
    """知识持久化管理"""
    
    def __init__(self, base_dir: str = "./knowledge_store"):
        """
        初始化持久化管理器
        
        Args:
            base_dir: 存储基础目录
        """
        self.base_dir = base_dir
        self._ensure_directories()
    
    def _ensure_directories(self):
        """确保必要的目录存在"""
        directories = [
            self.base_dir,
            os.path.join(self.base_dir, "rules"),
            os.path.join(self.base_dir, "patterns"),
            os.path.join(self.base_dir, "objects"),
            os.path.join(self.base_dir, "prior")
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def save_json(self, data: Any, category: str, name: str) -> str:
        """
        保存JSON数据
        
        Args:
            data: 要保存的数据
            category: 数据类别
            name: 数据名称
            
        Returns:
            保存的文件路径
        """
        directory = os.path.join(self.base_dir, category)
        os.makedirs(directory, exist_ok=True)
        
        file_path = os.path.join(directory, f"{name}.json")
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        return file_path
    
    def load_json(self, category: str, name: str) -> Any:
        """
        加载JSON数据
        
        Args:
            category: 数据类别
            name: 数据名称
            
        Returns:
            加载的数据
        """
        file_path = os.path.join(self.base_dir, category, f"{name}.json")
        
        if not os.path.exists(file_path):
            return None
        
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def save_object(self, obj: Any, category: str, name: str) -> str:
        """
        保存Python对象
        
        Args:
            obj: 要保存的对象
            category: 对象类别
            name: 对象名称
            
        Returns:
            保存的文件路径
        """
        directory = os.path.join(self.base_dir, category)
        os.makedirs(directory, exist_ok=True)
        
        file_path = os.path.join(directory, f"{name}.pkl")
        
        with open(file_path, 'wb') as f:
            pickle.dump(obj, f)
        
        return file_path
    
    def load_object(self, category: str, name: str) -> Any:
        """
        加载Python对象
        
        Args:
            category: 对象类别
            name: 对象名称
            
        Returns:
            加载的对象
        """
        file_path = os.path.join(self.base_dir, category, f"{name}.pkl")
        
        if not os.path.exists(file_path):
            return None
        
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    
    def list_files(self, category: str) -> List[str]:
        """
        列出特定类别的所有文件
        
        Args:
            category: 文件类别
            
        Returns:
            文件名称列表
        """
        directory = os.path.join(self.base_dir, category)
        
        if not os.path.exists(directory):
            return []
        
        files = []
        for filename in os.listdir(directory):
            base_name, ext = os.path.splitext(filename)
            if ext in ['.json', '.pkl']:
                files.append(base_name)
        
        return files
    
    def delete_file(self, category: str, name: str) -> bool:
        """
        删除文件
        
        Args:
            category: 文件类别
            name: 文件名称
            
        Returns:
            是否成功删除
        """
        json_path = os.path.join(self.base_dir, category, f"{name}.json")
        pkl_path = os.path.join(self.base_dir, category, f"{name}.pkl")
        
        deleted = False
        
        if os.path.exists(json_path):
            os.remove(json_path)
            deleted = True
        
        if os.path.exists(pkl_path):
            os.remove(pkl_path)
            deleted = True
        
        return deleted