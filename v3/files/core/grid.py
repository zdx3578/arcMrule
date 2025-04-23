import numpy as np
from typing import List, Tuple, Dict, Any

class Grid:
    """ARC网格的基本表示和操作"""
    
    def __init__(self, data: np.ndarray):
        """
        初始化网格
        
        Args:
            data: 表示网格的NumPy数组，每个元素为颜色代码(0-9)
        """
        self.data = np.array(data, dtype=np.int8)
        self.height, self.width = self.data.shape
        
    @classmethod
    def from_list(cls, data: List[List[int]]):
        """从嵌套列表创建网格"""
        return cls(np.array(data))
    
    def get_diff(self, other: 'Grid') -> 'Grid':
        """
        计算两个网格之间的差异
        
        Args:
            other: 另一个网格对象
            
        Returns:
            包含差异的新网格，相同位置值为0，不同位置保留目标网格的值
        """
        if self.data.shape != other.data.shape:
            raise ValueError("网格形状不匹配，无法计算差异")
        
        diff_data = np.zeros_like(self.data)
        mask = (self.data != other.data)
        diff_data[mask] = other.data[mask]
        
        return Grid(diff_data)
    
    def rotate(self, k: int = 1) -> 'Grid':
        """旋转网格，k表示90度旋转的次数"""
        return Grid(np.rot90(self.data, k))
    
    def flip(self, axis: int = 0) -> 'Grid':
        """翻转网格，axis=0表示上下翻转，axis=1表示左右翻转"""
        return Grid(np.flip(self.data, axis))
    
    def get_colors(self) -> List[int]:
        """获取网格中出现的所有颜色"""
        return sorted(list(set(self.data.flatten())))
    
    def count_color(self, color: int) -> int:
        """计算特定颜色在网格中出现的次数"""
        return np.sum(self.data == color)
    
    def __str__(self) -> str:
        """字符串表示，用于调试"""
        rows = []
        for row in self.data:
            rows.append(" ".join(str(x) for x in row))
        return "\n".join(rows)