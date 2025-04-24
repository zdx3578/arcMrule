import numpy as np
from typing import List, Tuple, Dict, Set
from collections import defaultdict
from .grid import Grid

class Object:
    """表示网格中提取的对象"""
    
    def __init__(self, grid: Grid, positions: List[Tuple[int, int]], color: int):
        """
        初始化对象
        
        Args:
            grid: 来源网格
            positions: 对象包含的像素位置列表 [(row, col), ...]
            color: 对象颜色
        """
        self.grid = grid
        self.positions = sorted(positions)  # 排序以确保一致性
        self.color = color
        self.sub_objects = []  # 子对象列表
        self._calculate_bounds()
        
    def _calculate_bounds(self):
        """计算对象的边界"""
        rows = [p[0] for p in self.positions]
        cols = [p[1] for p in self.positions]
        self.min_row = min(rows)
        self.max_row = max(rows)
        self.min_col = min(cols)
        self.max_col = max(cols)
        self.height = self.max_row - self.min_row + 1
        self.width = self.max_col - self.min_col + 1
        self.size = len(self.positions)
        
    def extract_sub_objects(self, min_size: int = 2):
        """
        提取所有可能的子对象
        
        Args:
            min_size: 最小子对象大小
        """
        self.sub_objects = []
        # 对于n个像素的对象，生成所有可能的n-1, n-2, ..., min_size个像素的子对象
        for size in range(self.size - 1, min_size - 1, -1):
            from itertools import combinations
            for combo in combinations(self.positions, size):
                self.sub_objects.append(Object(self.grid, list(combo), self.color))
                
    def get_bounding_box(self) -> np.ndarray:
        """获取对象的边界框"""
        box = np.zeros((self.height, self.width), dtype=np.int8)
        for r, c in self.positions:
            box[r - self.min_row, c - self.min_col] = self.color
        return box
    
    def is_rectangle(self) -> bool:
        """检查对象是否为矩形"""
        return len(self.positions) == self.height * self.width
    
    def is_frame(self) -> bool:
        """检查对象是否为框架（中间为空的矩形）"""
        if not self.is_rectangle():
            return False
        
        # 检查是否有内部空白
        for r in range(self.min_row + 1, self.max_row):
            for c in range(self.min_col + 1, self.max_col):
                if (r, c) in self.positions:
                    return False
        return True
    
    def __str__(self) -> str:
        """字符串表示，用于调试"""
        return f"Object(color={self.color}, size={self.size}, bounds=({self.min_row},{self.min_col})-({self.max_row},{self.max_col}))"

class ObjectExtractor:
    """从网格中提取对象"""
    
    def __init__(self, grid: Grid):
        """
        初始化提取器
        
        Args:
            grid: 要分析的网格
        """
        self.grid = grid
        self.objects = []
        
    def extract_all(self, background: int = 0) -> List[Object]:
        """
        提取网格中的所有对象
        
        Args:
            background: 背景颜色，默认为0
            
        Returns:
            提取的对象列表
        """
        self.objects = []
        visited = set()
        
        for r in range(self.grid.height):
            for c in range(self.grid.width):
                if (r, c) not in visited and self.grid.data[r, c] != background:
                    # 发现新对象，使用BFS提取
                    color = self.grid.data[r, c]
                    positions = self._extract_connected_component(r, c, color, visited)
                    obj = Object(self.grid, positions, color)
                    self.objects.append(obj)
        
        return self.objects
    
    def _extract_connected_component(self, start_r: int, start_c: int, color: int, visited: Set[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """使用BFS提取连通组件"""
        queue = [(start_r, start_c)]
        positions = []
        
        while queue:
            r, c = queue.pop(0)
            if (r, c) in visited:
                continue
                
            visited.add((r, c))
            if self.grid.data[r, c] == color:
                positions.append((r, c))
                
                # 检查四个相邻位置
                for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    nr, nc = r + dr, c + dc
                    if (0 <= nr < self.grid.height and 
                        0 <= nc < self.grid.width and 
                        (nr, nc) not in visited and
                        self.grid.data[nr, nc] == color):
                        queue.append((nr, nc))
        
        return positions
    
    def extract_with_sub_objects(self, background: int = 0, min_sub_size: int = 2) -> List[Object]:
        """提取所有对象及其子对象"""
        self.extract_all(background)
        
        for obj in self.objects:
            obj.extract_sub_objects(min_sub_size)
            
        return self.objects