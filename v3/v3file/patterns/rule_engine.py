from typing import List, Dict, Tuple, Any, Callable
import numpy as np
from core.grid import Grid
from core.object_extractor import Object

class Rule:
    """表示一个转换规则"""
    
    def __init__(self, name: str, description: str, apply_func: Callable, confidence: float = 1.0):
        """
        初始化规则
        
        Args:
            name: 规则名称
            description: 规则描述
            apply_func: 应用规则的函数
            confidence: 规则的置信度
        """
        self.name = name
        self.description = description
        self.apply_func = apply_func
        self.confidence = confidence
        self.use_count = 0
        self.success_count = 0
    
    def apply(self, grid: Grid, *args, **kwargs) -> Tuple[Grid, bool]:
        """
        应用规则到网格
        
        Args:
            grid: 输入网格
            args, kwargs: 传递给应用函数的参数
            
        Returns:
            转换后的网格和是否成功应用的标志
        """
        self.use_count += 1
        try:
            result = self.apply_func(grid, *args, **kwargs)
            self.success_count += 1
            return result, True
        except Exception as e:
            print(f"应用规则 '{self.name}' 失败: {e}")
            return grid, False
    
    def get_success_rate(self) -> float:
        """获取规则的成功率"""
        if self.use_count == 0:
            return 0.0
        return self.success_count / self.use_count
    
    def __str__(self) -> str:
        return f"Rule(name='{self.name}', confidence={self.confidence:.2f}, success_rate={self.get_success_rate():.2f})"

class RuleEngine:
    """规则引擎，管理和应用转换规则"""
    
    def __init__(self):
        """初始化规则引擎"""
        self.rules = {}  # 规则名称到规则对象的映射
        self._initialize_basic_rules()
    
    def _initialize_basic_rules(self):
        """初始化基本规则集"""
        # 颜色映射规则
        def color_mapping_rule(grid, from_color, to_color):
            new_grid = Grid(grid.data.copy())
            new_grid.data[new_grid.data == from_color] = to_color
            return new_grid
        
        self.add_rule(
            name="color_mapping",
            description="将一种颜色映射到另一种颜色",
            apply_func=color_mapping_rule
        )
        
        # 对象平移规则
        def object_translation_rule(grid, objects, delta_row, delta_col):
            new_data = grid.data.copy()
            
            # 创建掩码，标记所有对象的位置
            mask = np.zeros_like(new_data, dtype=bool)
            for obj in objects:
                for r, c in obj.positions:
                    mask[r, c] = True
            
            # 保存对象颜色
            colors = {}
            for obj in objects:
                for r, c in obj.positions:
                    colors[(r, c)] = new_data[r, c]
            
            # 清除原始位置
            new_data[mask] = 0
            
            # 移动到新位置
            for obj in objects:
                for r, c in obj.positions:
                    nr, nc = r + delta_row, c + delta_col
                    if 0 <= nr < grid.height and 0 <= nc < grid.width:
                        new_data[nr, nc] = colors[(r, c)]
            
            return Grid(new_data)
        
        self.add_rule(
            name="object_translation",
            description="平移对象",
            apply_func=object_translation_rule
        )
        
        # 对象复制规则
        def object_copy_rule(grid, objects, copies, delta_row, delta_col):
            new_data = grid.data.copy()
            
            for _ in range(copies):
                for obj in objects:
                    for r, c in obj.positions:
                        nr, nc = r + delta_row, c + delta_col
                        if 0 <= nr < grid.height and 0 <= nc < grid.width:
                            new_data[nr, nc] = new_data[r, c]
            
            return Grid(new_data)
        
        self.add_rule(
            name="object_copy",
            description="复制对象",
            apply_func=object_copy_rule
        )
        
        # 对象旋转规则
        def object_rotation_rule(grid, objects, k=1):
            new_data = grid.data.copy()
            
            for obj in objects:
                # 提取对象的边界框
                box = obj.get_bounding_box()
                # 旋转边界框
                rotated_box = np.rot90(box, k)
                
                # 清除原始位置
                for r, c in obj.positions:
                    new_data[r, c] = 0
                
                # 放置旋转后的对象
                h, w = rotated_box.shape
                top_r, left_c = obj.min_row, obj.min_col
                
                for dr in range(h):
                    for dc in range(w):
                        if rotated_box[dr, dc] > 0:
                            r, c = top_r + dr, left_c + dc
                            if 0 <= r < grid.height and 0 <= c < grid.width:
                                new_data[r, c] = rotated_box[dr, dc]
            
            return Grid(new_data)
        
        self.add_rule(
            name="object_rotation",
            description="旋转对象",
            apply_func=object_rotation_rule
        )
    
    def add_rule(self, name: str, description: str, apply_func: Callable, confidence: float = 1.0) -> Rule:
        """
        添加新规则
        
        Args:
            name: 规则名称
            description: 规则描述
            apply_func: 应用规则的函数
            confidence: 规则的置信度
            
        Returns:
            创建的规则对象
        """
        rule = Rule(name, description, apply_func, confidence)
        self.rules[name] = rule
        return rule
    
    def get_rule(self, name: str) -> Rule:
        """获取规则"""
        if name in self.rules:
            return self.rules[name]
        raise ValueError(f"未找到规则: {name}")
    
    def apply_rule(self, rule_name: str, grid: Grid, *args, **kwargs) -> Tuple[Grid, bool]:
        """
        应用指定规则
        
        Args:
            rule_name: 规则名称
            grid: 输入网格
            args, kwargs: 传递给规则的参数
            
        Returns:
            转换后的网格和是否成功应用的标志
        """
        rule = self.get_rule(rule_name)
        return rule.apply(grid, *args, **kwargs)
    
    def get_sorted_rules(self) -> List[Rule]:
        """获取按置信度排序的规则列表"""
        return sorted(self.rules.values(), key=lambda r: r.confidence * r.get_success_rate(), reverse=True)