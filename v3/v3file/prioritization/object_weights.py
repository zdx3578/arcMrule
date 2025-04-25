from typing import List, Dict, Tuple, Any
import numpy as np
from core.object_extractor import Object

class ObjectWeightCalculator:
    """计算对象的权重，用于优先级排序"""
    
    def __init__(self):
        """初始化权重计算器"""
        # 权重规则及其默认权重
        self.weight_rules = {
            "size_small": 1.5,          # 小对象优先
            "repeated_appearance": 1.3,  # 重复出现的对象
            "same_color": 1.2,          # 相同颜色
            "same_position": 1.4,        # 相同位置
            "same_row_col": 1.1,         # 相同行列
            "same_pattern": 1.3,         # 相同图案
            "grid_diff": 1.2,           # 差异网格中的对象
            "rectangle": 1.1,           # 长方形对象
            "frame": 1.2                # 框架对象
        }
        
    def set_rule_weight(self, rule_name: str, weight: float):
        """
        设置规则权重
        
        Args:
            rule_name: 规则名称
            weight: 权重值
        """
        if rule_name in self.weight_rules:
            self.weight_rules[rule_name] = weight
        else:
            raise ValueError(f"未知的权重规则: {rule_name}")
    
    def calculate_weights(self, objects: List[Object], 
                          diff_objects: List[Object] = None,
                          repeated_objects: Dict[int, int] = None) -> Dict[int, float]:
        """
        计算对象权重
        
        Args:
            objects: 要计算权重的对象列表
            diff_objects: 差异网格中的对象
            repeated_objects: 对象重复出现的次数统计
            
        Returns:
            对象ID到权重的映射
        """
        weights = {}
        
        for i, obj in enumerate(objects):
            weight = 1.0
            
            # 小对象权重更高（反比于大小）
            size_factor = 1.0 / max(1, obj.size)
            weight *= size_factor * self.weight_rules["size_small"]
            
            # 重复出现的对象权重更高
            if repeated_objects and obj.color in repeated_objects:
                frequency = repeated_objects[obj.color]
                if frequency > 1:
                    weight *= self.weight_rules["repeated_appearance"]
            
            # 矩形对象
            if obj.is_rectangle():
                weight *= self.weight_rules["rectangle"]
                
            # 框架对象
            if obj.is_frame():
                weight *= self.weight_rules["frame"]
            
            # 如果对象在差异网格中
            if diff_objects:
                for diff_obj in diff_objects:
                    if self._is_overlapping(obj, diff_obj):
                        weight *= self.weight_rules["grid_diff"]
                        break
            
            weights[i] = weight
        
        return weights
    
    def sort_objects_by_weight(self, objects: List[Object], weights: Dict[int, float]) -> List[Tuple[Object, float]]:
        """
        按权重排序对象
        
        Args:
            objects: 对象列表
            weights: 对象权重字典
            
        Returns:
            按权重排序的(对象, 权重)元组列表
        """
        weighted_objects = [(obj, weights.get(i, 1.0)) for i, obj in enumerate(objects)]
        return sorted(weighted_objects, key=lambda x: x[1], reverse=True)
    
    def _is_overlapping(self, obj1: Object, obj2: Object) -> bool:
        """检查两个对象是否重叠"""
        # 转换为集合进行高效的交集运算
        pos1 = set(obj1.positions)
        pos2 = set(obj2.positions)
        return len(pos1.intersection(pos2)) > 0
    
    def analyze_repeated_objects(self, all_objects: List[List[Object]]) -> Dict[int, int]:
        """
        分析多个对象列表中重复出现的对象
        
        Args:
            all_objects: 多个对象列表的列表
            
        Returns:
            颜色到出现次数的映射
        """
        color_count = {}
        
        for objects in all_objects:
            # 收集每个颜色的出现次数
            colors = set()
            for obj in objects:
                colors.add(obj.color)
            
            # 更新颜色计数
            for color in colors:
                color_count[color] = color_count.get(color, 0) + 1
        
        return color_count