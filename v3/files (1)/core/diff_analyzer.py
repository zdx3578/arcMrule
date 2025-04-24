import numpy as np
from typing import List, Dict, Tuple, Any
from .grid import Grid
from .object_extractor import Object, ObjectExtractor

class DiffAnalyzer:
    """分析输入和输出网格之间的差异"""
    
    def __init__(self, input_grid: Grid, output_grid: Grid):
        """
        初始化差异分析器
        
        Args:
            input_grid: 输入网格
            output_grid: 输出网格
        """
        self.input_grid = input_grid
        self.output_grid = output_grid
        self.diff_grid = input_grid.get_diff(output_grid)
        
        # 提取各个网格中的对象
        self.input_extractor = ObjectExtractor(input_grid)
        self.output_extractor = ObjectExtractor(output_grid)
        self.diff_extractor = ObjectExtractor(self.diff_grid)
        
        self.input_objects = []
        self.output_objects = []
        self.diff_objects = []
        
        # 存储对象映射关系
        self.object_mappings = {}
        
    def extract_all_objects(self, background: int = 0) -> Tuple[List[Object], List[Object], List[Object]]:
        """
        从输入、输出和差异网格中提取所有对象
        
        Returns:
            输入对象、输出对象和差异对象的元组
        """
        self.input_objects = self.input_extractor.extract_all(background)
        self.output_objects = self.output_extractor.extract_all(background)
        self.diff_objects = self.diff_extractor.extract_all(background)
        
        return self.input_objects, self.output_objects, self.diff_objects
    
    def analyze_object_transformations(self) -> Dict[int, Dict[str, Any]]:
        """
        分析对象的变换规则
        
        Returns:
            对象变换信息字典
        """
        transformations = {}
        
        # 颜色映射分析
        color_mapping = self._analyze_color_mapping()
        
        # 对象移动分析
        position_changes = self._analyze_position_changes()
        
        # 对象增减分析
        object_changes = self._analyze_object_changes()
        
        # 合并所有分析结果
        for obj_id, info in object_changes.items():
            transformations[obj_id] = {
                "color_change": color_mapping.get(info["original_color"], None),
                "position_change": position_changes.get(obj_id, None),
                "size_change": info.get("size_change", 0),
                "is_new": info.get("is_new", False),
                "is_removed": info.get("is_removed", False)
            }
        
        return transformations
    
    def _analyze_color_mapping(self) -> Dict[int, int]:
        """分析颜色映射规则"""
        color_mapping = {}
        input_colors = self.input_grid.get_colors()
        output_colors = self.output_grid.get_colors()
        
        # 简单的颜色频率匹配
        input_color_counts = {c: self.input_grid.count_color(c) for c in input_colors}
        output_color_counts = {c: self.output_grid.count_color(c) for c in output_colors}
        
        for in_color in input_colors:
            if in_color in output_colors:
                # 如果颜色相同，检查数量变化
                if input_color_counts[in_color] != output_color_counts[in_color]:
                    color_mapping[in_color] = in_color  # 保持颜色，但数量变化
            else:
                # 颜色完全改变，尝试基于频率匹配
                best_match = None
                min_diff = float('inf')
                
                for out_color in output_colors:
                    if out_color not in input_colors:
                        diff = abs(input_color_counts[in_color] - output_color_counts[out_color])
                        if diff < min_diff:
                            min_diff = diff
                            best_match = out_color
                
                if best_match is not None:
                    color_mapping[in_color] = best_match
        
        return color_mapping
    
    def _analyze_position_changes(self) -> Dict[int, Dict[str, Any]]:
        """分析对象位置变化"""
        position_changes = {}
        
        # 简单的位置变化分析，匹配大小和形状相似的对象
        for i, in_obj in enumerate(self.input_objects):
            best_match = None
            best_match_score = -1
            
            for j, out_obj in enumerate(self.output_objects):
                # 计算相似度得分：考虑颜色、大小和形状
                score = 0
                if in_obj.color == out_obj.color:
                    score += 2
                if abs(in_obj.size - out_obj.size) <= 2:
                    score += 1
                if in_obj.is_rectangle() and out_obj.is_rectangle():
                    score += 1
                
                if score > best_match_score:
                    best_match_score = score
                    best_match = j
            
            if best_match is not None and best_match_score > 1:
                out_obj = self.output_objects[best_match]
                position_changes[i] = {
                    "delta_row": out_obj.min_row - in_obj.min_row,
                    "delta_col": out_obj.min_col - in_obj.min_col,
                    "match_score": best_match_score,
                    "matched_object_id": best_match
                }
        
        return position_changes
    
    def _analyze_object_changes(self) -> Dict[int, Dict[str, Any]]:
        """分析对象的增减变化"""
        object_changes = {}
        
        # 分析输入对象
        for i, obj in enumerate(self.input_objects):
            object_changes[i] = {
                "original_color": obj.color,
                "original_size": obj.size
            }
        
        # 查找新增对象
        for j, obj in enumerate(self.output_objects):
            found = False
            for i, changes in object_changes.items():
                if i < len(self.input_objects) and self._is_similar_object(self.input_objects[i], obj):
                    found = True
                    changes["size_change"] = obj.size - self.input_objects[i].size
                    break
            
            if not found:
                object_changes[len(self.input_objects) + j] = {
                    "is_new": True,
                    "original_color": obj.color,
                    "original_size": obj.size
                }
        
        # 查找删除的对象
        for i, in_obj in enumerate(self.input_objects):
            found = False
            for out_obj in self.output_objects:
                if self._is_similar_object(in_obj, out_obj):
                    found = True
                    break
            
            if not found:
                object_changes[i]["is_removed"] = True
        
        return object_changes
    
    def _is_similar_object(self, obj1: Object, obj2: Object, threshold: float = 0.7) -> bool:
        """判断两个对象是否相似"""
        # 颜色相同加分
        if obj1.color != obj2.color:
            return False
        
        # 大小相似加分
        size_ratio = min(obj1.size, obj2.size) / max(obj1.size, obj2.size)
        if size_ratio < threshold:
            return False
        
        # 形状相似加分（矩形、框架）
        shape_match = (obj1.is_rectangle() == obj2.is_rectangle() and 
                       obj1.is_frame() == obj2.is_frame())
        
        return shape_match