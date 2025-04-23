import numpy as np
from typing import List, Dict, Tuple, Any
from arc_framework import ARCObject, RulePattern, ActionType, DiffGrid

class RuleInference:
    """规则推断器，从输入输出对中提取规则"""
    
    def __init__(self):
        self.known_patterns = []
        self.confidence_threshold = 0.7
    
    def infer_rules(self, 
                   input_objects: List[ARCObject], 
                   output_objects: List[ARCObject], 
                   diff_grid: DiffGrid) -> List[RulePattern]:
        """推断输入输出对象之间的规则"""
        inferred_rules = []
        
        # 1. 尝试对象匹配
        object_matches = self._match_objects(input_objects, output_objects)
        
        # 2. 对每对匹配的对象分析变换
        for in_obj, out_obj in object_matches:
            # 分析对象变换
            rules = self._analyze_transformation(in_obj, out_obj, diff_grid)
            inferred_rules.extend(rules)
        
        # 3. 分析整体变换（不限于对象级别）
        global_rules = self._analyze_global_transformation(input_objects, output_objects, diff_grid)
        inferred_rules.extend(global_rules)
        
        return inferred_rules
    
    def _match_objects(self, input_objs: List[ARCObject], output_objs: List[ARCObject]) -> List[Tuple[ARCObject, ARCObject]]:
        """匹配输入和输出中对应的对象"""
        matches = []
        
        # 创建一个已匹配输出对象的集合
        matched_outputs = set()
        
        # 对于每个输入对象，找到最佳匹配的输出对象
        for in_obj in input_objs:
            best_match = None
            best_score = 0
            
            for out_obj in output_objs:
                if id(out_obj) in matched_outputs:
                    continue  # 跳过已匹配的输出对象
                
                # 计算匹配分数
                score = self._calculate_match_score(in_obj, out_obj)
                
                if score > best_score:
                    best_score = score
                    best_match = out_obj
            
            # 如果找到足够好的匹配，添加到结果中
            if best_match and best_score > self.confidence_threshold:
                matches.append((in_obj, best_match))
                matched_outputs.add(id(best_match))
        
        return matches
    
    def _calculate_match_score(self, obj1: ARCObject, obj2: ARCObject) -> float:
        """计算两个对象之间的匹配分数"""
        # 简单版本：基于颜色、大小和形状相似度
        score = 0.0
        
        # 颜色匹配度
        if obj1.color == obj2.color:
            score += 0.3
        
        # 大小匹配度（基于像素数量）
        size_ratio = min(len(obj1.pixels) / len(obj2.pixels), len(obj2.pixels) / len(obj1.pixels)) if len(obj1.pixels) > 0 and len(obj2.pixels) > 0 else 0
        score += 0.3 * size_ratio
        
        # 形状匹配度（基于像素相对位置的相似度）
        shape_sim = self._calculate_shape_similarity(obj1, obj2)
        score += 0.4 * shape_sim
        
        return score
    
    def _calculate_shape_similarity(self, obj1: ARCObject, obj2: ARCObject) -> float:
        """计算两个对象形状的相似度"""
        # 计算相对坐标
        if not obj1.pixels or not obj2.pixels:
            return 0.0
            
        # 对象1的中心
        center1_x = sum(p.x for p in obj1.pixels) / len(obj1.pixels)
        center1_y = sum(p.y for p in obj1.pixels) / len(obj1.pixels)
        
        # 对象2的中心
        center2_x = sum(p.x for p in obj2.pixels) / len(obj2.pixels)
        center2_y = sum(p.y for p in obj2.pixels) / len(obj2.pixels)
        
        # 计算相对坐标集
        rel_pixels1 = set((p.x - center1_x, p.y - center1_y) for p in obj1.pixels)
        rel_pixels2 = set((p.x - center2_x, p.y - center2_y) for p in obj2.pixels)
        
        # 计算交集大小
        intersection = rel_pixels1.intersection(rel_pixels2)
        
        # 使用Jaccard相似度
        union = rel_pixels1.union(rel_pixels2)
        similarity = len(intersection) / len(union) if union else 0
        
        return similarity
    
    def _analyze_transformation(self, in_obj: ARCObject, out_obj: ARCObject, diff_grid: DiffGrid) -> List[RulePattern]:
        """分析两个匹配对象之间的变换"""
        rules = []
        
        # 检测移动
        move_rule = self._detect_movement(in_obj, out_obj)
        if move_rule:
            rules.append(move_rule)
        
        # 检测颜色变化
        color_rule = self._detect_color_change(in_obj, out_obj)
        if color_rule:
            rules.append(color_rule)
        
        # 检测旋转
        rotation_rule = self._detect_rotation(in_obj, out_obj)
        if rotation_rule:
            rules.append(rotation_rule)
        
        # 检测翻转
        flip_rule = self._detect_flip(in_obj, out_obj)
        if flip_rule:
            rules.append(flip_rule)
        
        # 检测大小变化
        resize_rule = self._detect_resize(in_obj, out_obj)
        if resize_rule:
            rules.append(resize_rule)
        
        return rules
    
    def _detect_movement(self, in_obj: ARCObject, out_obj: ARCObject) -> RulePattern:
        """检测移动变换"""
        # 计算输入和输出对象的中心点
        in_center_x = sum(p.x for p in in_obj.pixels) / len(in_obj.pixels)
        in_center_y = sum(p.y for p in in_obj.pixels) / len(in_obj.pixels)
        
        out_center_x = sum(p.x for p in out_obj.pixels) / len(out_obj.pixels)
        out_center_y = sum(p.y for p in out_obj.pixels) / len(out_obj.pixels)
        
        # 计算移动量，四舍五入到整数
        dx = round(out_center_x - in_center_x)
        dy = round(out_center_y - in_center_y)
        
        # 如果有显著移动，创建规则
        if abs(dx) > 0 or abs(dy) > 0:
            rule = RulePattern(f"移动规则", f"将对象移动 dx={dx}, dy={dy}")
            rule.add_action(ActionType.MOVE, {"dx": dx, "dy": dy})
            
            # 添加简单条件：匹配同类型的对象
            def match_condition(obj, grid):
                return obj.type == in_obj.type and obj.color == in_obj.color
                
            rule.add_condition(match_condition, f"匹配颜色为{in_obj.color}的{in_obj.type}对象")
            
            return rule
        
        return None
    
    def _detect_color_change(self, in_obj: ARCObject, out_obj: ARCObject) -> RulePattern:
        """检测颜色变化"""
        if in_obj.color != out_obj.color:
            rule = RulePattern(f"颜色变化规则", f"将颜色从{in_obj.color}改为{out_obj.color}")
            rule.add_action(ActionType.COLOR_CHANGE, {"color": out_obj.color})
            
            # 添加简单条件：匹配特定颜色的对象
            def match_condition(obj, grid):
                return obj.color == in_obj.color
                
            rule.add_condition(match_condition, f"匹配颜色为{in_obj.color}的对象")
            
            return rule
        
        return None
    
    def _detect_rotation(self, in_obj: ARCObject, out_obj: ARCObject) -> RulePattern:
        """检测旋转变换"""
        # 实现旋转检测的逻辑
        # ...
        
        return None
    
    def _detect_flip(self, in_obj: ARCObject, out_obj: ARCObject) -> RulePattern:
        """检测翻转变换"""
        # 实现翻转检测的逻辑
        # ...
        
        return None
    
    def _detect_resize(self, in_obj: ARCObject, out_obj: ARCObject) -> RulePattern:
        """检测大小变化"""
        # 实现大小变化检测的逻辑
        # ...
        
        return None
    
    def _analyze_global_transformation(self, input_objects: List[ARCObject], 
                                     output_objects: List[ARCObject], 
                                     diff_grid: DiffGrid) -> List[RulePattern]:
        """分析全局变换规则"""
        global_rules = []
        
        # 检测整体模式，如填充、对称性等
        
        return global_rules