import numpy as np
from typing import Dict, List, Any
from arc_framework import ARCObject, Position

class WeightManager:
    """对象权重管理器"""
    
    def __init__(self):
        self.weight_rules = {
            "size_inverse": 1.0,        # 小对象权重更大
            "color_rarity": 0.8,        # 罕见颜色权重更大
            "position_center": 0.5,     # 中心位置权重更大
            "shape_complexity": 0.7,    # 复杂形状权重更大
            "repeated_pattern": 1.2,    # 重复出现的模式权重更大
            "symmetry": 0.9,            # 对称形状权重更大
            "same_position": 1.1,       # 相同位置权重更大
            "same_row_col": 0.6,        # 同行列权重更大
            "rectangle_shape": 0.8,     # 矩形形状权重
        }
        self.color_frequency = {}       # 跟踪颜色频率
        self.position_frequency = {}    # 跟踪位置频率
        self.pattern_frequency = {}     # 跟踪模式频率
        
    def update_frequencies(self, objects: List[ARCObject], grid: np.ndarray):
        """更新各种特征的频率统计"""
        # 更新颜色频率
        rows, cols = grid.shape
        color_counts = {}
        
        for r in range(rows):
            for c in range(cols):
                color = grid[r, c]
                color_counts[color] = color_counts.get(color, 0) + 1
                
        # 将计数转换为频率
        total_cells = rows * cols
        for color, count in color_counts.items():
            self.color_frequency[color] = count / total_cells
            
        # 更新位置和模式频率
        for obj in objects:
            # 使用对象的中心点作为位置键
            center = self._calculate_center(obj.pixels)
            pos_key = (center.x, center.y)
            self.position_frequency[pos_key] = self.position_frequency.get(pos_key, 0) + 1
            
            # 为模式生成简单的指纹
            pattern_key = self._generate_pattern_key(obj)
            self.pattern_frequency[pattern_key] = self.pattern_frequency.get(pattern_key, 0) + 1
    
    def _calculate_center(self, pixels: List[Position]) -> Position:
        """计算像素集的中心点"""
        if not pixels:
            return Position(0, 0)
            
        sum_x = sum(p.x for p in pixels)
        sum_y = sum(p.y for p in pixels)
        return Position(sum_x // len(pixels), sum_y // len(pixels))
    
    def _generate_pattern_key(self, obj: ARCObject) -> str:
        """为对象生成模式键"""
        # 简单方法：使用相对坐标排序生成键
        if not obj.pixels:
            return ""
            
        # 找到最小x和y坐标
        min_x = min(p.x for p in obj.pixels)
        min_y = min(p.y for p in obj.pixels)
        
        # 生成相对坐标列表
        rel_positions = [(p.x - min_x, p.y - min_y) for p in obj.pixels]
        rel_positions.sort()
        
        # 将坐标和颜色合并为键
        return f"{obj.color}_{rel_positions}"
    
    def calculate_object_weight(self, obj: ARCObject, grid: np.ndarray) -> float:
        """计算单个对象的权重"""
        weight = 0.0
        
        # 1. 大小反比（小对象权重大）
        if "size_inverse" in self.weight_rules:
            size = len(obj.pixels)
            weight += self.weight_rules["size_inverse"] * (1.0 / max(1, size))
        
        # 2. 颜色稀有度
        if "color_rarity" in self.weight_rules and obj.color in self.color_frequency:
            rarity = 1.0 - self.color_frequency.get(obj.color, 0)
            weight += self.weight_rules["color_rarity"] * rarity
        
        # 3. 位置（中心位置权重大）
        if "position_center" in self.weight_rules:
            center = self._calculate_center(obj.pixels)
            rows, cols = grid.shape
            grid_center_x, grid_center_y = rows // 2, cols // 2
            
            # 计算到中心的归一化距离
            max_dist = (rows**2 + cols**2)**0.5 / 2
            dist = ((center.x - grid_center_x)**2 + (center.y - grid_center_y)**2)**0.5
            norm_dist = dist / max_dist if max_dist > 0 else 0
            
            # 越接近中心，权重越大
            weight += self.weight_rules["position_center"] * (1.0 - norm_dist)
        
        # 4. 形状复杂度
        if "shape_complexity" in self.weight_rules:
            # 使用周长与面积比作为复杂度的简单度量
            perimeter = self._calculate_perimeter(obj.pixels)
            area = len(obj.pixels)
            complexity = perimeter / (4 * (area**0.5)) if area > 0 else 0
            weight += self.weight_rules["shape_complexity"] * min(1.0, complexity)
        
        # 5. 重复模式
        if "repeated_pattern" in self.weight_rules:
            pattern_key = self._generate_pattern_key(obj)
            frequency = self.pattern_frequency.get(pattern_key, 0)
            # 标准化频率（假设最大出现次数为10）
            norm_freq = min(frequency / 10.0, 1.0)
            weight += self.weight_rules["repeated_pattern"] * norm_freq
        
        # 6. 对称性
        if "symmetry" in self.weight_rules:
            symmetry_score = self._calculate_symmetry(obj.pixels)
            weight += self.weight_rules["symmetry"] * symmetry_score
        
        # 7. 矩形检测
        if "rectangle_shape" in self.weight_rules:
            is_rectangle, rect_score = self._is_rectangle(obj.pixels)
            if is_rectangle:
                weight += self.weight_rules["rectangle_shape"] * rect_score
        
        return weight
    
    def _calculate_perimeter(self, pixels: List[Position]) -> int:
        """计算对象的周长"""
        if not pixels:
            return 0
            
        # 将像素转换为集合以加速查找
        pixel_set = {(p.x, p.y) for p in pixels}
        
        # 统计边界像素
        perimeter = 0
        for p in pixels:
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                if (p.x + dx, p.y + dy) not in pixel_set:
                    perimeter += 1
                    
        return perimeter
    
    def _calculate_symmetry(self, pixels: List[Position]) -> float:
        """计算对象的对称性分数"""
        if not pixels:
            return 0.0
            
        # 找到包围盒
        min_x = min(p.x for p in pixels)
        max_x = max(p.x for p in pixels)
        min_y = min(p.y for p in pixels)
        max_y = max(p.y for p in pixels)
        
        # 转换为集合以加速查找
        pixel_set = {(p.x, p.y) for p in pixels}
        
        # 检查水平对称
        h_sym = 0
        total_h = 0
        for x in range(min_x, max_x + 1):
            for y in range(min_y, max_y + 1):
                mirror_x = max_x - (x - min_x)
                total_h += 1
                if ((x, y) in pixel_set) == ((mirror_x, y) in pixel_set):
                    h_sym += 1
        
        # 检查垂直对称
        v_sym = 0
        total_v = 0
        for x in range(min_x, max_x + 1):
            for y in range(min_y, max_y + 1):
                mirror_y = max_y - (y - min_y)
                total_v += 1
                if ((x, y) in pixel_set) == ((x, mirror_y) in pixel_set):
                    v_sym += 1
        
        h_score = h_sym / total_h if total_h > 0 else 0
        v_score = v_sym / total_v if total_v > 0 else 0
        
        # 取最大的对称分数
        return max(h_score, v_score)
    
    def _is_rectangle(self, pixels: List[Position]) -> tuple:
        """检查对象是否是矩形，返回(是否矩形, 矩形分数)"""
        if not pixels:
            return (False, 0.0)
            
        # 找到包围盒
        min_x = min(p.x for p in pixels)
        max_x = max(p.x for p in pixels)
        min_y = min(p.y for p in pixels)
        max_y = max(p.y for p in pixels)
        
        # 计算包围盒内的像素数量
        box_area = (max_x - min_x + 1) * (max_y - min_y + 1)
        
        # 检查是否所有包围盒内的点都属于对象
        pixel_set = {(p.x, p.y) for p in pixels}
        filled_count = 0
        
        for x in range(min_x, max_x + 1):
            for y in range(min_y, max_y + 1):
                if (x, y) in pixel_set:
                    filled_count += 1
        
        # 计算填充率
        fill_ratio = filled_count / box_area if box_area > 0 else 0
        
        # 如果填充率接近1，那么它是矩形
        is_rect = fill_ratio > 0.95
        
        return (is_rect, fill_ratio)