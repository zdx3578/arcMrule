from typing import List, Dict, Any, Tuple
import numpy as np
from core.grid import Grid
from core.object_extractor import Object
from core.diff_analyzer import DiffAnalyzer

class PatternRecognizer:
    """识别网格和对象的模式"""
    
    def __init__(self):
        """初始化模式识别器"""
        self.recognized_patterns = {}
    
    def analyze_color_changes(self, input_grid: Grid, output_grid: Grid) -> Dict[int, int]:
        """
        分析颜色变化模式
        
        Args:
            input_grid: 输入网格
            output_grid: 输出网格
            
        Returns:
            从输入颜色到输出颜色的映射
        """
        color_mapping = {}
        in_colors = input_grid.get_colors()
        out_colors = output_grid.get_colors()
        
        # 基于频率的映射
        in_counts = {c: np.sum(input_grid.data == c) for c in in_colors}
        out_counts = {c: np.sum(output_grid.data == c) for c in out_colors}
        
        # 按频率排序
        in_colors_by_freq = sorted(in_colors, key=lambda c: in_counts[c], reverse=True)
        out_colors_by_freq = sorted(out_colors, key=lambda c: out_counts[c], reverse=True)
        
        # 尝试映射相同频率的颜色
        for i, in_color in enumerate(in_colors_by_freq):
            if i < len(out_colors_by_freq):
                out_color = out_colors_by_freq[i]
                if in_counts[in_color] == out_counts[out_color]:
                    color_mapping[in_color] = out_color
        
        # 分析位置对应关系
        height, width = input_grid.data.shape
        for r in range(height):
            for c in range(width):
                in_color = input_grid.data[r, c]
                out_color = output_grid.data[r, c]
                
                if in_color != out_color and in_color not in color_mapping:
                    # 检查这种映射在其他位置是否也成立
                    valid = True
                    for r2 in range(height):
                        for c2 in range(width):
                            if input_grid.data[r2, c2] == in_color and output_grid.data[r2, c2] != out_color:
                                valid = False
                                break
                    
                    if valid:
                        color_mapping[in_color] = out_color
        
        return color_mapping
    
    def analyze_positional_patterns(self, objects: List[Object]) -> Dict[str, Any]:
        """
        分析对象的位置模式
        
        Args:
            objects: 对象列表
            
        Returns:
            位置模式信息
        """
        patterns = {}
        
        # 分析行列分布
        row_distribution = {}
        col_distribution = {}
        
        for obj in objects:
            # 记录行分布
            for r in range(obj.min_row, obj.max_row + 1):
                row_distribution[r] = row_distribution.get(r, 0) + 1
            
            # 记录列分布
            for c in range(obj.min_col, obj.max_col + 1):
                col_distribution[c] = col_distribution.get(c, 0) + 1
        
        # 检查行列的规律性
        row_pattern = self._check_distribution_pattern(row_distribution)
        col_pattern = self._check_distribution_pattern(col_distribution)
        
        patterns["row_pattern"] = row_pattern
        patterns["col_pattern"] = col_pattern
        
        # 检查对象之间的相对位置
        if len(objects) > 1:
            relative_positions = []
            for i, obj1 in enumerate(objects[:-1]):
                for obj2 in objects[i+1:]:
                    delta_row = obj2.min_row - obj1.min_row
                    delta_col = obj2.min_col - obj1.min_col
                    relative_positions.append((delta_row, delta_col))
            
            # 检查相对位置是否有规律
            if len(set(relative_positions)) == 1:
                patterns["uniform_spacing"] = relative_positions[0]
            
        return patterns
    
    def _check_distribution_pattern(self, distribution: Dict[int, int]) -> Dict[str, Any]:
        """
        检查分布的规律性
        
        Args:
            distribution: 位置到数量的映射
            
        Returns:
            分布模式信息
        """
        positions = sorted(distribution.keys())
        if not positions:
            return {"type": "none"}
        
        # 检查等距分布
        if len(positions) > 2:
            diffs = [positions[i+1] - positions[i] for i in range(len(positions) - 1)]
            if len(set(diffs)) == 1:
                return {
                    "type": "uniform",
                    "spacing": diffs[0],
                    "start": positions[0]
                }
        
        # 检查递增/递减模式
        counts = [distribution[p] for p in positions]
        if counts == sorted(counts):
            return {"type": "increasing"}
        elif counts == sorted(counts, reverse=True):
            return {"type": "decreasing"}
        
        return {"type": "irregular"}
    
    def find_common_patterns(self, examples: List[Tuple[Grid, Grid]]) -> Dict[str, Any]:
        """
        在多个例子中寻找共同模式
        
        Args:
            examples: 输入输出网格对的列表
            
        Returns:
            共同模式信息
        """
        common_patterns = {}
        
        # 分析每个例子的颜色变化
        color_mappings = []
        for input_grid, output_grid in examples:
            color_mappings.append(self.analyze_color_changes(input_grid, output_grid))
        
        # 寻找共同的颜色映射规则
        common_color_mapping = self._find_common_color_mappings(color_mappings)
        if common_color_mapping:
            common_patterns["color_mapping"] = common_color_mapping
        
        # 分析每个例子的差异
        diff_analyses = []
        for input_grid, output_grid in examples:
            analyzer = DiffAnalyzer(input_grid, output_grid)
            analyzer.extract_all_objects()
            transformations = analyzer.analyze_object_transformations()
            diff_analyses.append(transformations)
        
        # 寻找共同的变换模式
        common_transformations = self._find_common_transformations(diff_analyses)
        if common_transformations:
            common_patterns["transformations"] = common_transformations
        
        return common_patterns
    
    def _find_common_color_mappings(self, color_mappings: List[Dict[int, int]]) -> Dict[str, Any]:
        """
        寻找共同的颜色映射规则
        
        Args:
            color_mappings: 颜色映射列表
            
        Returns:
            共同颜色映射规则
        """
        if not color_mappings:
            return {}
        
        # 检查每个映射是否在所有例子中都一致
        common_mapping = {}
        all_in_colors = set()
        for mapping in color_mappings:
            all_in_colors.update(mapping.keys())
        
        for in_color in all_in_colors:
            out_colors = []
            for mapping in color_mappings:
                if in_color in mapping:
                    out_colors.append(mapping[in_color])
            
            if len(set(out_colors)) == 1 and out_colors:
                common_mapping[in_color] = out_colors[0]
        
        # 检查是否有特殊规则，如颜色交换、颜色递增等
        if len(common_mapping) >= 2:
            # 检查是否是颜色交换
            in_colors = list(common_mapping.keys())
            out_colors = list(common_mapping.values())
            if set(in_colors) == set(out_colors) and len(in_colors) == len(out_colors):
                return {
                    "type": "swap",
                    "mapping": common_mapping
                }
            
            # 检查是否是颜色递增/递减
            if all(isinstance(c, int) for c in in_colors) and all(isinstance(c, int) for c in out_colors):
                diffs = [common_mapping[c] - c for c in in_colors]
                if len(set(diffs)) == 1:
                    return {
                        "type": "offset",
                        "value": diffs[0],
                        "mapping": common_mapping
                    }
        
        return {"type": "direct", "mapping": common_mapping} if common_mapping else {}
    
    def _find_common_transformations(self, transformations_list: List[Dict[int, Dict[str, Any]]]) -> Dict[str, Any]:
        """
        寻找共同的变换模式
        
        Args:
            transformations_list: 变换信息列表
            
        Returns:
            共同变换模式
        """
        if not transformations_list:
            return {}
        
        # 统计各种变换的出现频率
        transformation_types = {}
        
        for transformations in transformations_list:
            for obj_id, info in transformations.items():
                # 分类变换类型
                if info.get("is_new", False):
                    key = "object_creation"
                elif info.get("is_removed", False):
                    key = "object_deletion"
                elif info.get("color_change") is not None:
                    key = "color_change"
                elif info.get("position_change") is not None:
                    key = "position_change"
                elif info.get("size_change", 0) != 0:
                    key = "size_change"
                else:
                    key = "no_change"
                
                transformation_types[key] = transformation_types.