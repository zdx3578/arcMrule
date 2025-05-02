"""
规则应用模块

负责将识别出的模式和规则应用到新数据上。
"""

from typing import List, Dict, Any, Callable, Optional


class RuleApplier:
    """处理规则应用的类"""
    
    def __init__(self, debug_print=None):
        """
        初始化规则应用器
        
        Args:
            debug_print: 调试打印函数（可选）
        """
        self.debug_print = debug_print
        
    def apply_patterns(self, input_grid, common_patterns, input_obj_infos, debug=False):
        """
        将识别的共有模式应用到输入网格上
        
        Args:
            input_grid: 输入网格
            common_patterns: 共有模式字典
            input_obj_infos: 输入对象信息列表
            debug: 是否启用调试模式
            
        Returns:
            预测的输出网格
        """
        # 创建输出网格（初始为输入的副本）
        output_grid = [list(row) for row in input_grid]
        height, width = len(input_grid), len(input_grid[0])
        
        # 创建2D的转换记录，跟踪每个位置是否已被转换
        transformed = [[False for _ in range(width)] for _ in range(height)]
        
        # 按权重对对象排序（权重高的优先处理）
        sorted_obj_infos = sorted(input_obj_infos, key=lambda x: x.obj_weight, reverse=True)
        
        # 1. 应用颜色映射，优先考虑高权重映射
        if "color_mappings" in common_patterns:
            # 获取颜色映射并按加权置信度排序
            color_mappings = common_patterns["color_mappings"].get("mappings", {})
            sorted_mappings = sorted(
                [(from_color, mapping) for from_color, mapping in color_mappings.items()],
                key=lambda x: x[1].get("weighted_confidence", 0),
                reverse=True
            )

            for from_color, mapping in sorted_mappings:
                to_color = mapping["to_color"]
                cells_changed = 0

                # 根据对象权重应用颜色映射
                for obj_info in sorted_obj_infos:
                    for val, (i, j) in obj_info.original_obj:
                        if val == from_color:
                            output_grid[i][j] = to_color
                            transformed[i][j] = True
                            cells_changed += 1

                if debug and cells_changed > 0 and self.debug_print:
                    weighted_conf = mapping.get("weighted_confidence", 0)
                    self.debug_print(f"应用颜色映射: {from_color} -> {to_color}, 加权置信度: {weighted_conf:.2f}, 改变了 {cells_changed} 个单元格")

        # 2. 应用颜色模式（如颜色偏移）
        if "color_mappings" in common_patterns:
            for pattern in common_patterns["color_mappings"].get("patterns", []):
                if pattern["type"] == "color_offset" and pattern.get("weighted_score", 0) > 1:
                    offset = pattern["offset"]
                    cells_changed = 0

                    # 优先处理高权重对象
                    for obj_info in sorted_obj_infos:
                        for val, (i, j) in obj_info.original_obj:
                            if val != 0 and not transformed[i][j]:  # 不处理背景和已转换的单元格
                                try:
                                    new_val = (int(val) + offset) % 10  # 假设颜色范围是0-9
                                    output_grid[i][j] = new_val
                                    transformed[i][j] = True
                                    cells_changed += 1
                                except (ValueError, TypeError):
                                    # 跳过无法进行数值运算的颜色
                                    pass

                    if debug and cells_changed > 0 and self.debug_print:
                        weighted_score = pattern.get("weighted_score", 0)
                        self.debug_print(f"应用颜色偏移: +{offset}, 加权得分: {weighted_score:.2f}, 改变了 {cells_changed} 个单元格")

        # 3. 应用位置变化，优先考虑高权重变化
        position_changes = common_patterns.get("position_changes", [])
        if position_changes:
            # 找到最高加权得分的位置变化
            best_position_change = max(position_changes, key=lambda x: x.get("weight_score", 0))

            if best_position_change["type"] == "absolute_position" and best_position_change.get("weight_score", 0) > 1:
                # 应用绝对位置变化
                dr, dc = best_position_change["delta_row"], best_position_change["delta_col"]

                # 创建临时网格保存结果
                temp_grid = [[0 for _ in range(width)] for _ in range(height)]
                cells_moved = 0

                # 对每个对象应用位置变化，优先处理高权重对象
                for obj_info in sorted_obj_infos:
                    # 对象中的每个像素
                    for val, (r, c) in obj_info.original_obj:
                        nr, nc = int(r + dr), int(c + dc)
                        if 0 <= nr < height and 0 <= nc < width:
                            temp_grid[nr][nc] = val  # 保留原始颜色
                            cells_moved += 1

                # 合并结果到最终输出网格
                for i in range(height):
                    for j in range(width):
                        if temp_grid[i][j] != 0:  # 只覆盖非零值
                            output_grid[i][j] = temp_grid[i][j]
                            transformed[i][j] = True

                if debug and cells_moved > 0 and self.debug_print:
                    weight_score = best_position_change.get("weight_score", 0)
                    self.debug_print(f"应用位置变化: ({dr}, {dc}), 加权得分: {weight_score:.2f}, 移动了 {cells_moved} 个单元格")

        # 4. 应用形状变换（如果有）
        shape_transformations = common_patterns.get("shape_transformations", [])
        if shape_transformations:
            best_shape_transform = max(shape_transformations, key=lambda x: x.get("weight_score", 0))
            
            # 目前形状变换与位置变化的应用方式类似，可以在此基础上扩展特殊的形状变换处理
            if debug and self.debug_print:
                self.debug_print(f"发现形状变换: {best_shape_transform['transform_type']}-{best_shape_transform['transform_name']}")
                
        return output_grid