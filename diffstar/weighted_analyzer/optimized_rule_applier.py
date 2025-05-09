"""
优化规则应用器

负责将从模式分析中提取的优化执行计划应用于输入网格。
支持全局操作规则、条件规则和复合规则。
"""

from collections import defaultdict
import numpy as np
from typing import List, Dict, Any, Tuple, Set, Optional, Union

class OptimizedRuleApplier:
    """
    优化规则应用器：执行基于优化计划的网格转换
    """

    def __init__(self, debug=False, debug_print=None):
        """
        初始化优化规则应用器

        Args:
            debug: 是否启用调试模式
            debug_print: 调试打印函数
        """
        self.debug = debug
        self.debug_print = debug_print or (lambda x: print(x) if debug else None)

    def apply_optimized_plan(self, input_grid, optimized_plan, input_objects, background_color=0):
        """
        应用优化执行计划转换输入网格

        Args:
            input_grid: 输入网格
            optimized_plan: 优化执行计划
            input_objects: 输入网格中提取的对象
            background_color: 背景颜色，默认为0

        Returns:
            预测的输出网格
        """
        if self.debug:
            self.debug_print("应用优化执行计划")
            self.debug_print(f"计划步骤数: {len(optimized_plan.get('steps', []))}")

        # 创建输出网格副本
        output_grid = [list(row) for row in input_grid]

        # 创建对象映射 (shape_hash -> 对象列表, color -> 对象列表)
        shape_to_objects = defaultdict(list)
        color_to_objects = defaultdict(list)

        # 映射对象到其形状哈希和颜色
        for obj in input_objects:
            # 计算形状哈希
            shape_hash = self._calculate_object_shape_hash(obj)
            if shape_hash:
                shape_to_objects[shape_hash].append(obj)

            # 获取对象颜色
            color = obj.main_color if hasattr(obj, 'main_color') else None
            if color is not None:
                color_to_objects[color].append(obj)

        # 按优先级排序步骤
        steps = sorted(
            optimized_plan.get('steps', []),
            key=lambda x: x.get('priority', 0),
            reverse=True
        )

        # 执行每个步骤
        for step in steps:
            step_type = step.get('step_type')
            action = step.get('action', {})

            if self.debug:
                self.debug_print(f"执行步骤: {step_type}")

            if step_type == 'apply_global_rule':
                # 应用全局规则 (如: 移除所有特定颜色的对象)
                color = action.get('apply_to_color')
                operation = action.get('operation')

                if color is not None and operation == 'removed':
                    self._remove_objects_by_color(output_grid, color_to_objects[color], background_color)

            elif step_type == 'apply_conditional_rule':
                # 应用条件规则 (如: 当移除特定形状时，改变特定颜色)
                shape_hash = action.get('when_removed_shape_change_color')
                from_color = action.get('color_change', {}).get('from')
                to_color = action.get('color_change', {}).get('to')

                if shape_hash and from_color is not None and to_color is not None:
                    # 首先移除匹配形状的对象
                    # removed = self._remove_objects_by_shape(output_grid, shape_to_objects[shape_hash], background_color)

                    # if removed:
                    #     # 然后改变颜色
                    self._change_color_in_grid(output_grid, from_color, to_color)

            elif step_type == 'apply_composite_rule':
                # 应用复合规则
                base_operation = action.get('base_operation', {})
                conditional_changes = action.get('conditional_changes', [])

                # 应用基础操作
                if base_operation.get('type') == 'global_color_operation':
                    color = base_operation.get('color')
                    operation = base_operation.get('operation')

                    if color is not None and operation == 'removed':
                        self._remove_objects_by_color(output_grid, color_to_objects[color], background_color)

                # 应用条件变化
                for change in conditional_changes:
                    shape_hash = change.get('when_removed_shape_change_color')
                    from_color = change.get('color_change', {}).get('from')
                    to_color = change.get('color_change', {}).get('to')

                    if shape_hash and from_color is not None and to_color is not None:
                        # ! error code
                        # removed = self._remove_objects_by_shape(output_grid, shape_to_objects[shape_hash], background_color)

                        if removed:
                            # 改变颜色
                            self._change_color_in_grid(output_grid, from_color, to_color)

        if self.debug:
            self.debug_print("优化执行计划应用完成")

        # 将列表转换回元组
        # return tuple(tuple(row) for row in output_grid)
        return output_grid

    def apply_transformation_rules(self, input_grid, common_patterns, input_objects,
                                  transformation_rules=None, traditional_rule_applier=None, background_color=0):
        """
        应用转换规则，支持优化执行计划和传统规则

        Args:
            input_grid: 输入网格
            common_patterns: 识别的共有模式
            input_objects: 输入网格中提取的对象
            transformation_rules: 可选，特定的转换规则列表
            traditional_rule_applier: 可选，传统规则应用器
            background_color: 背景颜色，默认为0

        Returns:
            预测的输出网格
        """
        # 检查是否有优化执行计划
        has_optimized_plan = isinstance(common_patterns, dict) and 'optimized_plan' in common_patterns

        if has_optimized_plan:
            return self.apply_optimized_plan(
                input_grid, common_patterns['optimized_plan'], input_objects, background_color
            )
        elif traditional_rule_applier:
            # 委托给传统规则应用器
            return traditional_rule_applier.apply_transformation_rules(
                input_grid, common_patterns, input_objects, transformation_rules
            )
        else:
            if self.debug:
                self.debug_print("警告：既没有优化计划，也没有传统规则应用器，返回原始网格")
            return input_grid

    def _calculate_object_shape_hash(self, obj):
        """计算对象的形状哈希"""
        try:
            if hasattr(obj, 'obj_000'):
                return hash(tuple(map(tuple, obj.obj_000)))
            return None
        except (TypeError, AttributeError):
            return None

    def _remove_objects_by_color(self, grid, objects, background_color=0):
        """从网格中移除指定颜色的对象"""
        removed = False
        for obj in objects:
            removed = self._remove_object_from_grid(grid, obj, background_color) or removed
        return removed

    def _remove_objects_by_shape(self, grid, objects, background_color=0):
        """从网格中移除指定形状的对象"""
        removed = False
        for obj in objects:
            removed = self._remove_object_from_grid(grid, obj, background_color) or removed
        return removed

    def _remove_object_from_grid(self, grid, objinfo, background_color=0):
        """从网格中移除单个对象"""
        # if not hasattr(obj, 'full_obj_mask') or not hasattr(obj, 'top') or not hasattr(obj, 'left'):
        #     return False

        removed = False
        obj = objinfo.obj #_000 if hasattr(objinfo, 'obj_000') else objinfo.obj

        # 直接遍历frozenset中的元素
        for _, (x, y) in obj:
            # 将坐标对应的网格位置设为背景色
            if 0 <= x < len(grid) and 0 <= y < len(grid[0]):
                grid[x][y] = background_color
                removed = True

        return removed

    def _change_color_in_grid(self, grid, from_color, to_color):
        """在网格中将一种颜色更改为另一种颜色"""
        changed = False
        for i in range(len(grid)):
            for j in range(len(grid[i])):
                if grid[i][j] == from_color:
                    grid[i][j] = to_color
                    changed = True
        return changed