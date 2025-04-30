import numpy as np
from typing import List, Dict, Tuple, Any, Set, FrozenSet, Optional, Union
from collections import defaultdict
import json
import os
import matplotlib.pyplot as plt
import copy

# 导入现有函数和类
from objutil import pureobjects_from_grid, objects_fromone_params, shift_pure_obj_to_0_0_0
from objutil import uppermost, leftmost, lowermost, rightmost, palette, extend_obj
from weightgird import grid2grid_fromgriddiff, apply_color_matching_weights, display_weight_grid, display_matrices
from arc_diff_analyzer import ARCDiffAnalyzer, ObjInfo, JSONSerializer


# 修改ObjInfo类，添加权重属性
class WeightedObjInfo(ObjInfo):
    """增强型对象信息类，存储对象的所有相关信息、变换和权重"""

    def __init__(self, pair_id, in_or_out, obj, obj_params=None, grid_hw=None, background=0):
        """
        初始化对象信息

        Args:
            pair_id: 训练对ID
            in_or_out: 'in', 'out', 'diff_in', 'diff_out'等
            obj: 原始对象 (frozenset形式)
            obj_params: 对象参数 (univalued, diagonal, without_bg)
            grid_hw: 网格尺寸 [height, width]
            background: 背景值
        """
        # 调用父类初始化
        super().__init__(pair_id, in_or_out, obj, obj_params, grid_hw, background)

        # 新增：对象权重属性
        self.obj_weight = 0  # 初始权重为0，后续根据各种规则增加权重

    def to_dict(self):
        """转换为可序列化的字典表示"""
        result = super().to_dict()
        result["obj_weight"] = self.obj_weight  # 添加权重到字典表示
        return result

    def increase_weight(self, amount):
        """增加对象权重"""
        self.obj_weight += amount
        return self.obj_weight

    def set_weight(self, value):
        """设置对象权重"""
        self.obj_weight = value
        return self.obj_weight


class WeightedARCDiffAnalyzer(ARCDiffAnalyzer):
    """
    扩展ARCDiffAnalyzer，整合对象权重系统，优化对象分析
    """

    def __init__(self, debug=True, debug_dir="debug_output", pixel_threshold_pct=60,
                 weight_increment=1, diff_weight_increment=2):
        """
        初始化加权分析器

        Args:
            debug: 是否启用调试模式
            debug_dir: 调试信息输出目录
            pixel_threshold_pct: 颜色占比阈值（百分比），超过此阈值的颜色视为背景
            weight_increment: 对象权重增量
            diff_weight_increment: 差异区域权重增量
        """
        # 调用父类初始化
        super().__init__(debug, debug_dir)

        # 权重相关参数
        self.pixel_threshold_pct = pixel_threshold_pct

        self.weight1 = 1
        self.weight2 = 2
        self.weight3 = 3
        self.weight4 = 4
        self.weight5 = 5

        self.weight_increment = weight_increment
        self.diff_weight_increment = diff_weight_increment

        # 保存颜色映射统计
        self.color_statistics = {}

        # 重写对象存储结构，使用WeightedObjInfo替代ObjInfo
        self.all_objects = {
            'input': [],  # [(pair_id, [WeightedObjInfo]), ...]
            'output': [], # [(pair_id, [WeightedObjInfo]), ...]
            'diff_in': [], # [(pair_id, [WeightedObjInfo]), ...]
            'diff_out': []  # [(pair_id, [WeightedObjInfo]), ...]
        }

    def add_train_pair(self, pair_id, input_grid, output_grid, param):
        """
        添加一对训练数据，提取对象并计算权重

        Args:
            pair_id: 训练对ID
            input_grid: 输入网格
            output_grid: 输出网格
        """
        if self.debug:
            self._debug_print(f"处理训练对 {pair_id}")
            self._debug_save_grid(input_grid, f"input_{pair_id}")
            self._debug_save_grid(output_grid, f"output_{pair_id}")

        # 确保网格是元组的元组格式
        if isinstance(input_grid, list):
            input_grid = tuple(tuple(row) for row in input_grid)
        if isinstance(output_grid, list):
            output_grid = tuple(tuple(row) for row in output_grid)

        # 保存原始网格对
        self.train_pairs.append((input_grid, output_grid))

        # 计算差异网格
        diff_in, diff_out = grid2grid_fromgriddiff(input_grid, output_grid)
        self.diff_pairs.append((diff_in, diff_out))

        if self.debug:
            self._debug_save_grid(diff_in, f"diff_in_{pair_id}")
            self._debug_save_grid(diff_out, f"diff_out_{pair_id}")

        # 获取网格尺寸
        height_in, width_in = len(input_grid), len(input_grid[0])
        height_out, width_out = len(output_grid), len(output_grid[0])

        # 提取对象
        input_objects = pureobjects_from_grid(
            param, pair_id, 'in', input_grid, [height_in, width_in]
        )
        output_objects = pureobjects_from_grid(
            param, pair_id, 'out', output_grid, [height_out, width_out]
        )

        # 转换为加权对象信息
        input_obj_infos = [
            WeightedObjInfo(pair_id, 'in', obj, obj_params=None, grid_hw=[height_in, width_in])
            for obj in input_objects
        ]

        output_obj_infos = [
            WeightedObjInfo(pair_id, 'out', obj, obj_params=None, grid_hw=[height_out, width_out])
            for obj in output_objects
        ]

        if self.debug:
            self._debug_print(f"从输入网格提取了 {len(input_obj_infos)} 个对象")
            self._debug_print(f"从输出网格提取了 {len(output_obj_infos)} 个对象")

        # 为diff网格也提取对象
        if diff_in is not None and diff_out is not None:
            height_diff, width_diff = len(diff_in), len(diff_in[0])
            diff_in_objects = pureobjects_from_grid(
                param, pair_id, 'diff_in', diff_in, [height_diff, width_diff]
            )
            diff_out_objects = pureobjects_from_grid(
                param, pair_id, 'diff_out', diff_out, [height_diff, width_diff]
            )

            # 转换为加权对象信息
            diff_in_obj_infos = [
                WeightedObjInfo(pair_id, 'diff_in', obj, obj_params=None, grid_hw=[height_diff, width_diff])
                for obj in diff_in_objects
            ]

            diff_out_obj_infos = [
                WeightedObjInfo(pair_id, 'diff_out', obj, obj_params=None, grid_hw=[height_diff, width_diff])
                for obj in diff_out_objects
            ]

            if self.debug:
                self._debug_print(f"从差异输入网格提取了 {len(diff_in_obj_infos)} 个对象")
                self._debug_print(f"从差异输出网格提取了 {len(diff_out_obj_infos)} 个对象")
        else:
            diff_in_obj_infos = []
            diff_out_obj_infos = []
            if self.debug:
                self._debug_print("差异网格为空")

        # 存储提取的对象
        self.all_objects['input'].append((pair_id, input_obj_infos))
        self.all_objects['output'].append((pair_id, output_obj_infos))
        self.all_objects['diff_in'].append((pair_id, diff_in_obj_infos))
        self.all_objects['diff_out'].append((pair_id, diff_out_obj_infos))

        # 更新形状库
        self._update_shape_library(input_obj_infos + output_obj_infos)

        # 分析对象间的部分-整体关系
        self._analyze_part_whole_relationships(input_obj_infos)
        self._analyze_part_whole_relationships(output_obj_infos)

        # 应用权重计算 - 为每个对象设置权重
        self._calculate_object_weights(pair_id, input_grid, output_grid,
                                      input_obj_infos, output_obj_infos,
                                      diff_in_obj_infos, diff_out_obj_infos)

        # 分析diff映射关系
        mapping_rule = self._analyze_diff_mapping_with_weights(
            pair_id, input_grid, output_grid, diff_in, diff_out,
            input_obj_infos, output_obj_infos, diff_in_obj_infos, diff_out_obj_infos
        )

        self.mapping_rules.append(mapping_rule)

        if self.debug:
            self._debug_save_json(mapping_rule, f"mapping_rule_{pair_id}")
            self._debug_print(f"完成训练对 {pair_id} 的分析和权重计算")
            self._debug_print_object_weights(input_obj_infos, f"input_obj_weights_{pair_id}")
            self._debug_print_object_weights(output_obj_infos, f"output_obj_weights_{pair_id}")
            self._debug_print_object_weights(diff_in_obj_infos, f"diff_in_obj_weights_{pair_id}")
            self._debug_print_object_weights(diff_out_obj_infos, f"diff_out_obj_weights_{pair_id}")

    def _calculate_object_weights(self, pair_id, input_grid, output_grid,
                                 input_obj_infos, output_obj_infos,
                                 diff_in_obj_infos, diff_out_obj_infos):
        """
        为所有对象计算权重

        Args:
            pair_id: 训练对ID
            input_grid, output_grid: 输入输出网格
            input_obj_infos, output_obj_infos: 输入输出对象信息
            diff_in_obj_infos, diff_out_obj_infos: 差异对象信息
        """
        if self.debug:
            self._debug_print(f"计算训练对 {pair_id} 的对象权重")

        # 1. 初始权重 - 基于对象大小
        for obj_list in [input_obj_infos, output_obj_infos, diff_in_obj_infos, diff_out_obj_infos]:
            for obj_info in obj_list:
                # 基础权重为对象大小
                obj_info.obj_weight = 0

        # 2. 增加差异区域对象的权重
        for obj_info in diff_in_obj_infos + diff_out_obj_infos:
            obj_info.increase_weight(self.diff_weight_increment)

        # 3. 处理位于差异位置的原始对象
        # 创建坐标到对象的映射
        input_pos_to_obj = self._create_position_object_map(input_obj_infos)
        output_pos_to_obj = self._create_position_object_map(output_obj_infos)

        # 跟踪已增加权重的对象，避免重复增加
        increased_input_objs = set()
        increased_output_objs = set()

        # 如果有差异区域，为涉及差异的原始对象增加权重
        diff_in, diff_out = self.diff_pairs[-1]  # 最新添加的diff对

        if diff_in is not None and diff_out is not None:
            for i in range(len(diff_in)):
                for j in range(len(diff_in[0])):
                    if diff_in[i][j] is not None:  # 发现差异
                        # 检查该位置是否有输入对象
                        pos = (i, j)
                        if pos in input_pos_to_obj:
                            for obj_info in input_pos_to_obj[pos]:
                                # 确保每个对象只增加一次权重
                                if obj_info.obj_id not in increased_input_objs:
                                    obj_info.increase_weight(self.diff_weight_increment)
                                    increased_input_objs.add(obj_info.obj_id)
                                    if self.debug:
                                        self._debug_print(f"增加位置涉及差异的输入对象 {obj_info.obj_id} 权重，现在为 {obj_info.obj_weight}")

                        # 检查该位置是否有输出对象
                        if pos in output_pos_to_obj:
                            for obj_info in output_pos_to_obj[pos]:
                                # 确保每个对象只增加一次权重
                                if obj_info.obj_id not in increased_output_objs:
                                    obj_info.increase_weight(self.diff_weight_increment)
                                    increased_output_objs.add(obj_info.obj_id)
                                    if self.debug:
                                        self._debug_print(f"增加位置涉及差异的输出对象 {obj_info.obj_id} 权重，现在为 {obj_info.obj_weight}")



        # 4. 基于形状匹配增加权重
        self._add_shape_matching_weights(input_obj_infos, output_obj_infos)

        # 5. 考虑颜色占比，调整背景对象权重
        self._adjust_background_object_weights(input_grid, input_obj_infos)
        self._adjust_background_object_weights(output_grid, output_obj_infos)

    def _create_position_object_map(self, obj_infos):
        """
        创建坐标到对象的映射，用于快速查找特定位置的对象

        Args:
            obj_infos: 对象信息列表

        Returns:
            字典 {(row, col): [obj_info, ...]}
        """
        pos_to_obj = defaultdict(list)

        for obj_info in obj_infos:
            for _, (i, j) in obj_info.original_obj:
                pos_to_obj[(i, j)].append(obj_info)

        return pos_to_obj

    def _add_shape_matching_weights(self, input_obj_infos, output_obj_infos):
        """
        基于形状匹配增加对象权重

        Args:
            input_obj_infos: 输入对象信息列表
            output_obj_infos: 输出对象信息列表
        """
        # 创建形状匹配字典
        normalized_shapes = {}

        # 处理输入对象
        for obj_info in input_obj_infos:
            # 获取规范化形状
            normalized_obj = obj_info.obj_000

            # 使用可哈希的表示
            hashable_obj = self._get_hashable_representation(normalized_obj)

            if hashable_obj not in normalized_shapes:
                normalized_shapes[hashable_obj] = []
            normalized_shapes[hashable_obj].append(('input', obj_info))

        # 处理输出对象
        for obj_info in output_obj_infos:
            # 获取规范化形状
            normalized_obj = obj_info.obj_000

            # 使用可哈希的表示
            hashable_obj = self._get_hashable_representation(normalized_obj)

            if hashable_obj not in normalized_shapes:
                normalized_shapes[hashable_obj] = []
            normalized_shapes[hashable_obj].append(('output', obj_info))

        # 为相同形状的对象增加权重
        for shape, obj_list in normalized_shapes.items():
            if len(obj_list) <= 1:
                continue  # 跳过没有匹配的形状

            # 相同形状的对象数量作为额外权重
            shape_bonus = len(obj_list)

            for _, obj_info in obj_list:
                obj_info.increase_weight(shape_bonus)
                if self.debug:
                    self._debug_print(f"增加对象 {obj_info.obj_id} 的形状匹配权重 +{shape_bonus}，现在为 {obj_info.obj_weight}")

    def _adjust_background_object_weights(self, grid, obj_infos):
        """
        基于颜色占比调整背景对象权重

        Args:
            grid: 网格
            obj_infos: 对象信息列表
        """
        # 计算每种颜色的像素数
        color_counts = defaultdict(int)
        total_pixels = len(grid) * len(grid[0])

        for i in range(len(grid)):
            for j in range(len(grid[0])):
                color_counts[grid[i][j]] += 1

        # 找出背景颜色（占比超过阈值的颜色）
        background_colors = set()
        for color, count in color_counts.items():
            percentage = (count / total_pixels) * 100
            if percentage > self.pixel_threshold_pct:
                background_colors.add(color)
                if self.debug:
                    self._debug_print(f"识别到背景颜色: {color}, 占比: {percentage:.2f}%")

        # 调整背景对象的权重
        for obj_info in obj_infos:
            # 检查对象主色是否为背景色
            if obj_info.main_color in background_colors:
                # 计算对象中背景色的占比
                bg_pixels = sum(1 for val, _ in obj_info.original_obj if val in background_colors)
                bg_percentage = (bg_pixels / obj_info.size) * 100

                # 如果对象主要由背景色组成，降低其权重
                if bg_percentage > 80:  # 80%以上为背景色
                    # 将权重设为初始权重的一半
                    # new_weight = max(1, obj_info.obj_weight // 2)
                    new_weight = 0

                    obj_info.set_weight(new_weight)
                    if self.debug:
                        self._debug_print(f"降低背景对象 {obj_info.obj_id} 权重至 {new_weight}，背景色占比 {bg_percentage:.1f}%")
                        # display_matrices(obj_info.original_obj,obj_info.grid_hw )

    def _get_hashable_representation(self, obj_set):
        """
        将对象集合转换为可哈希的表示

        Args:
            obj_set: 对象集合

        Returns:
            可哈希的表示（元组）
        """
        sorted_elements = []
        for value, loc in obj_set:
            i, j = loc
            sorted_elements.append((value, i, j))

        return tuple(sorted(sorted_elements))

    def _analyze_diff_mapping_with_weights(self, pair_id, input_grid, output_grid, diff_in, diff_out,
                                          input_obj_infos, output_obj_infos, diff_in_obj_infos, diff_out_obj_infos):
        """
        基于对象权重分析差异网格映射关系

        Args:
            pair_id: 训练对ID
            input_grid, output_grid: 输入输出网格
            diff_in, diff_out: 差异网格
            input_obj_infos, output_obj_infos: 输入输出对象信息
            diff_in_obj_infos, diff_out_obj_infos: 差异对象信息

        Returns:
            映射规则字典
        """
        if self.debug:
            self._debug_print(f"基于权重分析差异映射关系，pair_id={pair_id}")

        mapping_rule = {
            "pair_id": pair_id,
            "object_mappings": [],
            "shape_transformations": [],
            "color_mappings": {},
            "position_changes": [],
            "part_whole_relationships": [],
            "weighted_objects": []  # 添加权重信息
        }

        # 准备分析diff对象间的映射
        if not diff_in_obj_infos or not diff_out_obj_infos:
            if self.debug:
                self._debug_print("差异对象为空，无法分析映射")
            return mapping_rule

        # 添加权重信息
        for obj_info in diff_in_obj_infos + diff_out_obj_infos:
            mapping_rule["weighted_objects"].append({
                "obj_id": obj_info.obj_id,
                "weight": obj_info.obj_weight,
                "size": obj_info.size,
                "type": obj_info.in_or_out
            })

        # 按权重降序排序对象，优先考虑高权重对象
        sorted_diff_in = sorted(diff_in_obj_infos, key=lambda x: x.obj_weight, reverse=True)
        sorted_diff_out = sorted(diff_out_obj_infos, key=lambda x: x.obj_weight, reverse=True)

        # 基于形状匹配寻找对象映射，但考虑权重
        object_mappings = self._find_object_mappings_by_shape_and_weight(sorted_diff_in, sorted_diff_out)

        if self.debug:
            self._debug_print(f"找到 {len(object_mappings)} 个基于形状和权重的对象匹配")

        # 分析每个映射的变换
        for in_obj, out_obj, match_info in object_mappings:
            # 分析颜色变换
            color_transformation = in_obj.get_color_transformation(out_obj)

            # 分析位置变换
            position_change = in_obj.get_positional_change(out_obj)

            # 添加到映射规则
            mapping_rule["object_mappings"].append({
                "diff_in_object": in_obj.to_dict(),
                "diff_out_object": out_obj.to_dict(),
                "match_info": match_info,
                "weight_product": in_obj.obj_weight * out_obj.obj_weight  # 添加权重乘积作为匹配强度
            })

            # 记录形状变换
            mapping_rule["shape_transformations"].append({
                "in_obj_id": in_obj.obj_id,
                "out_obj_id": out_obj.obj_id,
                "transform_type": match_info["transform_type"],
                "transform_name": match_info["transform_name"],
                "confidence": match_info["confidence"],
                "weight_in": in_obj.obj_weight,
                "weight_out": out_obj.obj_weight
            })

            # 记录颜色映射
            if color_transformation and color_transformation.get("color_mapping"):
                for from_color, to_color in color_transformation["color_mapping"].items():
                    if from_color not in mapping_rule["color_mappings"]:
                        mapping_rule["color_mappings"][from_color] = {
                            "to_color": to_color,
                            "weight": in_obj.obj_weight  # 使用输入对象权重作为颜色映射权重
                        }
                    elif in_obj.obj_weight > mapping_rule["color_mappings"][from_color]["weight"]:
                        # 如果当前对象权重更高，更新颜色映射
                        mapping_rule["color_mappings"][from_color] = {
                            "to_color": to_color,
                            "weight": in_obj.obj_weight
                        }

            # 记录位置变化
            mapping_rule["position_changes"].append({
                "in_obj_id": in_obj.obj_id,
                "out_obj_id": out_obj.obj_id,
                "delta_row": position_change["delta_row"],
                "delta_col": position_change["delta_col"],
                "direction": position_change.get("direction"),
                "orientation": position_change.get("orientation"),
                "weight_in": in_obj.obj_weight,
                "weight_out": out_obj.obj_weight
            })

        return mapping_rule

    def _find_object_mappings_by_shape_and_weight(self, diff_in_obj_infos, diff_out_obj_infos):
        """
        基于形状匹配和权重寻找对象映射

        Args:
            diff_in_obj_infos: 差异输入对象列表，已按权重降序排序
            diff_out_obj_infos: 差异输出对象列表，已按权重降序排序

        Returns:
            列表 [(in_obj, out_obj, match_info), ...]
        """
        mappings = []
        matched_out_objs = set()  # 跟踪已匹配的输出对象

        # 为每个输入对象找到最匹配的输出对象，优先考虑高权重对象
        for in_obj in diff_in_obj_infos:
            best_match = None
            best_match_info = None
            best_confidence = -1
            best_weight_score = -1

            for out_obj in diff_out_obj_infos:
                # 跳过已匹配的输出对象
                if out_obj.obj_id in matched_out_objs:
                    continue

                # 检查是否匹配任何变换形式
                matches, transform_type, transform_name = in_obj.matches_with_transformation(out_obj)

                if matches:
                    # 计算匹配置信度（基本置信度）
                    confidence = 0.7  # 初始置信度

                    # 对于完全相同形状，增加置信度
                    if transform_type == "same_shape":
                        confidence += 0.3

                    # 计算权重得分（输入权重 * 输出权重）
                    weight_score = in_obj.obj_weight * out_obj.obj_weight

                    # 如果权重得分更高，或权重得分相同但置信度更高，则更新最佳匹配
                    if weight_score > best_weight_score or (weight_score == best_weight_score and confidence > best_confidence):
                        best_weight_score = weight_score
                        best_confidence = confidence
                        best_match = out_obj
                        best_match_info = {
                            "transform_type": transform_type,
                            "transform_name": transform_name,
                            "confidence": confidence,
                            "weight_score": weight_score
                        }

            if best_match and best_confidence > 0:
                mappings.append((in_obj, best_match, best_match_info))
                matched_out_objs.add(best_match.obj_id)  # 标记该输出对象已匹配

        return mappings

    def analyze_common_patterns_with_weights(self):
        """
        分析多对训练数据的共有模式，考虑权重因素

        Returns:
            共有模式字典
        """
        if self.debug:
            self._debug_print("基于权重分析共有模式")

        if not self.mapping_rules:
            return {}

        # 分析共有的形状变换模式
        common_shape_transformations = self._find_common_shape_transformations_with_weights()

        # 分析共有的颜色映射模式
        common_color_mappings = self._find_common_color_mappings_with_weights()

        # 分析共有的位置变化模式
        common_position_changes = self._find_common_position_changes_with_weights()

        self.common_patterns = {
            "shape_transformations": common_shape_transformations,
            "color_mappings": common_color_mappings,
            "position_changes": common_position_changes
        }

        if self.debug:
            self._debug_save_json(self.common_patterns, "weighted_common_patterns")
            self._debug_print(f"找到 {len(common_shape_transformations)} 个加权共有形状变换模式")
            self._debug_print(f"找到 {len(common_color_mappings.get('mappings', {}))} 个加权共有颜色映射")
            self._debug_print(f"找到 {len(common_position_changes)} 个加权共有位置变化模式")

        return self.common_patterns

    def _find_common_shape_transformations_with_weights(self):
        """寻找共有的形状变换模式，考虑权重"""
        # 收集所有形状变换
        all_transformations = []
        for rule in self.mapping_rules:
            for transform in rule.get("shape_transformations", []):
                # 添加权重信息，如果有的话
                if "weight_in" in transform and "weight_out" in transform:
                    transform["weight_score"] = transform["weight_in"] * transform["weight_out"]
                else:
                    transform["weight_score"] = 1  # 默认权重
                all_transformations.append(transform)

        if not all_transformations:
            return []

        # 按变换类型分组
        transform_types = defaultdict(list)
        for transform in all_transformations:
            key = (transform["transform_type"], transform["transform_name"])
            transform_types[key].append(transform)

        common_transforms = []

        # 分析各种变换类型
        for (t_type, t_name), transforms in transform_types.items():
            # 如果变换至少出现两次，认为是共有模式
            if len(transforms) >= 2:
                # 计算加权平均置信度
                total_weight = sum(t.get("weight_score", 1) for t in transforms)
                avg_confidence = sum(t["confidence"] * t.get("weight_score", 1) for t in transforms) / total_weight

                common_transforms.append({
                    "transform_type": t_type,
                    "transform_name": t_name,
                    "count": len(transforms),
                    "confidence": avg_confidence,
                    "weight_score": total_weight,
                    "examples": [t["in_obj_id"] + "->" + t["out_obj_id"] for t in transforms]
                })

        # 按加权得分和出现次数排序
        return sorted(common_transforms, key=lambda x: (x["weight_score"], x["count"]), reverse=True)

    def _find_common_color_mappings_with_weights(self):
        """寻找共有的颜色映射模式，考虑权重"""
        # 收集所有颜色映射
        all_mappings = defaultdict(list)

        for rule in self.mapping_rules:
            for from_color, mapping in rule.get("color_mappings", {}).items():
                if isinstance(mapping, dict) and "to_color" in mapping:
                    # 新格式：包含权重
                    to_color = mapping["to_color"]
                    weight = mapping.get("weight", 1)
                    all_mappings[(from_color, to_color)].append(weight)
                else:
                    # 旧格式：直接是目标颜色
                    to_color = mapping
                    all_mappings[(from_color, to_color)].append(1)  # 默认权重为1

        # 找出共有的映射
        common_mappings = {}
        total_examples = len(self.mapping_rules)

        for (from_color, to_color), weights in all_mappings.items():
            if len(weights) > 1:  # 至少在两个示例中出现
                avg_weight = sum(weights) / len(weights)
                confidence = len(weights) / total_examples
                # 计算加权置信度
                weighted_confidence = confidence * avg_weight

                common_mappings[from_color] = {
                    "to_color": to_color,
                    "count": len(weights),
                    "confidence": confidence,
                    "avg_weight": avg_weight,
                    "weighted_confidence": weighted_confidence
                }

        # 分析颜色变化模式
        color_patterns = []

        # 检查是否有统一的颜色偏移
        offsets = []
        for from_color, mapping in common_mappings.items():
            to_color = mapping["to_color"]
            try:
                # 尝试计算颜色偏移
                offset = to_color - from_color
                offsets.append((offset, mapping["weighted_confidence"]))
            except (TypeError, ValueError):
                pass  # 跳过无法计算偏移的颜色对

        if offsets:
            # 对偏移值进行加权统计
            offset_counts = defaultdict(float)
            for offset, weight in offsets:
                offset_counts[offset] += weight

            # 找出权重最高的偏移
            if offset_counts:
                best_offset, best_score = max(offset_counts.items(), key=lambda x: x[1])
                color_patterns.append({
                    "type": "color_offset",
                    "offset": best_offset,
                    "weighted_score": best_score
                })

        return {
            "mappings": common_mappings,
            "patterns": color_patterns
        }

    def _find_common_position_changes_with_weights(self):
        """寻找共有的位置变化模式，考虑权重"""
        # 收集所有位置变化
        all_changes = []
        for rule in self.mapping_rules:
            for change in rule.get("position_changes", []):
                # 添加权重得分
                if "weight_in" in change and "weight_out" in change:
                    change["weight_score"] = change["weight_in"] * change["weight_out"]
                else:
                    change["weight_score"] = 1  # 默认权重
                all_changes.append(change)

        if not all_changes:
            return []

        # 按位移大小分组
        delta_groups = defaultdict(list)
        for change in all_changes:
            # 取整以处理浮点误差
            delta_row = round(change["delta_row"])
            delta_col = round(change["delta_col"])
            key = (delta_row, delta_col)
            delta_groups[key].append(change)

        # 按方向分组
        direction_groups = defaultdict(list)
        for change in all_changes:
            if "direction" in change and "orientation" in change:
                key = (change["direction"], change["orientation"])
                direction_groups[key].append(change)

        common_changes = []

        # 分析位移组，考虑权重
        for (dr, dc), changes in delta_groups.items():
            if len(changes) >= 2:  # 至少出现两次
                # 计算加权得分
                total_weight = sum(change.get("weight_score", 1) for change in changes)

                common_changes.append({
                    "type": "absolute_position",
                    "delta_row": dr,
                    "delta_col": dc,
                    "count": len(changes),
                    "confidence": len(changes) / len(all_changes),
                    "weight_score": total_weight
                })

        # 分析方向组，考虑权重
        for (direction, orientation), changes in direction_groups.items():
            if len(changes) >= 2:  # 至少出现两次
                # 计算加权得分
                total_weight = sum(change.get("weight_score", 1) for change in changes)

                common_changes.append({
                    "type": "directional",
                    "direction": direction,
                    "orientation": orientation,
                    "count": len(changes),
                    "confidence": len(changes) / len(all_changes),
                    "weight_score": total_weight
                })

        # 按加权得分和出现次数排序
        return sorted(common_changes, key=lambda x: (x["weight_score"], x["count"]), reverse=True)

    def analyze_common_patterns(self):
        """覆盖父类方法，使用加权版本"""
        return self.analyze_common_patterns_with_weights()

    def apply_common_patterns(self, input_grid, param):
        """
        将共有模式应用到新的输入网格，考虑权重

        Args:
            input_grid: 输入网格

        Returns:
            预测的输出网格
        """
        if self.debug:
            self._debug_print("开始应用加权共有模式到测试输入")
            self._debug_save_grid(input_grid, "test_input")

        # 分析共有模式，确保考虑权重
        if not self.common_patterns:
            self.analyze_common_patterns_with_weights()

        # 确保输入网格是元组格式
        if isinstance(input_grid, list):
            input_grid = tuple(tuple(row) for row in input_grid)

        # 获取网格尺寸
        height, width = len(input_grid), len(input_grid[0])

        # 提取输入网格中的对象
        input_objects = pureobjects_from_grid(
            param, -1, 'test_in', input_grid, [height, width]
        )

        # 转换为加权对象信息
        input_obj_infos = [
            WeightedObjInfo(-1, 'test_in', obj, obj_params=None, grid_hw=[height, width])
            for obj in input_objects
        ]

        # 计算测试输入对象的权重
        self._calculate_test_object_weights(input_grid, input_obj_infos)

        if self.debug:
            self._debug_print(f"从测试输入提取了 {len(input_obj_infos)} 个对象")
            self._debug_print_object_weights(input_obj_infos, "test_input_objects")

        # 创建输出网格（初始为输入的副本）
        output_grid = [list(row) for row in input_grid]

        # 应用颜色映射，优先考虑高权重映射
        if "color_mappings" in self.common_patterns:
            # 获取颜色映射并按加权置信度排序
            color_mappings = self.common_patterns["color_mappings"].get("mappings", {})
            sorted_mappings = sorted(
                [(from_color, mapping) for from_color, mapping in color_mappings.items()],
                key=lambda x: x[1].get("weighted_confidence", 0),
                reverse=True
            )

            for from_color, mapping in sorted_mappings:
                to_color = mapping["to_color"]
                cells_changed = 0

                # 根据对象权重应用颜色映射
                for obj_info in sorted(input_obj_infos, key=lambda x: x.obj_weight, reverse=True):
                    for val, (i, j) in obj_info.original_obj:
                        if val == from_color:
                            output_grid[i][j] = to_color
                            cells_changed += 1

                if self.debug and cells_changed > 0:
                    weighted_conf = mapping.get("weighted_confidence", 0)
                    self._debug_print(f"应用颜色映射: {from_color} -> {to_color}, 加权置信度: {weighted_conf:.2f}, 改变了 {cells_changed} 个单元格")

        # 应用颜色模式
        if "color_mappings" in self.common_patterns:
            for pattern in self.common_patterns["color_mappings"].get("patterns", []):
                if pattern["type"] == "color_offset" and pattern.get("weighted_score", 0) > 1:
                    offset = pattern["offset"]
                    cells_changed = 0

                    # 优先处理高权重对象
                    for obj_info in sorted(input_obj_infos, key=lambda x: x.obj_weight, reverse=True):
                        for val, (i, j) in obj_info.original_obj:
                            if val != 0:  # 不处理背景
                                output_grid[i][j] = (val + offset) % 10  # 假设颜色范围是0-9
                                cells_changed += 1

                    if self.debug and cells_changed > 0:
                        weighted_score = pattern.get("weighted_score", 0)
                        self._debug_print(f"应用颜色偏移: +{offset}, 加权得分: {weighted_score:.2f}, 改变了 {cells_changed} 个单元格")

        # 应用位置变化，优先考虑高权重变化
        position_changes = self.common_patterns.get("position_changes", [])
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
                for obj_info in sorted(input_obj_infos, key=lambda x: x.obj_weight, reverse=True):
                    obj_color = obj_info.main_color
                    # 对象中的每个像素
                    for val, (r, c) in obj_info.original_obj:
                        nr, nc = int(r + dr), int(c + dc)
                        if 0 <= nr < height and 0 <= nc < width:
                            temp_grid[nr][nc] = val  # 保留原始颜色
                            cells_moved += 1

                # 合并结果
                for i in range(height):
                    for j in range(width):
                        if temp_grid[i][j] != 0:  # 只覆盖非零值
                            output_grid[i][j] = temp_grid[i][j]

                if self.debug and cells_moved > 0:
                    weight_score = best_position_change.get("weight_score", 0)
                    self._debug_print(f"应用位置变化: ({dr}, {dc}), 加权得分: {weight_score:.2f}, 移动了 {cells_moved} 个单元格")

        if self.debug:
            self._debug_save_grid(output_grid, "test_output_predicted")
            self._debug_print("完成测试预测")

        return output_grid

    def _calculate_test_object_weights(self, input_grid, input_obj_infos):
        """
        计算测试输入对象的权重

        Args:
            input_grid: 输入网格
            input_obj_infos: 输入对象信息列表
        """
        # 1. 初始权重 - 基于对象大小
        for obj_info in input_obj_infos:
            obj_info.obj_weight = obj_info.size

        # 2. 基于形状库匹配增加权重
        for obj_info in input_obj_infos:
            normalized_obj = obj_info.obj_000
            hashable_obj = self._get_hashable_representation(normalized_obj)

            # 检查是否在形状库中
            for shape_key, shape_info in self.shape_library.items():
                lib_shape = shape_info["normalized_shape"]
                lib_hashable = self._get_hashable_representation(lib_shape)

                if hashable_obj == lib_hashable:
                    # 如果形状匹配，增加权重
                    match_bonus = shape_info["count"] * 2  # 根据出现次数增加权重
                    obj_info.increase_weight(match_bonus)
                    if self.debug:
                        self._debug_print(f"对象 {obj_info.obj_id} 匹配形状库中的形状，增加权重 +{match_bonus}")
                    break

        # 3. 考虑颜色占比，调整背景对象权重
        self._adjust_background_object_weights(input_grid, input_obj_infos)

    def _debug_print_object_weights(self, obj_infos, name):
        """
        输出对象权重信息到调试文件

        Args:
            obj_infos: 对象信息列表
            name: 输出文件名
        """
        if not self.debug:
            return

        weight_info = []
        for obj_info in sorted(obj_infos, key=lambda x: x.obj_weight, reverse=True):
            weight_info.append({
                "obj_id": obj_info.obj_id,
                "weight": obj_info.obj_weight,
                "size": obj_info.size,
                "main_color": obj_info.main_color,
                "height": obj_info.height,
                "width": obj_info.width
            })

        self._debug_save_json(weight_info, name)

        # 打印权重信息
        self._debug_print(f"对象权重信息 ({name}):")
        for info in weight_info[:5]:  # 只打印前5个
            self._debug_print(f"  对象 {info['obj_id']}: 权重={info['weight']}, 大小={info['size']}, 颜色={info['main_color']}")
        if len(weight_info) > 5:
            self._debug_print(f"  ... 还有 {len(weight_info) - 5} 个对象")