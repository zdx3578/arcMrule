"""
加权ARC差异分析器核心类

实现加权版的ARC差异分析器，扩展基础分析器功能。
"""

import numpy as np
from typing import List, Dict, Tuple, Any, Set, FrozenSet, Optional, Union
from collections import defaultdict

from arcMrule.diffstar.arc_diff_analyzer import ARCDiffAnalyzer
from objutil import pureobjects_from_grid
from weightgird import grid2grid_fromgriddiff

from .weighted_obj_info import WeightedObjInfo
from .weight_calculator import WeightCalculator
from .object_matching import ObjectMatcher
from .pattern_analyzer import PatternAnalyzer
from .rule_applier import RuleApplier
from .utils import get_hashable_representation


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
        self.transformation_rules = []

        # 新增: 形状-颜色关系规则
        self.shape_color_rules = []

        # 新增: 属性依赖规则
        self.attribute_dependency_rules = []

        # 重写对象存储结构，使用WeightedObjInfo替代ObjInfo
        self.all_objects = {
            'input': [],  # [(pair_id, [WeightedObjInfo]), ...]
            'output': [], # [(pair_id, [WeightedObjInfo]), ...]
            'diff_in': [], # [(pair_id, [WeightedObjInfo]), ...]
            'diff_out': []  # [(pair_id, [WeightedObjInfo]), ...]
        }

        # 初始化辅助组件
        self.weight_calculator = WeightCalculator(
            self.pixel_threshold_pct,
            self.weight_increment,
            self.diff_weight_increment,
            self._debug_print if debug else None
        )

        self.object_matcher = ObjectMatcher(self._debug_print if debug else None)
        self.pattern_analyzer = PatternAnalyzer(self._debug_print if debug else None)
        self.rule_applier = RuleApplier(self._debug_print if debug else None)

    def set_background_colors(self, background_colors):
        """
        设置全局背景色

        Args:
            background_colors: 背景色集合
        """
        self.background_colors = background_colors
        # 如果使用了权重计算器，将背景色传递给它
        if hasattr(self, 'weight_calculator'):
            self.weight_calculator.set_background_colors(background_colors)

    def add_train_pair(self, pair_id, input_grid, output_grid, param):
        """
        添加一对训练数据，提取对象并计算权重

        Args:
            pair_id: 训练对ID
            input_grid: 输入网格
            output_grid: 输出网格
            param: 对象提取参数
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
        self.weight_calculator.calculate_object_weights(
            pair_id, input_grid, output_grid,
            input_obj_infos, output_obj_infos,
            diff_in_obj_infos, diff_out_obj_infos,
            diff_in, diff_out
        )

        # 新增: 提取基于形状的颜色变化规则
        shape_color_rule = self._extract_shape_color_rules(
            pair_id, input_obj_infos, output_obj_infos,
            diff_in_obj_infos, diff_out_obj_infos
        )
        if shape_color_rule:
            self.shape_color_rules.append(shape_color_rule)

        # 新增: 提取更通用的属性依赖规则
        attr_rules = self._extract_attribute_dependency_rules(
            pair_id, input_obj_infos, output_obj_infos
        )
        self.attribute_dependency_rules.extend(attr_rules)

        # 分析diff映射关系
        mapping_rule = self.object_matcher.analyze_diff_mapping_with_weights(
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

            if shape_color_rule:
                self._debug_save_json(shape_color_rule, f"shape_color_rule_{pair_id}")
                self._debug_print(f"提取了 {len(shape_color_rule.get('rules', []))} 个形状-颜色规则")

            if attr_rules:
                self._debug_save_json(attr_rules, f"attr_dependency_rules_{pair_id}")
                self._debug_print(f"提取了 {len(attr_rules)} 个属性依赖规则")

    def _extract_shape_color_rules(self, pair_id, input_obj_infos, output_obj_infos,
                                diff_in_obj_infos, diff_out_obj_infos):
        """
        提取基于形状的颜色变化规则

        Args:
            pair_id: 训练对ID
            input_obj_infos: 输入对象信息列表
            output_obj_infos: 输出对象信息列表
            diff_in_obj_infos: 差异输入对象信息列表
            diff_out_obj_infos: 差异输出对象信息列表

        Returns:
            形状-颜色规则字典
        """
        rules = []

        # 首先匹配输入和输出对象
        matched_objects = self._match_input_output_objects(input_obj_infos, output_obj_infos)

        # 分析每个匹配对，寻找形状与颜色变化的关系
        for in_obj, out_obj in matched_objects:
            # 只关注颜色发生变化的对象
            if in_obj.main_color != out_obj.main_color:
                # 检查是否有形状特征可以解释颜色变化
                shape_features = self._extract_shape_features(in_obj)

                # 形成规则: 基于形状特征的颜色变化
                rule = {
                    "rule_type": "shape_to_color",
                    "pair_id": pair_id,
                    "object_id": in_obj.obj_id,
                    "shape_features": shape_features,
                    "original_color": in_obj.main_color,
                    "new_color": out_obj.main_color,
                    "weight": in_obj.obj_weight,  # 使用对象权重表示规则重要性
                    "confidence": 0.7  # 初始置信度
                }

                # 增强规则: 检查其他对象是否有相同形状特征且进行了相同颜色变化
                similar_changes = 0
                for other_in, other_out in matched_objects:
                    if (other_in != in_obj and
                        other_in.main_color == in_obj.main_color and
                        other_out.main_color == out_obj.main_color and
                        self._shape_similarity(other_in, in_obj) > 0.7):
                        similar_changes += 1
                        rule["confidence"] = min(1.0, rule["confidence"] + 0.1)

                if similar_changes > 0:
                    rule["similar_changes"] = similar_changes

                rules.append(rule)

        # 分析差异区域对象的颜色变化
        if diff_in_obj_infos and diff_out_obj_infos:
            diff_matched_objects = self._match_input_output_objects(diff_in_obj_infos, diff_out_obj_infos)

            for in_obj, out_obj in diff_matched_objects:
                if in_obj.main_color != out_obj.main_color:
                    shape_features = self._extract_shape_features(in_obj)

                    rule = {
                        "rule_type": "diff_shape_to_color",
                        "pair_id": pair_id,
                        "object_id": in_obj.obj_id,
                        "shape_features": shape_features,
                        "original_color": in_obj.main_color,
                        "new_color": out_obj.main_color,
                        "weight": in_obj.obj_weight * 1.5,  # 差异区域权重更高
                        "confidence": 0.8  # 差异区域置信度更高
                    }
                    rules.append(rule)

        # 寻找跨对象的形状-颜色关联
        # 例如: 对象A的形状决定对象B的颜色
        self._extract_cross_object_shape_color_rules(
            pair_id, input_obj_infos, output_obj_infos, rules
        )

        if rules:
            return {
                "pair_id": pair_id,
                "rules": rules
            }
        return None

    def _extract_attribute_dependency_rules(self, pair_id, input_obj_infos, output_obj_infos):
        """
        提取更通用的属性依赖规则

        Args:
            pair_id: 训练对ID
            input_obj_infos: 输入对象信息列表
            output_obj_infos: 输出对象信息列表

        Returns:
            属性依赖规则列表
        """
        rules = []

        # 匹配对象
        matched_objects = self._match_input_output_objects(input_obj_infos, output_obj_infos)

        # 属性变化分析
        for in_obj, out_obj in matched_objects:
            # 分析各种属性变化
            changes = {
                "color": in_obj.main_color != out_obj.main_color,
                "position": (in_obj.top != out_obj.top or in_obj.left != out_obj.left),
                "size": in_obj.size != out_obj.size,
                "shape": self._shape_similarity(in_obj, out_obj) < 0.9
            }

            # 如果有属性发生变化
            if any(changes.values()):
                # 尝试找出变化的依据
                for attr_name, changed in changes.items():
                    if changed:
                        # 尝试根据自身其他属性解释变化
                        self_rule = self._find_attribute_dependency(
                            in_obj, out_obj, attr_name, input_obj_infos, output_obj_infos
                        )

                        if self_rule:
                            self_rule["pair_id"] = pair_id
                            rules.append(self_rule)

                        # 尝试根据其他对象属性解释变化
                        for other_in in input_obj_infos:
                            if other_in != in_obj:
                                cross_rule = self._find_cross_object_dependency(
                                    in_obj, out_obj, other_in, attr_name
                                )

                                if cross_rule:
                                    cross_rule["pair_id"] = pair_id
                                    rules.append(cross_rule)

        return rules

    def _extract_cross_object_shape_color_rules(self, pair_id, input_obj_infos, output_obj_infos, rules_list):
        """提取跨对象的形状-颜色规则"""
        # 对于每个输出对象
        for out_obj in output_obj_infos:
            # 检查其颜色是否可能受到其他输入对象形状的影响
            for in_obj in input_obj_infos:
                # 跳过可能是同一对象变化前后的情况
                if self._shape_similarity(in_obj, out_obj) > 0.8:
                    continue

                shape_features = self._extract_shape_features(in_obj)

                # 尝试寻找规律: 输入对象in_obj的形状影响输出对象out_obj的颜色
                rule = {
                    "rule_type": "cross_shape_to_color",
                    "pair_id": pair_id,
                    "in_object_id": in_obj.obj_id,
                    "out_object_id": out_obj.obj_id,
                    "shape_features": shape_features,
                    "determining_color": in_obj.main_color,
                    "resulting_color": out_obj.main_color,
                    "weight": (in_obj.obj_weight + out_obj.obj_weight) / 2,
                    "confidence": 0.6
                }

                # 如果输入对象权重高，提高规则置信度
                if in_obj.obj_weight > 3:
                    rule["confidence"] = min(1.0, rule["confidence"] + 0.2)

                rules_list.append(rule)

    def _match_input_output_objects(self, input_objects, output_objects):
        """匹配输入和输出对象，返回(输入对象,输出对象)对列表"""
        matches = []

        # 简单启发式匹配: 优先考虑位置和形状相似性
        for in_obj in input_objects:
            best_match = None
            best_score = -1

            for out_obj in output_objects:
                # 计算相似度分数
                shape_sim = self._shape_similarity(in_obj, out_obj)
                pos_sim = self._position_similarity(in_obj, out_obj)

                # 综合分数
                score = 0.6 * shape_sim + 0.4 * pos_sim

                if score > best_score:
                    best_score = score
                    best_match = out_obj

            # 只有当相似度足够高时才认为匹配有效
            if best_score > 0.5 and best_match:
                matches.append((in_obj, best_match))

        return matches

    def _extract_shape_features(self, obj_info):
        """提取对象的形状特征"""
        return {
            "height": obj_info.height,
            "width": obj_info.width,
            "size": obj_info.size,
            "aspect_ratio": obj_info.width / max(1, obj_info.height),
            "is_symmetric_h": self._check_horizontal_symmetry(obj_info),
            "is_symmetric_v": self._check_vertical_symmetry(obj_info),
            "compactness": obj_info.size / (obj_info.height * obj_info.width),
            "num_corners": self._estimate_corners(obj_info)
        }

    def _check_horizontal_symmetry(self, obj_info):
        """检查水平对称性"""
        # 简化实现，根据实际情况可以完善
        return True  # 占位实现

    def _check_vertical_symmetry(self, obj_info):
        """检查垂直对称性"""
        # 简化实现，根据实际情况可以完善
        return True  # 占位实现

    def _estimate_corners(self, obj_info):
        """估计对象的角点数量"""
        # 简化实现，根据实际情况可以完善
        return 4  # 占位实现

    def _shape_similarity(self, obj1, obj2):
        """计算两个对象的形状相似度"""
        # 简单实现，可以根据需要增强
        size_sim = min(obj1.size, obj2.size) / max(obj1.size, obj2.size)
        aspect_sim = min(obj1.width/max(1,obj1.height), obj2.width/max(1,obj2.height)) / \
                    max(obj1.width/max(1,obj1.height), obj2.width/max(1,obj2.height))

        return 0.7 * size_sim + 0.3 * aspect_sim

    def _position_similarity(self, obj1, obj2):
        """计算两个对象的位置相似度，简化为只比较左上角位置"""
        try:
            # 尝试从对象或对象的obj属性获取位置信息
            x1 = getattr(obj1, 'left', None) or getattr(obj1.obj, 'left', 0)
            y1 = getattr(obj1, 'top', None) or getattr(obj1.obj, 'top', 0)

            x2 = getattr(obj2, 'left', None) or getattr(obj2.obj, 'left', 0)
            y2 = getattr(obj2, 'top', None) or getattr(obj2.obj, 'top', 0)

            # 计算左上角位置的距离
            dist = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

            # 获取网格尺寸用于归一化距离
            grid_hw1 = getattr(obj1, 'grid_hw', [10, 10])
            grid_hw2 = getattr(obj2, 'grid_hw', [10, 10])

            # 使用最大的网格尺寸进行归一化
            max_grid_size = max(max(grid_hw1), max(grid_hw2))

            # 将距离归一化为相似度
            return max(0, 1 - dist / max_grid_size)
        except (AttributeError, TypeError) as e:
            # 如果无法获取位置信息，则返回中性相似度值
            if self.debug:
                self._debug_print(f"位置相似度计算错误: {e}")
            return 0.5  # 返回中性值，既不完全相似也不完全不相似

    def _find_attribute_dependency(self, in_obj, out_obj, changed_attr, all_in_objs, all_out_objs):
        """寻找对象属性变化的依据"""
        # 这里实现寻找属性依赖的逻辑
        # 例如：如果颜色变化，尝试找出依赖于形状的规则

        if changed_attr == "color":
            # 检查是否基于形状的颜色变化规则
            shape_features = self._extract_shape_features(in_obj)

            return {
                "rule_type": "self_attr_dependency",
                "changed_attr": "color",
                "from_value": in_obj.main_color,
                "to_value": out_obj.main_color,
                "dependent_on": "shape",
                "shape_features": shape_features,
                "object_id": in_obj.obj_id,
                "confidence": 0.7
            }

        # 可以添加其他属性变化的依赖分析
        return None

    def _find_cross_object_dependency(self, in_obj, out_obj, other_in_obj, changed_attr):
        """寻找跨对象的属性依赖"""
        if changed_attr == "color":
            # 检查颜色变化是否依赖于其他对象的形状
            cross_rule = {
                "rule_type": "cross_obj_attr_dependency",
                "target_object_id": in_obj.obj_id,
                "reference_object_id": other_in_obj.obj_id,
                "changed_attr": "color",
                "from_value": in_obj.main_color,
                "to_value": out_obj.main_color,
                "dependent_on": "shape",
                "reference_shape": self._extract_shape_features(other_in_obj),
                "confidence": 0.65
            }
            return cross_rule

        return None

    def analyze_common_patterns_with_weights(self):
        """
        分析多对训练数据的共有模式，考虑权重因素

        Returns:
            共有模式字典
        """
        if not self.mapping_rules:
            return {}

        # 调用PatternAnalyzer的方法分析基本模式
        self.common_patterns = self.pattern_analyzer.analyze_common_patterns(self.mapping_rules)

        # 新增: 归纳形状-颜色规则
        shape_color_patterns = self._induce_shape_color_patterns()
        if shape_color_patterns:
            self.common_patterns['shape_color_rules'] = shape_color_patterns

        # 新增: 归纳属性依赖规则
        attr_dependency_patterns = self._induce_attribute_dependency_patterns()
        if attr_dependency_patterns:
            self.common_patterns['attribute_dependencies'] = attr_dependency_patterns

        if self.debug:
            self._debug_save_json(self.common_patterns, "weighted_common_patterns")
            self._debug_print(f"找到 {len(self.common_patterns.get('shape_transformations', []))} 个加权共有形状变换模式")
            self._debug_print(f"找到 {len(self.common_patterns.get('color_mappings', {}).get('mappings', {}))} 个加权共有颜色映射")
            self._debug_print(f"找到 {len(self.common_patterns.get('position_changes', []))} 个加权共有位置变化模式")

            # 输出新增的模式
            if 'shape_color_rules' in self.common_patterns:
                self._debug_print(f"找到 {len(self.common_patterns['shape_color_rules'])} 个形状-颜色规则模式")
            if 'attribute_dependencies' in self.common_patterns:
                self._debug_print(f"找到 {len(self.common_patterns['attribute_dependencies'])} 个属性依赖规则模式")

        return self.common_patterns

    def _induce_shape_color_patterns(self):
        """归纳形状-颜色规则模式"""
        if not self.shape_color_rules:
            return []

        patterns = []
        rule_groups = self._group_similar_shape_color_rules()

        for group in rule_groups:
            if len(group) >= 2:  # 至少需要2个相似规则才能形成模式
                pattern = self._create_shape_color_pattern(group)
                if pattern:
                    patterns.append(pattern)

        return patterns

    def _group_similar_shape_color_rules(self):
        """将相似的形状-颜色规则分组"""
        if not self.shape_color_rules:
            return []

        # 将所有规则平铺
        all_rules = []
        for rule_set in self.shape_color_rules:
            all_rules.extend(rule_set.get('rules', []))

        # 按规则类型分组
        rule_type_groups = defaultdict(list)
        for rule in all_rules:
            rule_type_groups[rule.get('rule_type', '')].append(rule)

        # 对每种规则类型内的规则进行相似性分组
        all_groups = []

        for rule_type, rules in rule_type_groups.items():
            if rule_type == 'shape_to_color':
                # 按颜色变化分组
                color_change_groups = defaultdict(list)
                for rule in rules:
                    key = (rule.get('original_color'), rule.get('new_color'))
                    color_change_groups[key].append(rule)

                # 将各组添加到结果中
                for group in color_change_groups.values():
                    if group:
                        all_groups.append(group)

            elif rule_type == 'cross_shape_to_color':
                # 跨对象规则按结果颜色分组
                result_color_groups = defaultdict(list)
                for rule in rules:
                    key = rule.get('resulting_color')
                    result_color_groups[key].append(rule)

                for group in result_color_groups.values():
                    if group:
                        all_groups.append(group)

        return all_groups

    def _create_shape_color_pattern(self, rules_group):
        """从规则组创建形状-颜色模式"""
        if not rules_group:
            return None

        rule_type = rules_group[0].get('rule_type')

        if rule_type == 'shape_to_color':
            # 提取共有颜色变化
            color_change = (rules_group[0].get('original_color'), rules_group[0].get('new_color'))

            # 提取共有形状特征
            common_shape_features = {}
            for feature_name in rules_group[0].get('shape_features', {}):
                # 检查该特征是否在所有规则中都相似
                values = [rule['shape_features'].get(feature_name) for rule in rules_group
                        if feature_name in rule.get('shape_features', {})]

                if values and all(abs(v - values[0]) < 0.2 for v in values):
                    common_shape_features[feature_name] = sum(values) / len(values)

            # 创建模式
            pattern = {
                "pattern_type": "shape_to_color",
                "color_change": {"from": color_change[0], "to": color_change[1]},
                "shape_conditions": common_shape_features,
                "supporting_rules": len(rules_group),
                "confidence": sum(rule.get('confidence', 0) for rule in rules_group) / len(rules_group),
                "weight": sum(rule.get('weight', 1) for rule in rules_group) / len(rules_group)
            }

            return pattern

        elif rule_type == 'cross_shape_to_color':
            # 处理跨对象规则
            resulting_color = rules_group[0].get('resulting_color')

            pattern = {
                "pattern_type": "cross_shape_to_color",
                "resulting_color": resulting_color,
                "confidence": sum(rule.get('confidence', 0) for rule in rules_group) / len(rules_group),
                "supporting_rules": len(rules_group),
                "weight": sum(rule.get('weight', 1) for rule in rules_group) / len(rules_group)
            }

            return pattern

        return None

    def _induce_attribute_dependency_patterns(self):
        """归纳属性依赖规则模式"""
        if not self.attribute_dependency_rules:
            return []

        # 按规则类型和变化属性分组
        rule_groups = defaultdict(list)
        for rule in self.attribute_dependency_rules:
            key = (rule.get('rule_type', ''), rule.get('changed_attr', ''))
            rule_groups[key].append(rule)

        patterns = []

        # 处理每种规则类型
        for (rule_type, changed_attr), rules in rule_groups.items():
            if len(rules) < 2:
                continue

            if rule_type == 'self_attr_dependency' and changed_attr == 'color':
                # 处理基于自身属性的颜色变化规则
                color_change_groups = defaultdict(list)
                for rule in rules:
                    key = (rule.get('from_value'), rule.get('to_value'))
                    color_change_groups[key].append(rule)

                # 为每组颜色变化创建一个模式
                for color_change, group_rules in color_change_groups.items():
                    if len(group_rules) >= 2:
                        pattern = {
                            "pattern_type": "self_attr_color_change",
                            "color_change": {"from": color_change[0], "to": color_change[1]},
                            "dependent_on": group_rules[0].get('dependent_on'),
                            "supporting_rules": len(group_rules),
                            "confidence": sum(rule.get('confidence', 0) for rule in group_rules) / len(group_rules)
                        }
                        patterns.append(pattern)

            elif rule_type == 'cross_obj_attr_dependency' and changed_attr == 'color':
                # 处理跨对象的颜色变化规则
                cross_pattern = {
                    "pattern_type": "cross_obj_color_dependency",
                    "changed_attr": changed_attr,
                    "dependent_on": rules[0].get('dependent_on'),
                    "supporting_rules": len(rules),
                    "confidence": sum(rule.get('confidence', 0) for rule in rules) / len(rules)
                }
                patterns.append(cross_pattern)

        return patterns

    def analyze_common_patterns(self):
        """覆盖父类方法，使用加权版本"""
        return self.analyze_common_patterns_with_weights()

    def apply_common_patterns(self, input_grid, param):
        """
        将共有模式应用到新的输入网格，考虑权重

        Args:
            input_grid: 输入网格
            param: 对象提取参数

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
        self.weight_calculator.calculate_test_object_weights(input_grid, input_obj_infos, self.shape_library)

        if self.debug:
            self._debug_print(f"从测试输入提取了 {len(input_obj_infos)} 个对象")
            self._debug_print_object_weights(input_obj_infos, "test_input_objects")

        # 由RuleApplier应用规则，生成输出网格
        output_grid = self.rule_applier.apply_patterns(
            input_grid, self.common_patterns, input_obj_infos, self.debug
        )

        if self.debug:
            self._debug_save_grid(output_grid, "test_output_predicted")
            self._debug_print("完成测试预测")

        return output_grid

    def apply_transformation_rules(self, input_grid, common_patterns=None, transformation_rules=None):
        """
        应用提取的转换规则，将输入网格转换为预测的输出网格（委托给 rule_applier）

        Args:
            input_grid: 输入网格
            common_patterns: 识别的共有模式，如果不提供则使用当前的共有模式
            transformation_rules: 可选，特定的转换规则列表，如果不提供则使用当前累积的规则

        Returns:
            预测的输出网格
        """
        if self.debug:
            self._debug_print("调用转换规则应用功能")

        # 使用当前的共有模式（如果未提供）
        if common_patterns is None:
            if not self.common_patterns:
                self.analyze_common_patterns_with_weights()
            common_patterns = self.common_patterns

        # 使用当前的转换规则（如果未提供）
        if transformation_rules is None:
            transformation_rules = self.transformation_rules

        # 获取网格尺寸
        if isinstance(input_grid, list):
            input_grid = tuple(tuple(row) for row in input_grid)

        height, width = len(input_grid), len(input_grid[0])

        # 提取输入网格中的对象
        input_objects = []
        for param in [(True, True, False), (True, False, False), (False, False, False), (False, True, False)]:
            objects = pureobjects_from_grid(param, -1, 'test_in', input_grid, [height, width])
            for obj in objects:
                input_objects.append(WeightedObjInfo(-1, 'test_in', obj, obj_params=None, grid_hw=[height, width]))

        # 计算测试输入对象的权重
        self.weight_calculator.calculate_test_object_weights(input_grid, input_objects, self.shape_library)

        # 委托给 rule_applier 处理转换规则的应用
        return self.rule_applier.apply_transformation_rules(
            input_grid, common_patterns, input_objects, transformation_rules, self.debug
        )

    def get_prediction_confidence(self, predicted_output, actual_output):
        """
        计算预测与实际输出的匹配程度，返回置信度得分

        Args:
            predicted_output: 预测的输出网格
            actual_output: 实际的输出网格

        Returns:
            匹配置信度 (0-1)
        """
        if predicted_output == actual_output:
            return 1.0  # 完全匹配

        # 计算网格大小
        if not predicted_output or not actual_output:
            return 0.0

        height_pred, width_pred = len(predicted_output), len(predicted_output[0])
        height_act, width_act = len(actual_output), len(actual_output[0])

        # 如果尺寸不同，返回较低的置信度
        if height_pred != height_act or width_pred != width_act:
            return 0.1  # 尺寸不匹配，几乎没有信心

        # 计算像素匹配率
        total_pixels = height_pred * width_pred
        matching_pixels = 0

        for i in range(height_pred):
            for j in range(width_pred):
                if predicted_output[i][j] == actual_output[i][j]:
                    matching_pixels += 1

        # 基本置信度：匹配像素比例
        base_confidence = matching_pixels / total_pixels

        # 优化：考虑重要区域的匹配程度
        # 例如：非背景像素（非0像素）的匹配更重要
        non_zero_pred = sum(1 for row in predicted_output for pixel in row if pixel != 0)
        non_zero_act = sum(1 for row in actual_output for pixel in row if pixel != 0)

        # 计算非零像素的匹配
        non_zero_matching = 0
        for i in range(height_pred):
            for j in range(width_pred):
                if predicted_output[i][j] != 0 and predicted_output[i][j] == actual_output[i][j]:
                    non_zero_matching += 1

        # 非零像素匹配率（避免除零）
        if max(non_zero_pred, non_zero_act) > 0:
            non_zero_confidence = non_zero_matching / max(non_zero_pred, non_zero_act)
        else:
            non_zero_confidence = 1.0  # 如果两者都没有非零像素，则认为匹配

        # 加权组合两种置信度，非零区域匹配更重要
        combined_confidence = 0.3 * base_confidence + 0.7 * non_zero_confidence

        if self.debug:
            self._debug_print(f"预测置信度: 基本={base_confidence:.4f}, 非零区域={non_zero_confidence:.4f}, 组合={combined_confidence:.4f}")

        return combined_confidence

    def calculate_rule_confidence(self, input_grid, predicted_output):
        """
        计算基于规则生成的预测输出的置信度

        Args:
            input_grid: 输入网格
            predicted_output: 预测的输出网格

        Returns:
            规则预测置信度 (0-1)
        """
        # 如果没有规则，置信度低
        if not self.transformation_rules:
            return 0.2

        # 获取应用规则的数量
        num_rules_applied = 0
        total_rule_confidence = 0.0

        # 计算各种规则的应用情况
        for rule in self.transformation_rules:
            # 检查规则是否适用于当前输入/输出
            if self._is_rule_applicable(rule, input_grid, predicted_output):
                num_rules_applied += 1

                # 计算规则的置信度
                rule_conf = 0.0

                # 1. 如果规则在训练数据中频繁出现，提高置信度
                if 'pair_id' in rule:
                    rule_conf += 0.3  # 基础置信度

                # 2. 考虑对象权重
                if 'weighted_objects' in rule and rule['weighted_objects']:
                    avg_weight = sum(obj['weight'] for obj in rule['weighted_objects']) / len(rule['weighted_objects'])
                    weight_factor = min(1.0, avg_weight / 5.0)  # 规范化到0-1范围
                    rule_conf += weight_factor * 0.3

                # 3. 考虑模式匹配
                if 'transformation_patterns' in rule and rule['transformation_patterns']:
                    patterns = rule['transformation_patterns']
                    for pattern in patterns:
                        if pattern.get('confidence', 0) > 0.7:
                            rule_conf += 0.2
                            break

                # 累加总置信度
                total_rule_confidence += rule_conf

        # 如果没有应用规则，返回低置信度
        if num_rules_applied == 0:
            return 0.3

        # 计算平均规则置信度，并确保不超过1.0
        avg_rule_confidence = min(1.0, total_rule_confidence / num_rules_applied)

        if self.debug:
            self._debug_print(f"规则预测置信度: {avg_rule_confidence:.4f} (应用了 {num_rules_applied} 条规则)")

        return avg_rule_confidence

    def _is_rule_applicable(self, rule, input_grid, predicted_output):
        """检查规则是否适用于给定的输入/输出对"""
        # 简化版规则适用性检查
        # 在实际应用中，可以根据规则的具体内容进行更复杂的检查
        # 例如检查对象匹配、位置变化、颜色变换等是否符合规则

        # 检查输入网格中是否存在与规则相关的特征
        if 'object_mappings' in rule and rule['object_mappings']:
            # 抽取输入网格中的对象
            height, width = len(input_grid), len(input_grid[0])
            input_objects = []
            for param in [(True, True, False), (False, False, False)]:
                objects = pureobjects_from_grid(param, -1, 'test_in', input_grid, [height, width])
                for obj in objects:
                    input_objects.append(WeightedObjInfo(-1, 'test_in', obj, obj_params=None, grid_hw=[height, width]))

            # 检查是否有对象匹配规则中的对象
            for mapping in rule['object_mappings']:
                if 'diff_in_object' in mapping:
                    in_obj_info = mapping['diff_in_object']
                    # 简化检查：只检查是否有类似大小和颜色的对象
                    for obj in input_objects:
                        if (hasattr(obj, 'size') and hasattr(obj, 'main_color') and
                            abs(obj.size - in_obj_info.get('size', 0)) < 3 and
                            obj.main_color == in_obj_info.get('main_color')):
                            return True

        # 默认返回True，表示规则适用
        return True  # 简化版，实际应用中需要更复杂的匹配逻辑

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