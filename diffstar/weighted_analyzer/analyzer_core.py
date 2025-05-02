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

    def analyze_common_patterns_with_weights(self):
        """
        分析多对训练数据的共有模式，考虑权重因素

        Returns:
            共有模式字典
        """
        if not self.mapping_rules:
            return {}

        self.common_patterns = self.pattern_analyzer.analyze_common_patterns(self.mapping_rules)

        if self.debug:
            self._debug_save_json(self.common_patterns, "weighted_common_patterns")
            self._debug_print(f"找到 {len(self.common_patterns.get('shape_transformations', []))} 个加权共有形状变换模式")
            self._debug_print(f"找到 {len(self.common_patterns.get('color_mappings', {}).get('mappings', {}))} 个加权共有颜色映射")
            self._debug_print(f"找到 {len(self.common_patterns.get('position_changes', []))} 个加权共有位置变化模式")

        return self.common_patterns

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