import json
import numpy as np
from typing import List, Dict, Any, Tuple
import os

from core.grid import Grid
from core.object_extractor import ObjectExtractor
from core.diff_analyzer import DiffAnalyzer
from patterns.rule_engine import RuleEngine
from patterns.pattern_recognizer import PatternRecognizer
from prioritization.object_weights import ObjectWeightCalculator
from knowledge.prior_knowledge import PriorKnowledge
from knowledge.persistence import KnowledgePersistence
from interaction.human_guidance import HumanGuidance

class ARCSolver:
    """ARC问题解决框架的主类"""
    
    def __init__(self, knowledge_dir: str = "./knowledge_store"):
        """
        初始化ARC解决框架
        
        Args:
            knowledge_dir: 知识存储目录
        """
        # 初始化各个组件
        self.rule_engine = RuleEngine()
        self.pattern_recognizer = PatternRecognizer()
        self.weight_calculator = ObjectWeightCalculator()
        self.prior_knowledge = PriorKnowledge()
        self.persistence = KnowledgePersistence(knowledge_dir)
        self.human_guidance = HumanGuidance(self.rule_engine, self.prior_knowledge)
        
        # 加载持久化的知识
        self._load_persisted_knowledge()
    
    def _load_persisted_knowledge(self):
        """加载持久化的知识"""
        # 加载先验知识
        prior = self.persistence.load_json("prior", "knowledge")
        if prior:
            self.prior_knowledge.shapes = prior.get("shapes", {})
            self.prior_knowledge.color_patterns = prior.get("color_patterns", {})
            self.prior_knowledge.transformations = prior.get("transformations", {})
            self.prior_knowledge.action_patterns = prior.get("action_patterns", {})
        
        # 加载人工指导历史
        guidance_history = self.persistence.load_json("guidance", "history")
        if guidance_history:
            for item in guidance_history:
                if item['type'] == 'add_rule':
                    self.human_guidance.add_rule(
                        item['name'], item['description'], item['code'], item.get('confidence', 1.0)
                    )
                elif item['type'] == 'add_shape':
                    self.human_guidance.add_shape_knowledge(
                        item['name'], item['description'], item['detection_rule']
                    )
                elif item['type'] == 'add_transformation':
                    self.human_guidance.add_transformation_knowledge(
                        item['name'], item['description'], item['parameters']
                    )
                elif item['type'] == 'add_action_pattern':
                    self.human_guidance.add_action_pattern(
                        item['name'], item['description'], item['parameters']
                    )
    
    def load_task(self, task_json: str) -> Dict[str, Any]:
        """
        加载ARC任务
        
        Args:
            task_json: 任务JSON文件路径
            
        Returns:
            加载的任务数据
        """
        try:
            with open(task_json, 'r') as f:
                task_data = json.load(f)
            
            # 验证任务格式
            if 'train' not in task_data or 'test' not in task_data:
                raise ValueError("任务JSON格式不正确，缺少train或test字段")
            
            return task_data
        except Exception as e:
            print(f"加载任务失败: {e}")
            return None
    
    def analyze_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        分析ARC任务
        
        Args:
            task_data: 任务数据
            
        Returns:
            分析结果
        """
        analysis_result = {
            "train_examples": [],
            "common_patterns": {},
            "object_weights": {},
            "proposed_rules": []
        }
        
        # 处理训练示例
        train_examples = []
        all_input_objects = []
        all_output_objects = []
        
        for example in task_data['train']:
            input_grid = Grid(np.array(example['input']))
            output_grid = Grid(np.array(example['output']))
            
            # 分析差异
            diff_analyzer = DiffAnalyzer(input_grid, output_grid)
            in_objs, out_objs, diff_objs = diff_analyzer.extract_all_objects()
            
            # 记录对象
            all_input_objects.append(in_objs)
            all_output_objects.append(out_objs)
            
            # 分析变换
            transformations = diff_analyzer.analyze_object_transformations()
            
            # 保存分析结果
            train_examples.append({
                "input_grid": input_grid,
                "output_grid": output_grid,
                "input_objects": in_objs,
                "output_objects": out_objs,
                "diff_objects": diff_objs,
                "transformations": transformations
            })
        
        analysis_result["train_examples"] = train_examples
        
        # 分析所有示例中的共同模式
        example_pairs = [(ex["input_grid"], ex["output_grid"]) for ex in train_examples]
        common_patterns = self.pattern_recognizer.find_common_patterns(example_pairs)
        analysis_result["common_patterns"] = common_patterns
        
        # 计算对象重复出现情况
        repeated_objects = self.weight_calculator.analyze_repeated_objects(all_input_objects)
        
        # 计算对象权重
        for i, example in enumerate(train_examples):
            weights = self.weight_calculator.calculate_weights(
                example["input_objects"],
                example["diff_objects"],
                repeated_objects
            )
            analysis_result["object_weights"][i] = weights
        
        # 根据分析结果提出可能的规则
        proposed_rules = self._propose_rules(common_patterns, train_examples)
        analysis_result["proposed_rules"] = proposed_rules
        
        return analysis_result
    
    def _propose_rules(self, common_patterns: Dict[str, Any], examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        根据分析结果提出可能的规则
        
        Args:
            common_patterns: 共同模式
            examples: 训练示例列表
            
        Returns:
            提出的规则列表
        """
        proposed_rules = []
        
        # 根据颜色映射提出规则
        if "color_mapping" in common_patterns and common_patterns["color_mapping"]:
            color_mapping = common_patterns["color_mapping"]
            if color_mapping["type"] == "direct":
                for from_color, to_color in color_mapping["mapping"].items():
                    proposed_rules.append({
                        "name": f"color_mapping_{from_color}_to_{to_color}",
                        "description": f"将颜色 {from_color} 映射为 {to_color}",
                        "rule_type": "color_mapping",
                        "parameters": {"from_color": from_color, "to_color": to_color},
                        "confidence": 0.9
                    })
            elif color_mapping["type"] == "offset":
                proposed_rules.append({
                    "name": f"color_offset_{color_mapping['value']}",
                    "description": f"所有颜色值偏移 {color_mapping['value']}",
                    "rule_type": "color_offset",
                    "parameters": {"offset": color_mapping['value']},
                    "confidence": 0.8
                })
        
        # 根据变换提出规则
        if "transformations" in common_patterns and common_patterns["transformations"]:
            for transform_type, transform_info in common_patterns["transformations"].items():
                if transform_type == "position_change" and transform_info.get("common_delta"):
                    delta_row, delta_col = transform_info["common_delta"]
                    proposed_rules.append({
                        "name": f"move_objects_r{delta_row}_c{delta_col}",
                        "description": f"将对象向下移动 {delta_row} 行，向右移动 {delta_col} 列",
                        "rule_type": "object_translation",
                        "parameters": {"delta_row": delta_row, "delta_col": delta_col},
                        "confidence": 0.85
                    })
        
        # 分析对象增删模式
        for example in examples:
            # 如果输入对象少，输出对象多，可能是复制或生成
            if len(example["input_objects"]) < len(example["output_objects"]):
                # 简单启发式：检查是否是复制
                for in_obj in example["input_objects"]:
                    similar_out_objs = [out_obj for out_obj in example["output_objects"] 
                                      if in_obj.color == out_obj.color and in_obj.size == out_obj.size]
                    if len(similar_out_objs) > 1:
                        proposed_rules.append({
                            "name": "object_duplication",
                            "description": "复制对象",
                            "rule_type": "object_copy",
                            "parameters": {"copies": len(similar_out_objs) - 1},
                            "confidence": 0.7
                        })
                        break
            
            # 如果输入对象多，输出对象少，可能是删除或合并
            elif len(example["input_objects"]) > len(example["output_objects"]):
                proposed_rules.append({
                    "name": "object_deletion",
                    "description": "删除某些对象",
                    "rule_type": "object_deletion",
                    "parameters": {},
                    "confidence": 0.6
                })
        
        return proposed_rules
    
    def apply_rules_to_test(self, task_data: Dict[str, Any], analysis_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        将分析出的规则应用到测试样例
        
        Args:
            task_data: 任务数据
            analysis_result: 分析结果
            
        Returns:
            应用规则后的测试结果
        """
        test_results = []
        
        # 获取分析出的规则
        proposed_rules = analysis_result["proposed_rules"]
        
        # 按置信度排序规则
        sorted_rules = sorted(proposed_rules, key=lambda r: r.get("confidence", 0), reverse=True)
        
        # 对每个测试样例应用规则
        for test_example in task_data['test']:
            input_grid = Grid(np.array(test_example['input']))
            
            # 提取输入对象
            input_extractor = ObjectExtractor(input_grid)
            input_objects = input_extractor.extract_all()
            
            # 使用权重计算器计算对象权重
            repeated_objects = analysis_result.get("repeated_objects", {})
            weights = self.weight_calculator.calculate_weights(input_objects, repeated_objects=repeated_objects)
            
            # 按权重排序对象
            sorted_objects = self.weight_calculator.sort_objects_by_weight(input_objects, weights)
            
            # 依次尝试应用规则
            result_grid = input_grid
            applied_rules = []
            
            for rule in sorted_rules:
                rule_type = rule["rule_type"]
                params = rule["parameters"]
                
                if rule_type == "color_mapping":
                    new_grid, success = self.rule_engine.apply_rule(
                        "color_mapping", result_grid, params["from_color"], params["to_color"]
                    )
                    if success:
                        result_grid = new_grid
                        applied_rules.append(rule["name"])
                
                elif rule_type == "color_offset":
                    # 自定义颜色偏移规则
                    new_data = result_grid.data.copy()
                    mask = new_data > 0  # 只处理非零颜色
                    new_data[mask] = (new_data[mask] + params["offset"]) % 10  # 假设颜色范围是0-9
                    result_grid = Grid(new_data)
                    applied_rules.append(rule["name"])
                
                elif rule_type == "object_translation":
                    # 对排序后的对象应用平移
                    objects_to_move = [obj for obj, _ in sorted_objects]
                    new_grid, success = self.rule_engine.apply_rule(
                        "object_translation", result_grid, objects_to_move, 
                        params["delta_row"], params["delta_col"]
                    )
                    if success:
                        result_grid = new_grid
                        applied_rules.append(rule["name"])
                
                elif rule_type == "object_copy":
                    # 对高权重对象应用复制
                    if sorted_objects:
                        high_weight_obj, _ = sorted_objects[0]
                        new_grid, success = self.rule_engine.apply_rule(
                            "object_copy", result_grid, [high_weight_obj], 
                            params.get("copies", 1), 1, 1  # 默认向右下复制
                        )
                        if success:
                            result_grid = new_grid
                            applied_rules.append(rule["name"])
            
            test_results.append({
                "input": test_example['input'],
                "predicted_output": result_grid.data.tolist(),
                "applied_rules": applied_rules
            })
        
        return test_results
    
    def save_analysis_results(self, analysis_result: Dict[str, Any], file_path: str):
        """
        保存分析结果
        
        Args:
            analysis_result: 分析结果
            file_path: 保存路径
        """
        # 创建可序列化的结果
        serializable_result = {
            "common_patterns": analysis_result["common_patterns"],
            "proposed_rules": analysis_result["proposed_rules"],
            "object_weights": {k: {str(i): v for i, v in weights.items()} 
                              for k, weights in analysis_result["object_weights"].items()}
        }
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_result, f, indent=2, ensure_ascii=False)
            print(f"分析结果已保存到 {file_path}")
        except Exception as e:
            print(f"保存分析结果失败: {e}")
    
    def save_knowledge(self):
        """保存所有知识到持久化存储"""
        # 保存先验知识
        prior_data = {
            "shapes": self.prior_knowledge.shapes,
            "color_patterns": self.prior_knowledge.color_patterns,
            "transformations": self.prior_knowledge.transformations,
            "action_patterns": self.prior_knowledge.action_patterns
        }
        self.persistence.save_json(prior_data, "prior", "knowledge")
        
        # 保存指导历史
        self.persistence.save_json(self.human_guidance.guidance_history, "guidance", "history")
        
        print("所有知识已保存到持久化存储")

if __name__ == "__main__":
    # 示例用法
    solver = ARCSolver()
    
    # 加载任务
    task_data = solver.load_task("path/to/task.json")
    if task_data:
        # 分析任务
        analysis_result = solver.analyze_task(task_data)
        
        # 应用规则到测试样例
        test_results = solver.apply_rules_