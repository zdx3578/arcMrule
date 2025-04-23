import json
import os
import pickle
from typing import Dict, Any, List
import numpy as np
from arc_framework import RulePattern, ARCSolver

class PersistenceManager:
    """持久化管理器，负责保存和加载框架状态"""
    
    def __init__(self, base_path: str = "arc_solver_data"):
        self.base_path = base_path
        self.ensure_directories()
    
    def ensure_directories(self):
        """确保所需目录存在"""
        directories = [
            self.base_path,
            f"{self.base_path}/rules",
            f"{self.base_path}/templates",
            f"{self.base_path}/weight_rules",
            f"{self.base_path}/training_data"
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def save_solver_state(self, solver: ARCSolver):
        """保存求解器的完整状态"""
        state = {
            "weight_rules": solver.weight_rules,
            "object_templates": {k: v.tolist() for k, v in solver.object_templates.items()}
        }
        
        # 保存主状态
        with open(f"{self.base_path}/solver_state.json", "w") as f:
            json.dump(state, f, indent=2)
        
        # 保存规则（由于可能包含函数，使用pickle）
        with open(f"{self.base_path}/rules.pkl", "wb") as f:
            pickle.dump(solver.rules, f)
    
    def load_solver_state(self, solver: ARCSolver):
        """加载求解器的完整状态"""
        # 加载主状态
        if os.path.exists(f"{self.base_path}/solver_state.json"):
            with open(f"{self.base_path}/solver_state.json", "r") as f:
                state = json.load(f)
                solver.weight_rules = state.get("weight_rules", solver.weight_rules)
                
                # 转换回numpy数组
                templates = {}
                for k, v in state.get("object_templates", {}).items():
                    templates[k] = np.array(v)
                solver.object_templates = templates
        
        # 加载规则
        if os.path.exists(f"{self.base_path}/rules.pkl"):
            with open(f"{self.base_path}/rules.pkl", "rb") as f:
                solver.rules = pickle.load(f)
    
    def save_rule(self, rule: RulePattern):
        """单独保存一个规则"""
        rule_data = {
            "name": rule.name,
            "description": rule.description,
            "weight": rule.weight,
            "actions": [{
                "type": action["type"].value,
                "params": action["params"]
            } for action in rule.actions]
        }
        
        # 保存规则元数据（不包含条件函数）
        with open(f"{self.base_path}/rules/{rule.name}.json", "w") as f:
            json.dump(rule_data, f, indent=2)
        
        # 保存完整规则（包含条件函数）
        with open(f"{self.base_path}/rules/{rule.name}.pkl", "wb") as f:
            pickle.dump(rule, f)
    
    def load_rules(self) -> List[RulePattern]:
        """加载所有保存的规则"""
        rules = []
        rules_dir = f"{self.base_path}/rules"
        
        if not os.path.exists(rules_dir):
            return rules
        
        # 查找所有的.pkl规则文件
        for filename in os.listdir(rules_dir):
            if filename.endswith(".pkl"):
                file_path = os.path.join(rules_dir, filename)
                try:
                    with open(file_path, "rb") as f:
                        rule = pickle.load(f)
                        rules.append(rule)
                except Exception as e:
                    print(f"Error loading rule from {filename}: {e}")
        
        return rules
    
    def save_training_example(self, name: str, input_grid: np.ndarray, output_grid: np.ndarray):
        """保存训练示例"""
        example = {
            "input": input_grid.tolist(),
            "output": output_grid.tolist()
        }
        
        with open(f"{self.base_path}/training_data/{name}.json", "w") as f:
            json.dump(example, f, indent=2)
    
    def load_training_examples(self):
        """加载所有训练示例"""
        examples = []
        training_dir = f"{self.base_path}/training_data"
        
        if not os.path.exists(training_dir):
            return examples
        
        for filename in os.listdir(training_dir):
            if filename.endswith(".json"):
                file_path = os.path.join(training_dir, filename)
                try:
                    with open(file_path, "r") as f:
                        data = json.load(f)
                        input_grid = np.array(data["input"])
                        output_grid = np.array(data["output"])
                        examples.append((input_grid, output_grid))
                except Exception as e:
                    print(f"Error loading training example from {filename}: {e}")
        
        return examples