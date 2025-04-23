import numpy as np
import json
import os
from typing import List, Dict, Tuple, Any, Optional, Set
from enum import Enum
from dataclasses import dataclass
import pickle

class ObjectType(Enum):
    """对象类型枚举"""
    RECTANGLE = "rectangle"
    SQUARE = "square"
    LINE_H = "horizontal_line"
    LINE_V = "vertical_line"
    PATTERN = "pattern"
    STAR = "star"
    PIXEL = "pixel"
    GROUP = "group"
    CUSTOM = "custom"

class ActionType(Enum):
    """动作类型枚举"""
    MOVE = "move"
    ROTATE = "rotate"
    FLIP = "flip"
    COPY = "copy"
    DELETE = "delete"
    COLOR_CHANGE = "color_change"
    RESIZE = "resize"
    SYMMETRY = "symmetry"
    FILL = "fill"
    MERGE = "merge"
    SPLIT = "split"
    CUSTOM = "custom"

@dataclass
class Position:
    """位置数据类"""
    x: int
    y: int
    
    def __eq__(self, other):
        if not isinstance(other, Position):
            return False
        return self.x == other.x and self.y == other.y
    
    def __hash__(self):
        return hash((self.x, self.y))

class ARCObject:
    """ARC对象类，表示从网格中提取的对象"""
    def __init__(self, pixels: List[Position], color: int, obj_type: ObjectType = None):
        self.pixels = pixels
        self.color = color
        self.type = obj_type or self._infer_type()
        self.weight = 0.0
        self.sub_objects = []
        self.parent = None
        self.metadata = {}
        
    def _infer_type(self) -> ObjectType:
        """根据像素分布推断对象类型"""
        # 实现对象类型推断逻辑
        return ObjectType.CUSTOM
    
    def extract_sub_objects(self, max_depth: int = 3):
        """提取所有可能的子对象，直到指定深度"""
        if max_depth <= 0 or len(self.pixels) <= 1:
            return
            
        # 对于n个像素，提取所有n-1个像素的子对象
        for i in range(len(self.pixels)):
            sub_pixels = self.pixels.copy()
            removed = sub_pixels.pop(i)
            
            # 仅当子对象至少有一个像素时创建
            if sub_pixels:
                sub_obj = ARCObject(sub_pixels, self.color)
                sub_obj.parent = self
                self.sub_objects.append(sub_obj)
                
                # 递归提取子对象的子对象
                sub_obj.extract_sub_objects(max_depth - 1)
    
    def calculate_weight(self, weight_rules: Dict[str, float]):
        """计算对象权重"""
        weight = 0.0
        
        # 应用权重规则
        # 小对象权重大于大对象
        if "size_inverse" in weight_rules:
            weight += weight_rules["size_inverse"] * (1.0 / len(self.pixels))
            
        # 其他权重规则...
        # 例如：颜色稀有度、特定形状、位置等
        
        self.weight = weight
        return weight

class DiffGrid:
    """差异网格类，表示输入和输出网格之间的差异"""
    def __init__(self, input_grid: np.ndarray, output_grid: np.ndarray):
        self.input_grid = input_grid
        self.output_grid = output_grid
        self.diff = self._compute_diff()
        self.changes = self._extract_changes()
        
    def _compute_diff(self) -> np.ndarray:
        """计算输入输出网格的差异"""
        if self.input_grid.shape != self.output_grid.shape:
            # 处理尺寸不同的情况
            # 可以填充较小的网格或裁剪较大的网格
            max_rows = max(self.input_grid.shape[0], self.output_grid.shape[0])
            max_cols = max(self.input_grid.shape[1], self.output_grid.shape[1])
            
            padded_input = np.zeros((max_rows, max_cols), dtype=int) - 1
            padded_output = np.zeros((max_rows, max_cols), dtype=int) - 1
            
            padded_input[:self.input_grid.shape[0], :self.input_grid.shape[1]] = self.input_grid
            padded_output[:self.output_grid.shape[0], :self.output_grid.shape[1]] = self.output_grid
            
            diff = (padded_input != padded_output).astype(int)
            return diff
        else:
            # 尺寸相同时直接计算差异
            return (self.input_grid != self.output_grid).astype(int)
    
    def _extract_changes(self) -> List[Dict[str, Any]]:
        """提取网格变化的详细信息"""
        changes = []
        rows, cols = self.diff.shape
        
        for r in range(rows):
            for c in range(cols):
                if self.diff[r, c] == 1:
                    # 对于每个变化的像素，记录其位置和前后值
                    input_value = self.input_grid[r, c] if r < self.input_grid.shape[0] and c < self.input_grid.shape[1] else -1
                    output_value = self.output_grid[r, c] if r < self.output_grid.shape[0] and c < self.output_grid.shape[1] else -1
                    
                    changes.append({
                        "position": Position(r, c),
                        "from_value": input_value,
                        "to_value": output_value
                    })
        
        return changes

class ObjectExtractor:
    """对象提取器，从网格中提取对象"""
    def __init__(self, grid: np.ndarray):
        self.grid = grid
        self.objects = []
        
    def extract_objects(self) -> List[ARCObject]:
        """从网格中提取所有对象"""
        visited = set()
        rows, cols = self.grid.shape
        
        for r in range(rows):
            for c in range(cols):
                pos = Position(r, c)
                
                if pos not in visited and self.grid[r, c] != 0:  # 假设0表示背景
                    # 使用BFS提取连通对象
                    color = self.grid[r, c]
                    obj_pixels = self._extract_connected_component(pos, color, visited)
                    
                    if obj_pixels:
                        obj = ARCObject(obj_pixels, color)
                        self.objects.append(obj)
        
        return self.objects
    
    def _extract_connected_component(self, start: Position, color: int, visited: Set[Position]) -> List[Position]:
        """使用BFS提取连通分量"""
        queue = [start]
        component = []
        rows, cols = self.grid.shape
        
        while queue:
            pos = queue.pop(0)
            
            if pos in visited:
                continue
                
            visited.add(pos)
            
            if 0 <= pos.x < rows and 0 <= pos.y < cols and self.grid[pos.x, pos.y] == color:
                component.append(pos)
                
                # 添加相邻位置到队列
                for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    next_pos = Position(pos.x + dx, pos.y + dy)
                    if next_pos not in visited:
                        queue.append(next_pos)
        
        return component

class RulePattern:
    """规则模式，描述输入到输出的变换规则"""
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.weight = 1.0
        self.examples = []
        self.actions = []
        self.conditions = []
        
    def add_example(self, input_grid: np.ndarray, output_grid: np.ndarray):
        """添加示例"""
        self.examples.append((input_grid, output_grid))
        
    def add_action(self, action_type: ActionType, params: Dict[str, Any]):
        """添加动作"""
        self.actions.append({"type": action_type, "params": params})
        
    def add_condition(self, condition_func, description: str):
        """添加条件"""
        self.conditions.append({"func": condition_func, "description": description})
        
    def matches(self, obj: ARCObject, grid: np.ndarray) -> bool:
        """检查对象是否匹配此规则"""
        for condition in self.conditions:
            if not condition["func"](obj, grid):
                return False
        return True
        
    def apply(self, obj: ARCObject, grid: np.ndarray) -> np.ndarray:
        """应用规则到对象"""
        result_grid = grid.copy()
        
        for action in self.actions:
            result_grid = self._apply_action(action, obj, result_grid)
            
        return result_grid
    
    def _apply_action(self, action: Dict[str, Any], obj: ARCObject, grid: np.ndarray) -> np.ndarray:
        """应用单个动作"""
        action_type = action["type"]
        params = action["params"]
        result = grid.copy()
        
        # 根据动作类型实现相应逻辑
        if action_type == ActionType.MOVE:
            dx, dy = params.get("dx", 0), params.get("dy", 0)
            # 实现移动逻辑...
            
        elif action_type == ActionType.COLOR_CHANGE:
            new_color = params.get("color", 0)
            # 实现颜色变化逻辑...
            
        # 添加更多动作类型的逻辑...
        
        return result

class ARCSolver:
    """ARC问题求解器"""
    def __init__(self):
        self.rules = []
        self.weight_rules = {
            "size_inverse": 1.0,  # 小对象权重大
            "color_rarity": 0.8,  # 罕见颜色权重大
            "position_center": 0.5,  # 中心位置权重大
            "shape_complexity": 0.7,  # 复杂形状权重大
        }
        self.object_templates = {}  # 预定义对象模板
        self.persistence_path = "arc_solver_data"
        
    def load_state(self, path: str = None):
        """加载持久化状态"""
        path = path or self.persistence_path
        if os.path.exists(f"{path}/state.pkl"):
            with open(f"{path}/state.pkl", "rb") as f:
                state = pickle.load(f)
                self.rules = state.get("rules", [])
                self.weight_rules = state.get("weight_rules", self.weight_rules)
                self.object_templates = state.get("object_templates", {})
    
    def save_state(self, path: str = None):
        """保存持久化状态"""
        path = path or self.persistence_path
        os.makedirs(path, exist_ok=True)
        
        state = {
            "rules": self.rules,
            "weight_rules": self.weight_rules,
            "object_templates": self.object_templates
        }
        
        with open(f"{path}/state.pkl", "wb") as f:
            pickle.dump(state, f)
            
    def add_rule(self, rule: RulePattern):
        """添加规则"""
        self.rules.append(rule)
        
    def add_object_template(self, name: str, template: np.ndarray):
        """添加对象模板"""
        self.object_templates[name] = template
        
    def update_weight_rule(self, rule_name: str, value: float):
        """更新权重规则"""
        self.weight_rules[rule_name] = value
        
    def solve(self, input_grid: np.ndarray) -> np.ndarray:
        """解决ARC问题"""
        # 1. 提取输入网格中的对象
        extractor = ObjectExtractor(input_grid)
        objects = extractor.extract_objects()
        
        # 2. 为每个对象计算权重并排序
        for obj in objects:
            obj.calculate_weight(self.weight_rules)
            obj.extract_sub_objects()
            
        sorted_objects = sorted(objects, key=lambda x: x.weight, reverse=True)
        
        # 3. 尝试应用规则
        result_grid = input_grid.copy()
        
        for obj in sorted_objects:
            for rule in sorted(self.rules, key=lambda x: x.weight, reverse=True):
                if rule.matches(obj, result_grid):
                    result_grid = rule.apply(obj, result_grid)
                    break  # 应用第一个匹配的规则
        
        return result_grid
    
    def analyze_train_pairs(self, train_pairs: List[Tuple[np.ndarray, np.ndarray]]):
        """分析训练数据对，提取共同模式"""
        common_patterns = []
        
        # 对每对训练数据分析
        for input_grid, output_grid in train_pairs:
            # 计算差异网格
            diff = DiffGrid(input_grid, output_grid)
            
            # 提取对象
            input_extractor = ObjectExtractor(input_grid)
            input_objects = input_extractor.extract_objects()
            
            output_extractor = ObjectExtractor(output_grid)
            output_objects = output_extractor.extract_objects()
            
            # 尝试推断规则
            inferred_rules = self._infer_rules(input_objects, output_objects, diff)
            
            # 将推断的规则添加到框架中
            for rule in inferred_rules:
                self.add_rule(rule)
            
        return common_patterns
    
    def _infer_rules(self, input_objects: List[ARCObject], 
                    output_objects: List[ARCObject], 
                    diff: DiffGrid) -> List[RulePattern]:
        """从输入输出对象对中推断规则"""
        inferred_rules = []
        
        # 实现规则推断逻辑
        # 例如：检测移动、旋转、颜色变化等模式
        
        return inferred_rules
    
    def interactive_rule_refinement(self, input_grid: np.ndarray, output_grid: np.ndarray):
        """交互式规则改进"""
        # 这里实现人工指导的规则改进逻辑
        # 例如：显示预测结果、询问用户是否正确、接收用户建议等
        pass

# 使用示例
if __name__ == "__main__":
    # 创建求解器实例
    solver = ARCSolver()
    
    # 尝试加载先前的状态
    solver.load_state()
    
    # 添加规则示例
    rule = RulePattern("移动规则", "将所有对象向右移动一格")
    rule.add_action(ActionType.MOVE, {"dx": 1, "dy": 0})
    solver.add_rule(rule)
    
    # 添加对象模板示例
    square_template = np.ones((3, 3), dtype=int)
    solver.add_object_template("square3x3", square_template)
    
    # 保存状态以便将来使用
    solver.save_state()
    
    # 实际解决问题...
    # input_grid = ...
    # output_grid = solver.solve(input_grid)