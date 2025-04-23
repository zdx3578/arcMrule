import numpy as np
from typing import List, Dict, Any, Tuple
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from arc_framework import ARCSolver, RulePattern, ActionType, ARCObject

class InteractiveARCFramework:
    """交互式ARC框架，提供用户界面进行操作"""
    
    def __init__(self, solver: ARCSolver):
        self.solver = solver
        self.current_input = None
        self.current_output = None
        self.predicted_output = None
        self.colors = ['#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00', 
                      '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25']
    
    def load_problem(self, input_grid: np.ndarray, output_grid: np.ndarray = None):
        """加载问题"""
        self.current_input = input_grid
        self.current_output = output_grid
        self.predicted_output = None
    
    def solve_current_problem(self):
        """解决当前问题"""
        if self.current_input is None:
            print("请先加载问题")
            return
            
        self.predicted_output = self.solver.solve(self.current_input)
        return self.predicted_output
    
    def visualize_grids(self):
        """可视化网格"""
        if self.current_input is None:
            print("请先加载问题")
            return
            
        fig, axes = plt.subplots(1, 3 if self.predicted_output is not None else 2, figsize=(15, 5))
        
        # 创建颜色映射
        cmap = ListedColormap(self.colors)
        
        # 显示输入网格
        axes[0].imshow(self.current_input, cmap=cmap, vmin=0, vmax=9)
        axes[0].set_title("输入网格")
        axes[0].grid(True, color='black', linestyle='-', linewidth=0.5)
        
        # 显示输出网格（如果有）
        if self.current_output is not None:
            axes[1].imshow(self.current_output, cmap=cmap, vmin=0, vmax=9)
            axes[1].set_title("期望输出")
            axes[1].grid(True, color='black', linestyle='-', linewidth=0.5)
        
        # 显示预测输出（如果有）
        if self.predicted_output is not None:
            idx = 2 if self.current_output is not None else 1
            axes[idx].imshow(self.predicted_output, cmap=cmap, vmin=0, vmax=9)
            axes[idx].set_title("预测输出")
            axes[idx].grid(True, color='black', linestyle='-', linewidth=0.5)
        
        plt.tight_layout()
        plt.show()
    
    def add_custom_rule(self):
        """添加自定义规则的交互界面"""
        print("添加自定义规则:")
        
        # 获取规则基本信息
        name = input("规则名称: ")
        description = input("规则描述: ")
        
        rule = RulePattern(name, description)
        
        # 添加动作
        print("\n可用的动作类型:")
        for i, action_type in enumerate(ActionType):
            print(f"{i+1}. {action_type.name}: {action_type.value}")
        
        action_idx = int(input("\n选择动作类型 (输入序号): ")) - 1
        selected_action = list(ActionType)[action_idx]
        
        # 获取动作参数
        params = {}
        if selected_action == ActionType.MOVE:
            params["dx"] = int(input("dx (水平移动量): "))
            params["dy"] = int(input("dy (垂直移动量): "))
        elif selected_action == ActionType.COLOR_CHANGE:
            params["color"] = int(input("新颜色 (0-9): "))
        elif selected_action == ActionType.ROTATE:
            params["angle"] = int(input("旋转角度 (90, 180, 270): "))
        # 添加更多动作类型的参数配置...
        
        # 添加动作到规则
        rule.add_action(selected_action, params)
        
        # 添加简单条件
        color = int(input("\n规则适用的对象颜色 (-1 表示任意): "))
        
        def match_condition(obj, grid):
            return color == -1 or obj.color == color
            
        condition_desc = "匹配任意颜色的对象" if color == -1 else f"匹配颜色为{color}的对象"
        rule.add_condition(match_condition, condition_desc)
        
        # 添加规则到求解器
        self.solver.add_rule(rule)
        print(f"\n规则 '{name}' 已添加!")
        
        return rule
    
    def update_weight_rules(self):
        """更新权重规则的交互界面"""
        print("当前权重规则:")
        for i, (rule_name, value) in enumerate(self.solver.weight_rules.items()):
            print(f"{i+1}. {rule_name}: {value}")
        
        # 选择要更新的规则
        rule_idx = int(input("\n选择要更新的规则 (输入序号): ")) - 1
        rule_name = list(self.solver.weight_rules.keys())[rule_idx]
        
        # 输入新权重
        new_weight = float(input(f"为 '{rule_name}' 输入新权重: "))
        
        # 更新权重
        self.solver.update_weight_rule(rule_name, new_weight)
        print(f"\n'{rule_name}' 权重已更新为 {new_weight}")
    
    def add_object_template(self):
        """添加对象模板的交互界面"""
        print("添加对象模板:")
        
        # 获取模板名称
        name = input("模板名称: ")
        
        # 获取模板尺寸
        rows = int(input("行数: "))
        cols = int(input("列数: "))
        
        # 创建模板
        template = np.zeros((rows, cols), dtype=int)
        
        print("\n输入模板内容，每个单元格输入一个0-9的数字:")
        for r in range(rows):
            row_input = input(f"行 {r+1}: ").strip().split()
            for c in range(min(cols, len(row_input))):
                template[r, c] = int(row_input[c])
        
        # 添加模板
        self.solver.add_object_template(name, template)
        print(f"\n模板 '{name}' 已添加!")
        
        # 显示添加的模板
        plt.figure(figsize=(5, 5))
        plt.imshow(template, cmap=ListedColormap(self.colors), vmin=0, vmax=9)
        plt.title(f"模板: {name}")
        plt.grid(True, color='black', linestyle='-', linewidth=0.5)
        plt.show()
    
    def analyze_and_improve(self):
        """分析当前问题并改进规则"""
        if self.current_input is None or self.current_output is None:
            print("请先加载完整的问题（包括输入和期望输出）")
            return
        
        # 解决当前问题
        self.solve_current_problem()
        
        # 可视化结果
        self.visualize_grids()
        
        # 检查预测是否正确
        is_correct = np.array_equal(self.predicted_output, self.current_output)
        
        if is_correct:
            print("预测正确！当前规则运行良好。")
        else:
            print("预测不正确。分析差异并尝试改进规则。")
            
            # 创建差异网格
            from arc_framework import DiffGrid
            diff = DiffGrid(self.current_output, self.predicted_output)
            
            # 分析差异
            print(f"发现 {len(diff.changes)} 个不匹配的单元格。")
            
            # 提取输入和输出对象
            from arc_framework import ObjectExtractor
            input_extractor = ObjectExtractor(self.current_input)
            input_objects = input_extractor.extract_objects()
            
            output_extractor = ObjectExtractor(self.current_output)
            output_objects = output_extractor.extract_objects()
            
            # 调用规则推断
            from rule_inference import RuleInference
            rule_inferrer = RuleInference()
            inferred_rules = rule_inferrer.infer_rules(input_objects, output_objects, diff)
            
            # 显示推断的规则
            if inferred_rules:
                print("\n推断出以下可能的规则:")
                for i, rule in enumerate(inferred_rules):
                    print(f"{i+1}. {rule.name}: {rule.description}")
                
                # 询问用户是否要添加推断的规则
                add_rule = input("\n是否要添加这些规则? (y/n): ").lower() == 'y'
                
                if add_rule:
                    # 添加选择的规则
                    rule_idx = int(input("选择要添加的规则 (输入序号): ")) - 1
                    self.solver.add_rule(inferred_rules[rule_idx])
                    print(f"规则 '{inferred_rules[rule_idx].name}' 已添加!")
            else:
                print("\n未能推断出明确的规则。建议手动添加自定义规则。")
                
                # 询问用户是否要手动添加规则
                add_manual = input("是否要手动添加规则? (y/n): ").lower() == 'y'
                
                if add_manual:
                    self.add_custom_rule()
    
    def run_interactive_session(self):
        """运行交互式会话"""
        print("欢迎使用交互式ARC框架")
        
        while True:
            print("\n选择操作:")
            print("1. 加载问题")
            print("2. 解决当前问题")
            print("3. 可视化网格")
            print("4. 添加自定义规则")
            print("5. 更新权重规则")
            print("6. 添加对象模板")
            print("7. 分析并改进规则")
            print("8. 保存当前状态")
            print("9. 退出")
            
            choice = input("\n选择 (1-9): ")
            
            if choice == '1':
                # 简化版：手动输入网格
                print("输入网格 (每行用空格分隔数字，空行结束):")
                rows = []
                while True:
                    row = input()
                    if not row:
                        break
                    rows.append([int(x) for x in row.split()])
                
                if rows:
                    self.current_input = np.array(rows)
                    
                    # 输入期望输出
                    print("输入期望输出 (可选，每行用空格分隔数字，空行结束):")
                    rows = []
                    while True:
                        row = input()
                        if not row:
                            break
                        rows.append([int(x) for x in row.split()])
                    
                    self.current_output = np.array(rows) if rows else None
                    
                    print("问题已加载")
            
            elif choice == '2':
                self.solve_current_problem()
                print("问题已解决")
            
            elif choice == '3':
                self.visualize_grids()
            
            elif choice == '4':
                self.add_custom_rule()
            
            elif choice == '5':
                self.update_weight_rules()
            
            elif choice == '6':
                self.add_object_template()
            
            elif choice == '7':
                self.analyze_and_improve()
            
            elif choice == '8':
                # 保存当前状态
                self.solver.save_state()
                print("当前状态已保存")
            
            elif choice == '9':
                print("谢谢使用，再见！")
                break
            
            else:
                print("无效选择，请重试")

# 使用示例
if __name__ == "__main__":
    # 创建求解器并加载状态
    solver = ARCSolver()
    solver.load_state()
    
    # 创建交互式界面
    interactive = InteractiveARCFramework(solver)
    
    # 运行交互式会话
    interactive.run_interactive_session()