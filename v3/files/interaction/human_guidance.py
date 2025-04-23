from typing import List, Dict, Any, Callable
import json

class HumanGuidance:
    """人工指导接口，允许用户添加规则和知识"""
    
    def __init__(self, rule_engine=None, prior_knowledge=None):
        """
        初始化人工指导接口
        
        Args:
            rule_engine: 规则引擎实例
            prior_knowledge: 先验知识管理器实例
        """
        self.rule_engine = rule_engine
        self.prior_knowledge = prior_knowledge
        self.guidance_history = []
    
    def add_rule(self, name: str, description: str, rule_code: str, confidence: float = 1.0) -> bool:
        """
        添加新规则
        
        Args:
            name: 规则名称
            description: 规则描述
            rule_code: 规则函数代码字符串
            confidence: 规则置信度
            
        Returns:
            是否成功添加
        """
        if not self.rule_engine:
            print("错误: 规则引擎未初始化")
            return False
        
        try:
            # 将代码字符串转换为函数
            local_vars = {}
            exec(f"def rule_func(grid, *args, **kwargs):\n{rule_code}", globals(), local_vars)
            rule_func = local_vars['rule_func']
            
            # 添加到规则引擎
            self.rule_engine.add_rule(name, description, rule_func, confidence)
            
            # 记录到历史
            self.guidance_history.append({
                'type': 'add_rule',
                'name': name,
                'description': description,
                'code': rule_code,
                'confidence': confidence
            })
            
            return True
        except Exception as e:
            print(f"添加规则失败: {e}")
            return False
    
    def add_shape_knowledge(self, name: str, description: str, detection_rule: str) -> bool:
        """
        添加形状知识
        
        Args:
            name: 形状名称
            description: 形状描述
            detection_rule: 检测规则
            
        Returns:
            是否成功添加
        """
        if not self.prior_knowledge:
            print("错误: 先验知识管理器未初始化")
            return False
        
        try:
            self.prior_knowledge.add_shape(name, description, detection_rule)
            
            # 记录到历史
            self.guidance_history.append({
                'type': 'add_shape',
                'name': name,
                'description': description,
                'detection_rule': detection_rule
            })
            
            return True
        except Exception as e:
            print(f"添加形状知识失败: {e}")
            return False
    
    def add_transformation_knowledge(self, name: str, description: str, parameters: List[str]) -> bool:
        """
        添加变换知识
        
        Args:
            name: 变换名称
            description: 变换描述
            parameters: 变换参数列表
            
        Returns:
            是否成功添加
        """
        if not self.prior_knowledge:
            print("错误: 先验知识管理器未初始化")
            return False
        
        try:
            self.prior_knowledge.add_transformation(name, description, parameters)
            
            # 记录到历史
            self.guidance_history.append({
                'type': 'add_transformation',
                'name': name,
                'description': description,
                'parameters': parameters
            })
            
            return True
        except Exception as e:
            print(f"添加变换知识失败: {e}")
            return False
    
    def add_action_pattern(self, name: str, description: str, parameters: List[str]) -> bool:
        """
        添加动作模式
        
        Args:
            name: 模式名称
            description: 模式描述
            parameters: 模式参数列表
            
        Returns:
            是否成功添加
        """
        if not self.prior_knowledge:
            print("错误: 先验知识管理器未初始化")
            return False
        
        try:
            self.prior_knowledge.add_action_pattern(name, description, parameters)
            
            # 记录到历史
            self.guidance_history.append({
                'type': 'add_action_pattern',
                'name': name,
                'description': description,
                'parameters': parameters
            })
            
            return True
        except Exception as e:
            print(f"添加动作模式失败: {e}")
            return False
    
    def save_guidance_history(self, file_path: str) -> bool:
        """
        保存指导历史到文件
        
        Args:
            file_path: 文件路径
            
        Returns:
            是否成功保存
        """
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self.guidance_history, f, indent=2, ensure_ascii=False)
            print(f"指导历史已保存到 {file_path}")
            return True
        except Exception as e:
            print(f"保存指导历史失败: {e}")
            return False
    
    def load_guidance_history(self, file_path: str) -> bool:
        """
        从文件加载指导历史并应用
        
        Args:
            file_path: 文件路径
            
        Returns:
            是否成功加载并应用
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                history = json.load(f)
            
            success = True
            for item in history:
                if item['type'] == 'add_rule':
                    success = success and self.add_rule(
                        item['name'], item['description'], item['code'], item.get('confidence', 1.0)
                    )
                elif item['type'] == 'add_shape':
                    success = success and self.add_shape_knowledge(
                        item['name'], item['description'], item['detection_rule']
                    )
                elif item['type'] == 'add_transformation':
                    success = success and self.add_transformation_knowledge(
                        item['name'], item['description'], item['parameters']
                    )
                elif item['type'] == 'add_action_pattern':
                    success = success and self.add_action_pattern(
                        item['name'], item['description'], item['parameters']
                    )
            
            print(f"从 {file_path} {'成功' if success else '部分成功'} 加载并应用指导历史")
            return success
        except Exception as e:
            print(f"加载指导历史失败: {e}")
            return False