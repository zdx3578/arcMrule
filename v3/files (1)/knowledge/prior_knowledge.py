from typing import List, Dict, Any, Set
import json
import os

class PriorKnowledge:
    """管理ARC问题解决的先验知识"""
    
    def __init__(self, knowledge_file: str = None):
        """
        初始化先验知识管理器
        
        Args:
            knowledge_file: 知识文件路径，如果提供则从文件加载
        """
        # 初始化知识库
        self.shapes = {}  # 形状知识
        self.color_patterns = {}  # 颜色模式知识
        self.transformations = {}  # 变换规则知识
        self.action_patterns = {}  # 行动模式知识
        
        # 如果提供了文件，尝试加载
        if knowledge_file and os.path.exists(knowledge_file):
            self.load_from_file(knowledge_file)
        else:
            self._initialize_default_knowledge()
    
    def _initialize_default_knowledge(self):
        """初始化默认先验知识"""
        # 基本形状
        self.shapes = {
            "rectangle": {
                "description": "矩形，所有像素形成一个矩形",
                "detection": "is_contiguous and width * height == size"
            },
            "frame": {
                "description": "框架，外部是矩形，内部是空心",
                "detection": "is_frame_shape"
            },
            "line": {
                "description": "线，所有像素在一条直线上",
                "detection": "width == 1 or height == 1"
            },
            "l_shape": {
                "description": "L形，两条垂直的线连接",
                "detection": "is_l_shape"
            }
        }
        
        # 基本颜色模式
        self.color_patterns = {
            "single_color": {
                "description": "单一颜色对象",
                "detection": "len(unique_colors) == 1"
            },
            "alternating_colors": {
                "description": "交替颜色模式",
                "detection": "is_alternating_pattern"
            },
            "gradient": {
                "description": "渐变颜色模式",
                "detection": "is_gradient_pattern"
            }
        }
        
        # 基本变换
        self.transformations = {
            "translation": {
                "description": "对象平移",
                "parameters": ["delta_row", "delta_col"]
            },
            "rotation": {
                "description": "对象旋转",
                "parameters": ["angle"]
            },
            "reflection": {
                "description": "对象反射",
                "parameters": ["axis"]
            },
            "scaling": {
                "description": "对象缩放",
                "parameters": ["scale_factor"]
            },
            "color_change": {
                "description": "颜色变换",
                "parameters": ["from_color", "to_color"]
            }
        }
        
        # 基本动作模式
        self.action_patterns = {
            "fill": {
                "description": "填充区域",
                "parameters": ["target_area", "color"]
            },
            "delete": {
                "description": "删除对象",
                "parameters": ["target_objects"]
            },
            "copy": {
                "description": "复制对象",
                "parameters": ["source_objects", "target_position"]
            },
            "move": {
                "description": "移动对象",
                "parameters": ["objects", "delta_row", "delta_col"]
            }
        }
    
    def add_shape(self, name: str, description: str, detection_rule: str):
        """添加形状知识"""
        self.shapes[name] = {
            "description": description,
            "detection": detection_rule
        }
    
    def add_color_pattern(self, name: str, description: str, detection_rule: str):
        """添加颜色模式知识"""
        self.color_patterns[name] = {
            "description": description,
            "detection": detection_rule
        }
    
    def add_transformation(self, name: str, description: str, parameters: List[str]):
        """添加变换规则知识"""
        self.transformations[name] = {
            "description": description,
            "parameters": parameters
        }
    
    def add_action_pattern(self, name: str, description: str, parameters: List[str]):
        """添加行动模式知识"""
        self.action_patterns[name] = {
            "description": description,
            "parameters": parameters
        }
    
    def get_all_shapes(self) -> Dict[str, Dict[str, Any]]:
        """获取所有形状知识"""
        return self.shapes
    
    def get_all_color_patterns(self) -> Dict[str, Dict[str, Any]]:
        """获取所有颜色模式知识"""
        return self.color_patterns
    
    def get_all_transformations(self) -> Dict[str, Dict[str, Any]]:
        """获取所有变换规则知识"""
        return self.transformations
    
    def get_all_action_patterns(self) -> Dict[str, Dict[str, Any]]:
        """获取所有行动模式知识"""
        return self.action_patterns
    
    def load_from_file(self, file_path: str):
        """从文件加载知识"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            if 'shapes' in data:
                self.shapes = data['shapes']
            if 'color_patterns' in data:
                self.color_patterns = data['color_patterns']
            if 'transformations' in data:
                self.transformations = data['transformations']
            if 'action_patterns' in data:
                self.action_patterns = data['action_patterns']
                
            print(f"从 {file_path} 成功加载先验知识")
        except Exception as e:
            print(f"加载知识文件失败: {e}")
            self._initialize_default_knowledge()
    
    def save_to_file(self, file_path: str):
        """保存知识到文件"""
        data = {
            'shapes': self.shapes,
            'color_patterns': self.color_patterns,
            'transformations': self.transformations,
            'action_patterns': self.action_patterns
        }
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"先验知识已保存到 {file_path}")
        except Exception as e:
            print(f"保存知识文件失败: {e}")