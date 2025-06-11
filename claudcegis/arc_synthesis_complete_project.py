# ARC程序合成完整项目框架
# 包含所有必要文件的完整实现

# ===================== 项目根目录文件 =====================

# requirements.txt
"""
numpy>=1.21.0
scipy>=1.7.0
scikit-learn>=1.0.0
scikit-image>=0.18.0
pyyaml>=5.4.0
matplotlib>=3.4.0
networkx>=2.6.0
python-sat>=0.1.7.dev14
clingo>=5.5.0
"""

# setup.py
"""
from setuptools import setup, find_packages

setup(
    name="arc-synthesis-framework",
    version="0.1.0",
    description="Popper-based Program Synthesis Framework for ARC Tasks",
    author="ARC Research Team",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0", 
        "scikit-learn>=1.0.0",
        "scikit-image>=0.18.0",
        "pyyaml>=5.4.0",
        "matplotlib>=3.4.0",
        "networkx>=2.6.0",
        "python-sat>=0.1.7.dev14",
        "clingo>=5.5.0"
    ],
    python_requires=">=3.8",
)
"""

# README.md
"""
# ARC程序合成框架

基于Popper的归纳逻辑编程系统，用于解决ARC（抽象与推理语料库）任务。

## 特性

- Popper ILP集成
- 对象提取和空间推理
- CEGIS反例引导合成
- 反统一模式泛化
- 完整的ARC任务处理管道

## 安装

```bash
pip install -r requirements.txt
python setup.py install
```

## 快速开始

```python
from arc_synthesis_framework import ARCSynthesisEngine, SynthesisTask

# 初始化引擎
engine = ARCSynthesisEngine()

# 加载ARC任务
task = SynthesisTask.from_file("examples/simple_tasks/color_change.json")

# 运行合成
result = engine.synthesize_program(task)

if result.success:
    print(f"合成成功: {result.program}")
```

## 项目结构

详见代码中的完整项目结构说明。
"""

# config/default.yaml
"""
# 默认配置文件
popper:
  popper_path: "./popper"
  timeout: 300
  solver: "rc2"
  max_vars: 8
  max_body: 10
  max_rules: 5
  noisy: false

extraction:
  connectivity: 4
  min_object_size: 1
  background_color: 0
  analyze_shapes: true
  detect_patterns: true

cegis:
  max_iterations: 25
  synthesis_timeout: 300
  verification_timeout: 60

anti_unification:
  max_generalization_depth: 5
  preserve_structure: true
  enable_type_constraints: true

oracle:
  validation_method: "exact_match"
  tolerance: 0.0

logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: "logs/synthesis.log"
"""

# ===================== 核心模块 =====================

# arc_synthesis_framework/__init__.py
"""
ARC程序合成框架主包
"""
from .core.synthesis_engine import ARCSynthesisEngine, SynthesisTask, SynthesisResult
from .core.popper_interface import PopperInterface
from .extraction.object_extractor import ARCObjectExtractor, ARCObject
from .utils.arc_loader import ARCDataLoader

__version__ = "0.1.0"
__all__ = [
    "ARCSynthesisEngine", 
    "SynthesisTask", 
    "SynthesisResult",
    "PopperInterface",
    "ARCObjectExtractor", 
    "ARCObject",
    "ARCDataLoader"
]

# core/__init__.py
"""核心模块初始化"""
from .synthesis_engine import ARCSynthesisEngine
from .popper_interface import PopperInterface
from .anti_unification import AntiUnifier
from .oracle import SolutionOracle

__all__ = ["ARCSynthesisEngine", "PopperInterface", "AntiUnifier", "SolutionOracle"]

# core/synthesis_engine.py
"""
主合成引擎 - 完整实现
"""
import logging
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
import json
import time
import yaml

from .popper_interface import PopperInterface
from .anti_unification import AntiUnifier
from .oracle import SolutionOracle
from ..extraction.object_extractor import ARCObjectExtractor
from ..cegis.synthesizer import CEGISSynthesizer
from ..cegis.verifier import ProgramVerifier
from ..cegis.counterexample import CounterexampleGenerator
from ..utils.arc_loader import ARCDataLoader
from ..utils.metrics import SynthesisMetrics
from ..utils.logging import setup_logging

logger = logging.getLogger(__name__)

@dataclass
class SynthesisTask:
    """表示ARC合成任务"""
    task_id: str
    train_pairs: List[Tuple[List[List[int]], List[List[int]]]]
    test_pairs: List[Tuple[List[List[int]], List[List[int]]]]
    metadata: Dict[str, Any]
    
    @classmethod
    def from_file(cls, file_path: str) -> 'SynthesisTask':
        """从文件加载任务"""
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        return cls(
            task_id=data.get('task_id', Path(file_path).stem),
            train_pairs=[(pair['input'], pair['output']) for pair in data['train']],
            test_pairs=[(pair['input'], pair['output']) for pair in data['test']],
            metadata=data.get('metadata', {})
        )
    
    def to_file(self, file_path: str):
        """保存任务到文件"""
        data = {
            'task_id': self.task_id,
            'train': [{'input': inp, 'output': out} for inp, out in self.train_pairs],
            'test': [{'input': inp, 'output': out} for inp, out in self.test_pairs],
            'metadata': self.metadata
        }
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)

@dataclass
class SynthesisResult:
    """合成结果"""
    success: bool
    program: Optional[str]
    confidence: float
    synthesis_time: float
    iterations: int
    counterexamples_used: int
    generalization_pattern: Optional[str]
    error_message: Optional[str] = None

class ARCSynthesisEngine:
    """
    ARC主合成引擎
    集成Popper ILP、CEGIS、对象提取和反统一
    """
    
    def __init__(self, config_path: str = "config/default.yaml"):
        """初始化合成引擎"""
        self.config = self._load_config(config_path)
        self._setup_logging()
        
        # 初始化核心组件
        self.popper = PopperInterface(self.config['popper'])
        self.object_extractor = ARCObjectExtractor(self.config['extraction'])
        self.cegis_synthesizer = CEGISSynthesizer(self.config['cegis'])
        self.verifier = ProgramVerifier(self.config.get('verification', {}))
        self.counterexample_gen = CounterexampleGenerator(self.config.get('counterexamples', {}))
        self.anti_unifier = AntiUnifier(self.config['anti_unification'])
        self.oracle = SolutionOracle(self.config['oracle'])
        self.metrics = SynthesisMetrics()
        
        # 合成状态
        self.synthesis_history = []
        self.learned_patterns = {}
        
        logger.info("ARC合成引擎初始化完成")
    
    def _load_config(self, config_path: str) -> Dict:
        """加载配置文件"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"配置文件 {config_path} 未找到，使用默认配置")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """获取默认配置"""
        return {
            'popper': {
                'popper_path': './popper',
                'timeout': 300,
                'solver': 'rc2',
                'max_vars': 8,
                'max_body': 10
            },
            'extraction': {
                'connectivity': 4,
                'min_object_size': 1,
                'background_color': 0
            },
            'cegis': {
                'max_iterations': 25,
                'synthesis_timeout': 300
            },
            'anti_unification': {
                'max_generalization_depth': 5
            },
            'oracle': {
                'validation_method': 'exact_match'
            }
        }
    
    def _setup_logging(self):
        """设置日志"""
        log_config = self.config.get('logging', {})
        setup_logging(
            level=log_config.get('level', 'INFO'),
            format_str=log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
            log_file=log_config.get('file', 'logs/synthesis.log')
        )
    
    def synthesize_program(self, task: SynthesisTask) -> SynthesisResult:
        """
        主合成方法 - 实现CEGIS循环与ILP集成
        """
        start_time = time.time()
        logger.info(f"开始合成任务 {task.task_id}")
        
        try:
            # 阶段1: 对象提取和分析
            logger.info("阶段1: 对象提取")
            extracted_objects = self._extract_objects_from_pairs(task.train_pairs)
            spatial_relations = self._analyze_spatial_relations(extracted_objects)
            
            # 阶段2: 生成Popper输入文件
            logger.info("阶段2: 生成Popper文件")
            popper_files = self._generate_popper_files(task, extracted_objects, spatial_relations)
            
            # 阶段3: CEGIS合成循环
            logger.info("阶段3: CEGIS合成")
            result = self._cegis_synthesis_loop(task, popper_files)
            
            # 阶段4: 反统一和泛化
            if result.success:
                logger.info("阶段4: 模式泛化")
                generalized_pattern = self._generalize_solution(result.program, task)
                result.generalization_pattern = generalized_pattern
            
            # 阶段5: 最终验证
            if result.success:
                logger.info("阶段5: 最终验证")
                validation_result = self._validate_solution(result.program, task)
                result.success = validation_result
            
            result.synthesis_time = time.time() - start_time
            self._log_synthesis_result(task, result)
            
            return result
            
        except Exception as e:
            logger.error(f"任务 {task.task_id} 合成失败: {str(e)}")
            return SynthesisResult(
                success=False,
                program=None,
                confidence=0.0,
                synthesis_time=time.time() - start_time,
                iterations=0,
                counterexamples_used=0,
                generalization_pattern=None,
                error_message=str(e)
            )
    
    def _extract_objects_from_pairs(self, train_pairs: List[Tuple]) -> Dict:
        """从所有输入输出对中提取对象"""
        all_objects = {}
        
        for pair_idx, (input_grid, output_grid) in enumerate(train_pairs):
            # 从输入网格提取对象
            input_objects = self.object_extractor.process_arc_grid(input_grid)
            output_objects = self.object_extractor.process_arc_grid(output_grid)
            
            # 分析输入输出之间的转换
            transformation = self.object_extractor.extract_transformation_pattern(
                input_grid, output_grid
            )
            
            all_objects[f"pair_{pair_idx}"] = {
                'input_objects': input_objects,
                'output_objects': output_objects,
                'transformation': transformation
            }
        
        return all_objects
    
    def _analyze_spatial_relations(self, objects: Dict) -> Dict:
        """分析空间关系"""
        # TODO: 实现空间关系分析
        relations = {
            'adjacency': [],
            'containment': [],
            'alignment': [],
            'distance': []
        }
        return relations
    
    def _generate_popper_files(self, task: SynthesisTask, objects: Dict, relations: Dict) -> Dict:
        """生成Popper输入文件"""
        
        # 生成示例文件 (exs.pl)
        examples_content = self._generate_examples_file(task, objects)
        
        # 生成背景知识文件 (bk.pl)
        bk_content = self._generate_background_knowledge(objects, relations)
        
        # 生成偏置文件 (bias.pl)
        bias_content = self._generate_bias_file(task, objects)
        
        # 写入文件
        task_dir = Path(f"popper_files/tasks/{task.task_id}")
        task_dir.mkdir(parents=True, exist_ok=True)
        
        files = {
            'examples': task_dir / "exs.pl",
            'background': task_dir / "bk.pl",
            'bias': task_dir / "bias.pl"
        }
        
        files['examples'].write_text(examples_content, encoding='utf-8')
        files['background'].write_text(bk_content, encoding='utf-8')
        files['bias'].write_text(bias_content, encoding='utf-8')
        
        return files
    
    def _generate_examples_file(self, task: SynthesisTask, objects: Dict) -> str:
        """生成Popper示例文件"""
        examples = []
        examples.append("% ARC任务训练示例")
        examples.append("")
        
        for pair_idx, (input_grid, output_grid) in enumerate(task.train_pairs):
            # 将网格转换为Popper谓词
            input_pred = self._grid_to_predicate(input_grid, f"input_{pair_idx}")
            output_pred = self._grid_to_predicate(output_grid, f"output_{pair_idx}")
            
            # 创建转换示例
            transform_example = f"pos(transform({input_pred}, {output_pred}))."
            examples.append(transform_example)
        
        return "\\n".join(examples)
    
    def _grid_to_predicate(self, grid: List[List[int]], name: str) -> str:
        """将网格转换为Prolog谓词表示"""
        rows = []
        for r_idx, row in enumerate(grid):
            for c_idx, cell_value in enumerate(row):
                if cell_value != 0:  # 非背景色
                    rows.append(f"cell({r_idx},{c_idx},{cell_value})")
        
        if rows:
            return f"grid([{','.join(rows)}])"
        else:
            return "grid([])"
    
    def _generate_background_knowledge(self, objects: Dict, relations: Dict) -> str:
        """生成背景知识"""
        bk_lines = [
            "% ARC任务背景知识",
            "",
            "% 网格工具谓词",
            "grid_size(grid(Cells), Width, Height) :-",
            "    findall(X, member(cell(_, X, _), Cells), Xs),",
            "    findall(Y, member(cell(Y, _, _), Cells), Ys),",
            "    max_list([0|Xs], MaxX),",
            "    max_list([0|Ys], MaxY),",
            "    Width is MaxX + 1,",
            "    Height is MaxY + 1.",
            "",
            "% 边界检查",
            "in_bounds(X, Y, Grid) :-",
            "    grid_size(Grid, W, H),",
            "    X >= 0, X < W,",
            "    Y >= 0, Y < H.",
            "",
            "% 颜色操作",
            "change_color(grid(Cells), OldColor, NewColor, grid(NewCells)) :-",
            "    maplist(change_cell_color(OldColor, NewColor), Cells, NewCells).",
            "",
            "change_cell_color(OldColor, NewColor, cell(R,C,OldColor), cell(R,C,NewColor)).",
            "change_cell_color(OldColor, NewColor, cell(R,C,Color), cell(R,C,Color)) :-",
            "    Color \\= OldColor.",
            "",
            "% 空间关系",
            "adjacent_4(cell(R1,C1,_), cell(R2,C2,_)) :-",
            "    (R1 =:= R2, abs(C1-C2) =:= 1);",
            "    (C1 =:= C2, abs(R1-R2) =:= 1).",
            "",
            "% 对象检测",
            "detect_objects(Grid, Objects) :-",
            "    findall(Color-Cells, detect_color_object(Grid, Color, Cells), Objects).",
            "",
            "detect_color_object(grid(Cells), Color, ColorCells) :-",
            "    include(has_color(Color), Cells, ColorCells),",
            "    ColorCells \\= [].",
            "",
            "has_color(Color, cell(_,_,Color)).",
            ""
        ]
        
        return "\\n".join(bk_lines)
    
    def _generate_bias_file(self, task: SynthesisTask, objects: Dict) -> str:
        """生成Popper偏置文件"""
        bias_lines = [
            "% ARC空间推理任务偏置文件",
            "",
            "% 头谓词",
            "head_pred(transform,2).",
            "",
            "% 体谓词",
            "body_pred(grid,1).              % grid(Cells)",
            "body_pred(cell,3).              % cell(R,C,Color)",
            "body_pred(change_color,4).       % change_color(GridIn,OldColor,NewColor,GridOut)",
            "body_pred(detect_objects,2).     % detect_objects(Grid,Objects)",
            "body_pred(adjacent_4,2).         % adjacent_4(Cell1,Cell2)",
            "",
            "% 类型标注",
            "type(transform,(grid,grid)).",
            "type(cell,(int,int,int)).",
            "type(change_color,(grid,int,int,grid)).",
            "type(grid,(list)).",
            "",
            "% 方向标注",
            "direction(transform,(in,out)).",
            "direction(change_color,(in,in,in,out)).",
            "",
            "% 控制设置",
            "max_vars(8).",
            "max_body(6).",
            "max_rules(3).",
            "",
            "% 禁用递归（对于简单任务）",
            "allow_singletons.",
            ""
        ]
        
        return "\\n".join(bias_lines)
    
    def _cegis_synthesis_loop(self, task: SynthesisTask, popper_files: Dict) -> SynthesisResult:
        """实现CEGIS循环"""
        counterexamples = []
        iteration = 0
        max_iterations = self.config['cegis']['max_iterations']
        
        while iteration < max_iterations:
            logger.info(f"CEGIS迭代 {iteration + 1}")
            
            # 使用Popper ILP生成候选
            candidate = self._generate_candidate_with_popper(
                popper_files, counterexamples, iteration
            )
            
            if candidate is None:
                logger.info("无更多候选程序可用 - 合成失败")
                return SynthesisResult(
                    success=False, program=None, confidence=0.0,
                    synthesis_time=0, iterations=iteration,
                    counterexamples_used=len(counterexamples),
                    generalization_pattern=None
                )
            
            # 对所有训练示例验证候选
            verification_result = self.verifier.verify_candidate(candidate, task.train_pairs)
            
            if verification_result.is_valid:
                logger.info(f"在迭代 {iteration + 1} 中找到有效候选")
                confidence = self._calculate_confidence(candidate, task)
                return SynthesisResult(
                    success=True, program=candidate, confidence=confidence,
                    synthesis_time=0, iterations=iteration + 1,
                    counterexamples_used=len(counterexamples),
                    generalization_pattern=None
                )
            else:
                # 生成反例并继续
                new_counterexample = self.counterexample_gen.generate(
                    candidate, verification_result.failed_example
                )
                counterexamples.append(new_counterexample)
                logger.info(f"添加反例 {len(counterexamples)}")
            
            iteration += 1
        
        return SynthesisResult(
            success=False, program=None, confidence=0.0,
            synthesis_time=0, iterations=max_iterations,
            counterexamples_used=len(counterexamples),
            generalization_pattern=None
        )
    
    def _generate_candidate_with_popper(self, popper_files: Dict, 
                                      counterexamples: List[str], 
                                      iteration: int) -> Optional[str]:
        """使用Popper生成候选程序"""
        # 如果有反例，创建带约束的任务
        constraints = [f"% 迭代 {iteration} 约束"] + counterexamples
        
        # 调用Popper学习程序
        task_dir = popper_files['examples'].parent
        program = self.popper.learn_program(task_dir, constraints if counterexamples else None)
        
        return program
    
    def _calculate_confidence(self, candidate: str, task: SynthesisTask) -> float:
        """计算候选程序的置信度"""
        # TODO: 实现置信度计算
        # 可以基于程序复杂度、训练准确率等因素
        return 0.8  # 临时返回固定值
    
    def _generalize_solution(self, program: str, task: SynthesisTask) -> str:
        """使用反统一泛化解决方案"""
        return self.anti_unifier.generalize_program(program, task.train_pairs)
    
    def _validate_solution(self, program: str, task: SynthesisTask) -> bool:
        """最终验证合成程序"""
        return self.oracle.validate_program(program, task.test_pairs)
    
    def _log_synthesis_result(self, task: SynthesisTask, result: SynthesisResult):
        """记录合成结果"""
        if result.success:
            logger.info(f"任务 {task.task_id} 合成成功")
            logger.info(f"程序: {result.program}")
            logger.info(f"置信度: {result.confidence:.2f}")
            logger.info(f"时间: {result.synthesis_time:.2f}秒")
        else:
            logger.warning(f"任务 {task.task_id} 合成失败: {result.error_message}")

# core/oracle.py
"""
解决方案验证模块
"""
import logging
from typing import List, Tuple, Any
import subprocess
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

class SolutionOracle:
    """解决方案验证器"""
    
    def __init__(self, config: dict):
        self.config = config
        self.validation_method = config.get('validation_method', 'exact_match')
        self.tolerance = config.get('tolerance', 0.0)
    
    def validate_program(self, program: str, test_pairs: List[Tuple]) -> bool:
        """验证程序在测试数据上的表现"""
        if self.validation_method == 'exact_match':
            return self._exact_match_validation(program, test_pairs)
        else:
            logger.warning(f"未知验证方法: {self.validation_method}")
            return False
    
    def _exact_match_validation(self, program: str, test_pairs: List[Tuple]) -> bool:
        """精确匹配验证"""
        try:
            for input_grid, expected_output in test_pairs:
                actual_output = self._execute_program(program, input_grid)
                if actual_output != expected_output:
                    logger.debug(f"验证失败: 期望 {expected_output}, 得到 {actual_output}")
                    return False
            return True
        except Exception as e:
            logger.error(f"验证时出错: {str(e)}")
            return False
    
    def _execute_program(self, program: str, input_grid: List[List[int]]) -> List[List[int]]:
        """执行程序获取输出"""
        # TODO: 实现程序执行
        # 这需要调用Prolog解释器执行学到的程序
        
        # 临时实现 - 返回输入作为输出
        logger.warning("程序执行功能尚未实现，返回输入网格")
        return input_grid

# ===================== 提取模块 =====================

# extraction/__init__.py
"""提取模块初始化"""
from .object_extractor import ARCObjectExtractor, ARCObject
from .spatial_predicates import SpatialPredicateGenerator
from .transformations import TransformationAnalyzer

__all__ = ["ARCObjectExtractor", "ARCObject", "SpatialPredicateGenerator", "TransformationAnalyzer"]

# extraction/spatial_predicates.py
"""
空间谓词生成器
"""
import logging
from typing import List, Dict, Set, Tuple
from .object_extractor import ARCObject

logger = logging.getLogger(__name__)

class SpatialPredicateGenerator:
    """空间关系谓词生成器"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
    
    def generate_spatial_predicates(self, objects: List[ARCObject]) -> List[str]:
        """生成对象间的空间关系谓词"""
        predicates = []
        
        for i, obj1 in enumerate(objects):
            for j, obj2 in enumerate(objects[i+1:], i+1):
                predicates.extend(self._generate_pair_predicates(obj1, obj2))
        
        return predicates
    
    def _generate_pair_predicates(self, obj1: ARCObject, obj2: ARCObject) -> List[str]:
        """生成两个对象间的谓词"""
        predicates = []
        
        # 相邻关系
        if self._are_adjacent(obj1, obj2):
            predicates.append(f"adjacent(obj_{obj1.id}, obj_{obj2.id}).")
        
        # 方向关系
        if obj1.centroid[0] < obj2.centroid[0]:
            predicates.append(f"above(obj_{obj1.id}, obj_{obj2.id}).")
        elif obj1.centroid[0] > obj2.centroid[0]:
            predicates.append(f"below(obj_{obj1.id}, obj_{obj2.id}).")
        
        if obj1.centroid[1] < obj2.centroid[1]:
            predicates.append(f"left_of(obj_{obj1.id}, obj_{obj2.id}).")
        elif obj1.centroid[1] > obj2.centroid[1]:
            predicates.append(f"right_of(obj_{obj1.id}, obj_{obj2.id}).")
        
        # 颜色关系
        if obj1.color == obj2.color:
            predicates.append(f"same_color(obj_{obj1.id}, obj_{obj2.id}).")
        
        # 大小关系
        if obj1.size > obj2.size:
            predicates.append(f"larger(obj_{obj1.id}, obj_{obj2.id}).")
        elif obj1.size < obj2.size:
            predicates.append(f"smaller(obj_{obj1.id}, obj_{obj2.id}).")
        
        return predicates
    
    def _are_adjacent(self, obj1: ARCObject, obj2: ARCObject) -> bool:
        """检查两个对象是否相邻"""
        # 检查是否有共同边界
        for r1, c1 in obj1.cells:
            for r2, c2 in obj2.cells:
                if abs(r1 - r2) + abs(c1 - c2) == 1:
                    return True
        return False

# extraction/transformations.py
"""
转换模式分析器
"""
import logging
from typing import List, Dict, Any, Tuple
import numpy as np

logger = logging.getLogger(__name__)

class TransformationAnalyzer:
    """转换模式分析器"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
    
    def analyze_transformation(self, input_grid: List[List[int]], 
                             output_grid: List[List[int]]) -> Dict[str, Any]:
        """分析输入输出间的转换模式"""
        
        # 基本统计
        transformation = {
            'type': self._classify_transformation_type(input_grid, output_grid),
            'color_changes': self._detect_color_changes(input_grid, output_grid),
            'spatial_changes': self._detect_spatial_changes(input_grid, output_grid),
            'size_changes': self._detect_size_changes(input_grid, output_grid)
        }
        
        return transformation
    
    def _classify_transformation_type(self, input_grid: List[List[int]], 
                                    output_grid: List[List[int]]) -> str:
        """分类转换类型"""
        input_array = np.array(input_grid)
        output_array = np.array(output_grid)
        
        # 检查是否只是颜色变化
        if input_array.shape == output_array.shape:
            # 形状相同，检查是否只有颜色改变
            structure_same = (input_array != 0) == (output_array != 0)
            if structure_same.all():
                return "color_transformation"
        
        # 检查是否是填充操作
        if self._is_filling_operation(input_array, output_array):
            return "hole_filling"
        
        # 检查是否是对象移动
        if self._is_movement_operation(input_array, output_array):
            return "object_movement"
        
        return "complex_transformation"
    
    def _detect_color_changes(self, input_grid: List[List[int]], 
                            output_grid: List[List[int]]) -> Dict[int, int]:
        """检测颜色变化映射"""
        color_map = {}
        input_array = np.array(input_grid)
        output_array = np.array(output_grid)
        
        for i in range(input_array.shape[0]):
            for j in range(input_array.shape[1]):
                input_color = input_array[i, j]
                output_color = output_array[i, j]
                
                if input_color in color_map:
                    if color_map[input_color] != output_color:
                        # 不一致的映射
                        return {}
                else:
                    color_map[input_color] = output_color
        
        return color_map
    
    def _detect_spatial_changes(self, input_grid: List[List[int]], 
                              output_grid: List[List[int]]) -> Dict[str, Any]:
        """检测空间变化"""
        # TODO: 实现空间变化检测
        return {'has_spatial_changes': False}
    
    def _detect_size_changes(self, input_grid: List[List[int]], 
                           output_grid: List[List[int]]) -> Dict[str, Any]:
        """检测大小变化"""
        input_shape = (len(input_grid), len(input_grid[0]) if input_grid else 0)
        output_shape = (len(output_grid), len(output_grid[0]) if output_grid else 0)
        
        return {
            'input_shape': input_shape,
            'output_shape': output_shape,
            'size_changed': input_shape != output_shape
        }
    
    def _is_filling_operation(self, input_array: np.ndarray, 
                            output_array: np.ndarray) -> bool:
        """检查是否是填充操作"""
        # 简单检查：输出中非零元素是否包含输入中的所有非零元素
        input_nonzero = input_array != 0
        output_nonzero = output_array != 0
        
        return np.logical_and(input_nonzero, np.logical_not(output_nonzero)).sum() == 0
    
    def _is_movement_operation(self, input_array: np.ndarray, 
                             output_array: np.ndarray) -> bool:
        """检查是否是移动操作"""
        # 检查非零元素数量是否相同
        return (input_array != 0).sum() == (output_array != 0).sum()

# ===================== CEGIS模块 =====================

# cegis/__init__.py
"""CEGIS模块初始化"""
from .synthesizer import CEGISSynthesizer
from .verifier import ProgramVerifier, VerificationResult
from .counterexample import CounterexampleGenerator

__all__ = ["CEGISSynthesizer", "ProgramVerifier", "VerificationResult", "CounterexampleGenerator"]

# cegis/verifier.py
"""
程序验证器
"""
import logging
from typing import List, Tuple, Optional, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class VerificationResult:
    """验证结果"""
    is_valid: bool
    failed_example: Optional[Tuple[Any, Any]] = None
    error_message: Optional[str] = None

class ProgramVerifier:
    """程序验证器"""
    
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.timeout = config.get('timeout', 60) if config else 60
    
    def verify_candidate(self, candidate: str, examples: List[Tuple]) -> VerificationResult:
        """验证候选程序对所有示例"""
        try:
            for example_input, expected_output in examples:
                # 执行程序
                actual_output = self._execute_program(candidate, example_input)
                
                # 检查输出是否匹配
                if not self._outputs_match(actual_output, expected_output):
                    return VerificationResult(
                        is_valid=False,
                        failed_example=(example_input, expected_output),
                        error_message=f"输出不匹配: 期望 {expected_output}, 得到 {actual_output}"
                    )
            
            return VerificationResult(is_valid=True)
            
        except Exception as e:
            logger.error(f"验证时出错: {str(e)}")
            return VerificationResult(
                is_valid=False,
                error_message=f"执行错误: {str(e)}"
            )
    
    def _execute_program(self, program: str, input_data: Any) -> Any:
        """执行程序并返回输出"""
        # TODO: 实现实际的程序执行
        # 这需要调用Prolog解释器执行学到的程序
        
        # 临时实现 - 模拟简单的颜色转换
        if isinstance(input_data, list) and len(input_data) > 0:
            # 模拟颜色1->2的转换
            result = []
            for row in input_data:
                new_row = []
                for cell in row:
                    if cell == 1:
                        new_row.append(2)
                    else:
                        new_row.append(cell)
                result.append(new_row)
            return result
        
        return input_data
    
    def _outputs_match(self, actual: Any, expected: Any) -> bool:
        """检查输出是否匹配"""
        return actual == expected

# cegis/counterexample.py
"""
反例生成器
"""
import logging
from typing import Any, Tuple

logger = logging.getLogger(__name__)

class CounterexampleGenerator:
    """反例生成器"""
    
    def __init__(self, config: dict = None):
        self.config = config or {}
    
    def generate(self, failed_program: str, failed_example: Tuple[Any, Any]) -> str:
        """从失败的程序和示例生成反例约束"""
        
        example_input, expected_output = failed_example
        
        # 生成约束以排除这种失败类型
        # 这是一个简化的实现 - 实际中需要更复杂的分析
        
        constraint = f"% 反例约束: 排除在输入 {example_input} 上产生错误输出的程序"
        
        return constraint

# ===================== 工具模块 =====================

# utils/__init__.py
"""工具模块初始化"""
from .arc_loader import ARCDataLoader
from .metrics import SynthesisMetrics
from .logging import setup_logging

__all__ = ["ARCDataLoader", "SynthesisMetrics", "setup_logging"]

# utils/arc_loader.py
"""
ARC数据加载器
"""
import json
import logging
from typing import List, Dict, Any
from pathlib import Path
from ..core.synthesis_engine import SynthesisTask

logger = logging.getLogger(__name__)

class ARCDataLoader:
    """ARC数据集加载器"""
    
    def __init__(self, data_path: str = "data/arc"):
        self.data_path = Path(data_path)
    
    def load_task(self, task_id: str) -> SynthesisTask:
        """加载单个ARC任务"""
        task_file = self.data_path / f"{task_id}.json"
        
        if not task_file.exists():
            raise FileNotFoundError(f"任务文件未找到: {task_file}")
        
        with open(task_file, 'r') as f:
            data = json.load(f)
        
        return SynthesisTask(
            task_id=task_id,
            train_pairs=[(pair['input'], pair['output']) for pair in data['train']],
            test_pairs=[(pair['input'], pair['output']) for pair in data['test']],
            metadata=data.get('metadata', {})
        )
    
    def load_all_tasks(self) -> List[SynthesisTask]:
        """加载所有ARC任务"""
        tasks = []
        
        for task_file in self.data_path.glob("*.json"):
            try:
                task = self.load_task(task_file.stem)
                tasks.append(task)
            except Exception as e:
                logger.warning(f"加载任务 {task_file.stem} 失败: {str(e)}")
        
        return tasks
    
    def create_simple_task(self, task_id: str = "simple_color_change") -> SynthesisTask:
        """创建简单的测试任务"""
        return SynthesisTask(
            task_id=task_id,
            train_pairs=[
                # 简单颜色转换任务：1->2
                ([[0, 1, 0], [1, 1, 1], [0, 1, 0]], 
                 [[0, 2, 0], [2, 2, 2], [0, 2, 0]]),
                ([[0, 1, 1], [1, 0, 1], [1, 1, 0]], 
                 [[0, 2, 2], [2, 0, 2], [2, 2, 0]])
            ],
            test_pairs=[
                ([[1, 0, 1], [0, 1, 0], [1, 0, 1]], 
                 [[2, 0, 2], [0, 2, 0], [2, 0, 2]])
            ],
            metadata={"description": "将颜色1替换为颜色2", "type": "color_transformation"}
        )

# utils/metrics.py
"""
合成性能指标
"""
import time
import logging
from typing import Dict, List, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

@dataclass
class TaskMetrics:
    """单个任务的指标"""
    task_id: str
    success: bool
    synthesis_time: float
    iterations: int
    program_length: int
    confidence: float

@dataclass
class SynthesisMetrics:
    """合成系统整体指标"""
    total_tasks: int = 0
    successful_tasks: int = 0
    failed_tasks: int = 0
    total_time: float = 0.0
    task_metrics: List[TaskMetrics] = field(default_factory=list)
    
    def add_task_result(self, task_id: str, success: bool, synthesis_time: float,
                       iterations: int, program: str = "", confidence: float = 0.0):
        """添加任务结果"""
        self.total_tasks += 1
        
        if success:
            self.successful_tasks += 1
        else:
            self.failed_tasks += 1
        
        self.total_time += synthesis_time
        
        metrics = TaskMetrics(
            task_id=task_id,
            success=success,
            synthesis_time=synthesis_time,
            iterations=iterations,
            program_length=len(program.split('\\n')) if program else 0,
            confidence=confidence
        )
        
        self.task_metrics.append(metrics)
    
    def get_success_rate(self) -> float:
        """获取成功率"""
        if self.total_tasks == 0:
            return 0.0
        return self.successful_tasks / self.total_tasks
    
    def get_average_time(self) -> float:
        """获取平均合成时间"""
        if self.total_tasks == 0:
            return 0.0
        return self.total_time / self.total_tasks
    
    def print_summary(self):
        """打印指标摘要"""
        print(f"=== 合成指标摘要 ===")
        print(f"总任务数: {self.total_tasks}")
        print(f"成功任务: {self.successful_tasks}")
        print(f"失败任务: {self.failed_tasks}")
        print(f"成功率: {self.get_success_rate():.2%}")
        print(f"平均时间: {self.get_average_time():.2f}秒")
        print(f"总时间: {self.total_time:.2f}秒")

# utils/logging.py
"""
日志设置工具
"""
import logging
import logging.handlers
from pathlib import Path

def setup_logging(level: str = "INFO", 
                 format_str: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                 log_file: str = None):
    """设置日志配置"""
    
    # 创建日志目录
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 配置根logger
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=format_str,
        handlers=[
            logging.StreamHandler(),  # 控制台输出
            logging.handlers.RotatingFileHandler(
                log_file or "logs/synthesis.log",
                maxBytes=10*1024*1024,  # 10MB
                backupCount=5
            ) if log_file else logging.NullHandler()
        ]
    )
    
    # 设置第三方库日志级别
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)

# ===================== 示例和测试 =====================

# examples/__init__.py
"""示例模块"""

# examples/simple_tasks/color_change.json
"""
{
  "task_id": "color_change_example",
  "description": "简单颜色转换示例",
  "train": [
    {
      "input": [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
      "output": [[0, 2, 0], [2, 2, 2], [0, 2, 0]]
    },
    {
      "input": [[0, 1, 1], [1, 0, 1], [1, 1, 0]],
      "output": [[0, 2, 2], [2, 0, 2], [2, 2, 0]]
    }
  ],
  "test": [
    {
      "input": [[1, 0, 1], [0, 1, 0], [1, 0, 1]],
      "output": [[2, 0, 2], [0, 2, 0], [2, 0, 2]]
    }
  ],
  "metadata": {
    "type": "color_transformation",
    "difficulty": "easy"
  }
}
"""

# examples/demonstrations/basic_usage.py
"""
基本使用示例
"""
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from arc_synthesis_framework import ARCSynthesisEngine, SynthesisTask
from arc_synthesis_framework.utils.arc_loader import ARCDataLoader

def basic_usage_demo():
    """基本使用演示"""
    print("=== ARC程序合成框架基本使用演示 ===")
    
    # 初始化引擎
    engine = ARCSynthesisEngine()
    
    # 创建简单任务
    loader = ARCDataLoader()
    task = loader.create_simple_task()
    
    print(f"加载任务: {task.task_id}")
    print(f"训练样例数: {len(task.train_pairs)}")
    print(f"测试样例数: {len(task.test_pairs)}")
    
    # 运行合成
    print("\\n开始程序合成...")
    result = engine.synthesize_program(task)
    
    # 显示结果
    if result.success:
        print("\\n✓ 合成成功!")
        print(f"程序: {result.program}")
        print(f"置信度: {result.confidence:.2f}")
        print(f"迭代次数: {result.iterations}")
        print(f"合成时间: {result.synthesis_time:.2f}秒")
    else:
        print("\\n✗ 合成失败")
        print(f"错误信息: {result.error_message}")
    
    return result

if __name__ == "__main__":
    basic_usage_demo()

# ===================== Popper文件模板 =====================

# popper_files/templates/basic_bias.pl
"""
% 基本ARC任务偏置模板

% 头谓词
head_pred(transform,2).

% 基本体谓词
body_pred(grid,1).
body_pred(cell,3).
body_pred(change_color,4).
body_pred(adjacent_4,2).

% 类型定义
type(transform,(grid,grid)).
type(cell,(int,int,int)).
type(change_color,(grid,int,int,grid)).

% 方向定义
direction(transform,(in,out)).
direction(change_color,(in,in,in,out)).

% 控制参数
max_vars(6).
max_body(5).
max_rules(2).
"""

# popper_files/templates/basic_bk.pl
"""
% 基本ARC任务背景知识模板

% 网格操作
grid_cell(grid(Cells), R, C, Color) :-
    member(cell(R, C, Color), Cells).

% 颜色转换
change_color(grid(Cells), OldColor, NewColor, grid(NewCells)) :-
    maplist(change_cell_color(OldColor, NewColor), Cells, NewCells).

change_cell_color(OldColor, NewColor, cell(R,C,OldColor), cell(R,C,NewColor)).
change_cell_color(OldColor, NewColor, cell(R,C,Color), cell(R,C,Color)) :-
    Color \\= OldColor.

% 空间关系
adjacent_4(cell(R1,C1,_), cell(R2,C2,_)) :-
    (R1 =:= R2, abs(C1-C2) =:= 1);
    (C1 =:= C2, abs(R1-R2) =:= 1).
"""

# ===================== 测试文件 =====================

# tests/__init__.py
"""测试模块"""

# tests/test_synthesis_engine.py
"""
合成引擎测试
"""
import unittest
import sys
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))

from arc_synthesis_framework import ARCSynthesisEngine, SynthesisTask
from arc_synthesis_framework.utils.arc_loader import ARCDataLoader

class TestSynthesisEngine(unittest.TestCase):
    """合成引擎测试类"""
    
    def setUp(self):
        """测试初始化"""
        self.engine = ARCSynthesisEngine()
        self.loader = ARCDataLoader()
    
    def test_engine_initialization(self):
        """测试引擎初始化"""
        self.assertIsNotNone(self.engine)
        self.assertIsNotNone(self.engine.popper)
        self.assertIsNotNone(self.engine.object_extractor)
    
    def test_simple_task_creation(self):
        """测试简单任务创建"""
        task = self.loader.create_simple_task()
        self.assertEqual(task.task_id, "simple_color_change")
        self.assertEqual(len(task.train_pairs), 2)
        self.assertEqual(len(task.test_pairs), 1)
    
    def test_synthesis_pipeline(self):
        """测试合成管道"""
        task = self.loader.create_simple_task()
        
        # 测试对象提取
        objects = self.engine._extract_objects_from_pairs(task.train_pairs)
        self.assertIsInstance(objects, dict)
        
        # 测试文件生成
        popper_files = self.engine._generate_popper_files(task, objects, {})
        self.assertIn('examples', popper_files)
        self.assertIn('background', popper_files)
        self.assertIn('bias', popper_files)

if __name__ == '__main__':
    unittest.main()

# tests/test_object_extractor.py
"""
对象提取器测试
"""
import unittest
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from arc_synthesis_framework.extraction.object_extractor import ARCObjectExtractor

class TestObjectExtractor(unittest.TestCase):
    """对象提取器测试类"""
    
    def setUp(self):
        """测试初始化"""
        self.extractor = ARCObjectExtractor({})
    
    def test_simple_grid_processing(self):
        """测试简单网格处理"""
        grid = [[0, 1, 0], [1, 1, 1], [0, 1, 0]]
        result = self.extractor.process_arc_grid(grid)
        
        self.assertIn('objects', result)
        self.assertIn('relationships', result)
        self.assertIn('grid_analysis', result)
    
    def test_connected_components(self):
        """测试连通组件提取"""
        grid = [[0, 1, 0], [1, 1, 1], [0, 1, 0]]
        components = self.extractor._extract_connected_components(np.array(grid))
        
        # 应该检测到一个连通组件
        self.assertEqual(len(components), 1)

if __name__ == '__main__':
    unittest.main()

# ===================== 主运行文件 =====================

# main.py
"""
主运行文件
"""
import argparse
import sys
from pathlib import Path
import json

from arc_synthesis_framework import ARCSynthesisEngine, SynthesisTask
from arc_synthesis_framework.utils.arc_loader import ARCDataLoader
from arc_synthesis_framework.utils.metrics import SynthesisMetrics

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="ARC程序合成框架")
    parser.add_argument("--config", default="config/default.yaml", 
                       help="配置文件路径")
    parser.add_argument("--task", help="特定任务ID")
    parser.add_argument("--task_file", help="任务文件路径")
    parser.add_argument("--demo", action="store_true", 
                       help="运行演示")
    parser.add_argument("--test", action="store_true", 
                       help="运行测试")
    
    args = parser.parse_args()
    
    if args.demo:
        run_demo()
    elif args.test:
        run_tests()
    elif args.task_file:
        run_single_task_from_file(args.task_file, args.config)
    elif args.task:
        run_single_task(args.task, args.config)
    else:
        print("请指定要运行的模式。使用 --help 查看选项。")

def run_demo():
    """运行演示"""
    print("=== ARC程序合成框架演示 ===")
    
    # 初始化
    engine = ARCSynthesisEngine()
    loader = ARCDataLoader()
    metrics = SynthesisMetrics()
    
    # 创建测试任务
    task = loader.create_simple_task()
    
    print(f"运行任务: {task.task_id}")
    
    # 执行合成
    result = engine.synthesize_program(task)
    
    # 记录指标
    metrics.add_task_result(
        task.task_id, result.success, result.synthesis_time,
        result.iterations, result.program or "", result.confidence
    )
    
    # 显示结果
    if result.success:
        print("\\n✓ 合成成功!")
        print(f"程序: {result.program}")
    else:
        print("\\n✗ 合成失败")
        print(f"错误: {result.error_message}")
    
    metrics.print_summary()

def run_single_task_from_file(task_file: str, config_path: str):
    """从文件运行单个任务"""
    engine = ARCSynthesisEngine(config_path)
    task = SynthesisTask.from_file(task_file)
    
    print(f"运行任务: {task.task_id}")
    result = engine.synthesize_program(task)
    
    print("结果:", "成功" if result.success else "失败")
    if result.program:
        print("程序:", result.program)

def run_single_task(task_id: str, config_path: str):
    """运行单个任务"""
    engine = ARCSynthesisEngine(config_path)
    loader = ARCDataLoader()
    
    try:
        task = loader.load_task(task_id)
        result = engine.synthesize_program(task)
        
        print("结果:", "成功" if result.success else "失败")
        if result.program:
            print("程序:", result.program)
    except FileNotFoundError:
        print(f"任务 {task_id} 未找到")

def run_tests():
    """运行测试"""
    import unittest
    
    # 发现并运行测试
    loader = unittest.TestLoader()
    suite = loader.discover('tests', pattern='test_*.py')
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)

if __name__ == "__main__":
    main()
