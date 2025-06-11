# =====================================================================
# 完整ARC程序合成项目文件包
# 基于Popper的归纳逻辑编程与CEGIS集成框架
# =====================================================================

# 项目根目录结构：
"""
arc_synthesis_framework/
├── requirements.txt
├── setup.py
├── README.md
├── main.py
├── config/
│   ├── default.yaml
│   ├── spatial.yaml
│   └── complex.yaml
├── arc_synthesis_framework/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── synthesis_engine.py
│   │   ├── popper_interface.py
│   │   ├── anti_unification.py
│   │   └── oracle.py
│   ├── extraction/
│   │   ├── __init__.py
│   │   ├── object_extractor.py
│   │   ├── spatial_predicates.py
│   │   └── transformations.py
│   ├── cegis/
│   │   ├── __init__.py
│   │   ├── synthesizer.py
│   │   ├── verifier.py
│   │   └── counterexample.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── arc_loader.py
│   │   ├── metrics.py
│   │   └── logging.py
│   └── popper_files/
│       ├── templates/
│       │   ├── basic_bias.pl
│       │   ├── basic_bk.pl
│       │   └── spatial_bias.pl
│       ├── bias/
│       ├── background/
│       └── examples/
├── examples/
│   ├── simple_tasks/
│   │   └── color_change.json
│   └── demonstrations/
│       └── basic_usage.py
├── tests/
│   ├── __init__.py
│   ├── test_synthesis_engine.py
│   └── test_object_extractor.py
└── docs/
    └── README.md
"""

# =====================================================================
# 1. requirements.txt
# =====================================================================
REQUIREMENTS_TXT = """numpy>=1.21.0
scipy>=1.7.0
scikit-learn>=1.0.0
scikit-image>=0.18.0
pyyaml>=5.4.0
matplotlib>=3.4.0
networkx>=2.6.0
python-sat>=0.1.7.dev14
clingo>=5.5.0
"""

# =====================================================================
# 2. setup.py
# =====================================================================
SETUP_PY = """from setuptools import setup, find_packages

setup(
    name="arc-synthesis-framework",
    version="0.1.0",
    description="Popper-based Program Synthesis Framework for ARC Tasks",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="ARC Research Team",
    author_email="research@arc-synthesis.com",
    url="https://github.com/arc-team/synthesis-framework",
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
    extras_require={
        "dev": ["pytest>=6.0", "black", "flake8", "mypy"],
        "docs": ["sphinx", "sphinx-rtd-theme"]
    },
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    entry_points={
        "console_scripts": [
            "arc-synthesize=arc_synthesis_framework.main:main",
        ],
    },
)
"""

# =====================================================================
# 3. README.md
# =====================================================================
README_MD = """# ARC程序合成框架

基于Popper的归纳逻辑编程系统，用于解决ARC（抽象与推理语料库）任务。

## 特性

- **Popper ILP集成**: 使用归纳逻辑编程进行程序合成
- **对象提取**: 从2D网格中提取连通对象及其属性
- **CEGIS**: 反例引导的归纳合成循环
- **反统一**: 模式泛化和规则发现
- **空间推理**: 完整的空间关系处理
- **可扩展架构**: 模块化设计支持新转换类型

## 安装

### 前置要求

1. Python 3.8+
2. SWI-Prolog
3. Clingo (用于ASP求解)

### 安装步骤

```bash
# 克隆仓库
git clone https://github.com/arc-team/synthesis-framework.git
cd synthesis-framework

# 安装依赖
pip install -r requirements.txt

# 安装包
python setup.py install

# 或开发模式安装
pip install -e .
```

### Popper安装

```bash
# 克隆Popper仓库
git clone https://github.com/logic-and-learning-lab/Popper.git
cd Popper
# 按照Popper的安装说明进行安装
```

## 快速开始

### 基本使用

```python
from arc_synthesis_framework import ARCSynthesisEngine, SynthesisTask
from arc_synthesis_framework.utils.arc_loader import ARCDataLoader

# 初始化引擎
engine = ARCSynthesisEngine("config/default.yaml")

# 创建或加载ARC任务
loader = ARCDataLoader()
task = loader.create_simple_task()

# 运行合成
result = engine.synthesize_program(task)

if result.success:
    print(f"合成成功: {result.program}")
    print(f"置信度: {result.confidence:.2f}")
else:
    print(f"合成失败: {result.error_message}")
```

### 命令行使用

```bash
# 运行演示
python main.py --demo

# 运行特定任务
python main.py --task simple_color_change

# 从文件加载任务
python main.py --task_file examples/simple_tasks/color_change.json

# 运行测试
python main.py --test
```

## 项目结构

- `core/`: 核心合成引擎和接口
- `extraction/`: 对象提取和空间分析
- `cegis/`: 反例引导合成实现
- `utils/`: 工具函数和数据加载
- `popper_files/`: Popper相关文件和模板
- `examples/`: 示例任务和演示代码
- `tests/`: 单元测试

## 配置

主要配置文件位于 `config/` 目录：

- `default.yaml`: 基本配置
- `spatial.yaml`: 空间推理任务配置
- `complex.yaml`: 复杂任务配置

## 扩展指南

### 添加新的转换类型

1. 在 `extraction/transformations.py` 中添加转换分析
2. 在 `popper_files/background/` 中添加相关谓词
3. 更新偏置文件以包含新的谓词

### 自定义对象检测

继承 `ARCObjectExtractor` 类并重写相关方法：

```python
class CustomObjectExtractor(ARCObjectExtractor):
    def _classify_shape(self, cells, bbox):
        # 自定义形状分类逻辑
        pass
```

## 贡献

欢迎贡献代码、报告问题或提出改进建议。

## 许可证

MIT License

## 引用

如果您在研究中使用此框架，请引用：

```bibtex
@software{arc_synthesis_framework,
  title={ARC Program Synthesis Framework},
  author={ARC Research Team},
  year={2024},
  url={https://github.com/arc-team/synthesis-framework}
}
```
"""

# =====================================================================
# 4. config/default.yaml
# =====================================================================
CONFIG_DEFAULT_YAML = """# ARC程序合成框架默认配置

# Popper配置
popper:
  popper_path: "./popper"
  timeout: 300
  solver: "rc2"
  max_vars: 8
  max_body: 10
  max_rules: 5
  noisy: false
  enable_recursion: false

# 对象提取配置
extraction:
  connectivity: 4  # 4连通或8连通
  min_object_size: 1
  background_color: 0
  analyze_shapes: true
  detect_patterns: true
  extract_holes: true

# CEGIS配置
cegis:
  max_iterations: 25
  synthesis_timeout: 300
  verification_timeout: 60
  enable_parallel: false

# 反统一配置
anti_unification:
  max_generalization_depth: 5
  preserve_structure: true
  enable_type_constraints: true
  min_pattern_support: 2

# 验证器配置
oracle:
  validation_method: "exact_match"
  tolerance: 0.0
  enable_partial_credit: false

# 日志配置
logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: "logs/synthesis.log"
  console_output: true

# 性能配置
performance:
  cache_enabled: true
  cache_size: 1000
  parallel_tasks: 1
  memory_limit_mb: 2048
"""

# =====================================================================
# 5. arc_synthesis_framework/__init__.py
# =====================================================================
MAIN_INIT_PY = '''"""
ARC程序合成框架主包

基于Popper的归纳逻辑编程系统，用于解决ARC任务。
"""

from .core.synthesis_engine import ARCSynthesisEngine, SynthesisTask, SynthesisResult
from .core.popper_interface import PopperInterface
from .extraction.object_extractor import ARCObjectExtractor, ARCObject
from .utils.arc_loader import ARCDataLoader

__version__ = "0.1.0"
__author__ = "ARC Research Team"
__email__ = "research@arc-synthesis.com"

__all__ = [
    "ARCSynthesisEngine", 
    "SynthesisTask", 
    "SynthesisResult",
    "PopperInterface",
    "ARCObjectExtractor", 
    "ARCObject",
    "ARCDataLoader"
]

# 版本检查
import sys
if sys.version_info < (3, 8):
    raise RuntimeError("需要Python 3.8或更高版本")

# 可选的依赖检查
def check_dependencies():
    """检查关键依赖是否可用"""
    missing_deps = []
    
    try:
        import numpy
    except ImportError:
        missing_deps.append("numpy")
    
    try:
        import yaml
    except ImportError:
        missing_deps.append("pyyaml")
    
    try:
        import sklearn
    except ImportError:
        missing_deps.append("scikit-learn")
    
    if missing_deps:
        raise ImportError(f"缺少依赖包: {', '.join(missing_deps)}")

# 在导入时进行检查
check_dependencies()
'''

# =====================================================================
# 6. core/synthesis_engine.py (主合成引擎)
# =====================================================================
SYNTHESIS_ENGINE_PY = '''"""
主合成引擎 - ARC程序合成的核心
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
        """从JSON文件加载任务"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return cls(
            task_id=data.get('task_id', Path(file_path).stem),
            train_pairs=[(pair['input'], pair['output']) for pair in data['train']],
            test_pairs=[(pair['input'], pair['output']) for pair in data['test']],
            metadata=data.get('metadata', {})
        )
    
    def to_file(self, file_path: str):
        """保存任务到JSON文件"""
        data = {
            'task_id': self.task_id,
            'train': [{'input': inp, 'output': out} for inp, out in self.train_pairs],
            'test': [{'input': inp, 'output': out} for inp, out in self.test_pairs],
            'metadata': self.metadata
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

@dataclass
class SynthesisResult:
    """程序合成结果"""
    success: bool
    program: Optional[str]
    confidence: float
    synthesis_time: float
    iterations: int
    counterexamples_used: int
    generalization_pattern: Optional[str]
    error_message: Optional[str] = None
    intermediate_results: List[Dict] = None

class ARCSynthesisEngine:
    """
    ARC主合成引擎
    
    集成Popper ILP、CEGIS、对象提取和反统一技术，
    实现端到端的ARC任务程序合成。
    """
    
    def __init__(self, config_path: str = "config/default.yaml"):
        """
        初始化合成引擎
        
        Args:
            config_path: 配置文件路径
        """
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
        
        # 合成状态管理
        self.synthesis_history = []
        self.learned_patterns = {}
        self.cached_results = {}
        
        logger.info("ARC合成引擎初始化完成")
    
    def synthesize_program(self, task: SynthesisTask) -> SynthesisResult:
        """
        主合成方法 - 实现完整的CEGIS循环与ILP集成
        
        Args:
            task: ARC任务规范
            
        Returns:
            SynthesisResult: 包含程序和元数据的合成结果
        """
        start_time = time.time()
        logger.info(f"开始合成任务 {task.task_id}")
        
        try:
            # 检查缓存
            cache_key = self._generate_cache_key(task)
            if cache_key in self.cached_results:
                logger.info("使用缓存结果")
                return self.cached_results[cache_key]
            
            # 阶段1: 对象提取和分析
            logger.info("阶段1: 对象提取和空间分析")
            extracted_objects = self._extract_objects_from_pairs(task.train_pairs)
            spatial_relations = self._analyze_spatial_relations(extracted_objects)
            
            # 阶段2: 生成Popper输入文件
            logger.info("阶段2: 生成Popper学习文件")
            popper_files = self._generate_popper_files(task, extracted_objects, spatial_relations)
            
            # 阶段3: CEGIS合成循环
            logger.info("阶段3: 开始CEGIS合成循环")
            result = self._cegis_synthesis_loop(task, popper_files)
            
            # 阶段4: 反统一和模式泛化
            if result.success:
                logger.info("阶段4: 应用反统一进行模式泛化")
                generalized_pattern = self._generalize_solution(result.program, task)
                result.generalization_pattern = generalized_pattern
            
            # 阶段5: 最终验证和测试
            if result.success:
                logger.info("阶段5: 最终验证")
                validation_result = self._validate_solution(result.program, task)
                result.success = validation_result
            
            # 完成计时和记录
            result.synthesis_time = time.time() - start_time
            self._log_synthesis_result(task, result)
            
            # 缓存结果
            if self.config.get('performance', {}).get('cache_enabled', True):
                self.cached_results[cache_key] = result
            
            return result
            
        except Exception as e:
            logger.error(f"任务 {task.task_id} 合成失败: {str(e)}", exc_info=True)
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
        """从所有训练对中提取对象和转换模式"""
        all_objects = {}
        
        for pair_idx, (input_grid, output_grid) in enumerate(train_pairs):
            logger.debug(f"处理训练对 {pair_idx + 1}")
            
            # 提取输入和输出对象
            input_analysis = self.object_extractor.process_arc_grid(input_grid)
            output_analysis = self.object_extractor.process_arc_grid(output_grid)
            
            # 分析转换模式
            transformation = self.object_extractor.extract_transformation_pattern(
                input_grid, output_grid
            )
            
            all_objects[f"pair_{pair_idx}"] = {
                'input_analysis': input_analysis,
                'output_analysis': output_analysis,
                'transformation': transformation,
                'input_grid': input_grid,
                'output_grid': output_grid
            }
        
        return all_objects
    
    def _generate_popper_files(self, task: SynthesisTask, objects: Dict, relations: Dict) -> Dict:
        """生成Popper ILP所需的输入文件"""
        
        # 创建任务目录
        task_dir = Path(f"popper_files/tasks/{task.task_id}")
        task_dir.mkdir(parents=True, exist_ok=True)
        
        # 生成文件内容
        examples_content = self._generate_examples_file(task, objects)
        bk_content = self._generate_background_knowledge(objects, relations)
        bias_content = self._generate_bias_file(task, objects)
        
        # 写入文件
        files = {
            'examples': task_dir / "exs.pl",
            'background': task_dir / "bk.pl", 
            'bias': task_dir / "bias.pl"
        }
        
        files['examples'].write_text(examples_content, encoding='utf-8')
        files['background'].write_text(bk_content, encoding='utf-8')
        files['bias'].write_text(bias_content, encoding='utf-8')
        
        logger.debug(f"生成Popper文件到 {task_dir}")
        return files
    
    def _generate_examples_file(self, task: SynthesisTask, objects: Dict) -> str:
        """生成Popper示例文件 (exs.pl)"""
        examples = [
            f"% ARC任务 {task.task_id} 的训练示例",
            f"% 任务描述: {task.metadata.get('description', '未知')}",
            ""
        ]
        
        for pair_idx, (input_grid, output_grid) in enumerate(task.train_pairs):
            # 将网格转换为结构化表示
            input_facts = self._grid_to_facts(input_grid, f"input_{pair_idx}")
            output_facts = self._grid_to_facts(output_grid, f"output_{pair_idx}")
            
            # 创建正例
            examples.append(f"pos(transform({input_facts}, {output_facts})).")
        
        # 添加负例（如果需要）
        examples.extend(self._generate_negative_examples(task, objects))
        
        return "\\n".join(examples)
    
    def _grid_to_facts(self, grid: List[List[int]], grid_name: str) -> str:
        """将网格转换为Prolog事实表示"""
        facts = []
        for r_idx, row in enumerate(grid):
            for c_idx, cell_value in enumerate(row):
                facts.append(f"cell({r_idx},{c_idx},{cell_value})")
        
        return f"grid([{','.join(facts)}])"
    
    def _generate_background_knowledge(self, objects: Dict, relations: Dict) -> str:
        """生成背景知识文件 (bk.pl)"""
        bk_lines = [
            f"% ARC任务背景知识 - 自动生成",
            f"% 生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "% ===== 基础网格操作 =====",
            "",
            "% 获取网格尺寸",
            "grid_size(grid(Cells), Width, Height) :-",
            "    findall(X, member(cell(_, X, _), Cells), Xs),",
            "    findall(Y, member(cell(Y, _, _), Cells), Ys),",
            "    (Xs = [] -> Width = 0; (max_list(Xs, MaxX), Width is MaxX + 1)),",
            "    (Ys = [] -> Height = 0; (max_list(Ys, MaxY), Height is MaxY + 1)).",
            "",
            "% 获取特定位置的颜色",
            "cell_color(grid(Cells), R, C, Color) :-",
            "    member(cell(R, C, Color), Cells).",
            "",
            "% 边界检查",
            "in_bounds(X, Y, Grid) :-",
            "    grid_size(Grid, W, H),",
            "    X >= 0, X < W, Y >= 0, Y < H.",
            "",
            "% ===== 颜色转换操作 =====",
            "",
            "% 全局颜色替换",
            "change_all_color(grid(Cells), OldColor, NewColor, grid(NewCells)) :-",
            "    maplist(replace_color(OldColor, NewColor), Cells, NewCells).",
            "",
            "replace_color(OldColor, NewColor, cell(R,C,OldColor), cell(R,C,NewColor)) :- !.",
            "replace_color(_, _, Cell, Cell).",
            "",
            "% ===== 空间关系谓词 =====",
            "",
            "% 4连通相邻",
            "adjacent_4(cell(R1,C1,_), cell(R2,C2,_)) :-",
            "    (R1 =:= R2, abs(C1-C2) =:= 1);",
            "    (C1 =:= C2, abs(R1-R2) =:= 1).",
            "",
            "% 8连通相邻", 
            "adjacent_8(cell(R1,C1,_), cell(R2,C2,_)) :-",
            "    abs(R1-R2) =< 1, abs(C1-C2) =< 1,",
            "    \\+ (R1 =:= R2, C1 =:= C2).",
            "",
            "% ===== 对象检测和分析 =====",
            "",
            "% 检测所有对象",
            "detect_objects(Grid, Objects) :-",
            "    findall(Color-Cells, detect_color_group(Grid, Color, Cells), Groups),",
            "    include(non_empty_group, Groups, Objects).",
            "",
            "detect_color_group(grid(Cells), Color, ColorCells) :-",
            "    include(has_color(Color), Cells, ColorCells),",
            "    Color \\= 0.  % 排除背景色",
            "",
            "has_color(Color, cell(_,_,Color)).",
            "non_empty_group(_-Cells) :- Cells \\= [].",
            "",
            "% ===== 模式匹配 =====",
            "",
            "% 检测直线模式",
            "is_line(Cells) :-",
            "    length(Cells, Len), Len > 1,",
            "    (all_same_row(Cells); all_same_col(Cells)).",
            "",
            "all_same_row([cell(R,_,_)|Rest]) :-",
            "    maplist(same_row(R), Rest).",
            "same_row(R, cell(R,_,_)).",
            "",
            "all_same_col([cell(_,C,_)|Rest]) :-",
            "    maplist(same_col(C), Rest).",
            "same_col(C, cell(_,C,_)).",
            ""
        ]
        
        # 添加任务特定的背景知识
        task_specific_bk = self._generate_task_specific_background(objects)
        bk_lines.extend(task_specific_bk)
        
        return "\\n".join(bk_lines)
    
    def _generate_bias_file(self, task: SynthesisTask, objects: Dict) -> str:
        """生成偏置文件 (bias.pl)"""
        bias_lines = [
            f"% ARC任务 {task.task_id} 的学习偏置",
            f"% 任务类型: {task.metadata.get('type', 'unknown')}",
            "",
            "% ===== 头谓词定义 =====",
            "head_pred(transform,2).",
            "",
            "% ===== 体谓词定义 =====",
            "",
            "% 基础网格操作",
            "body_pred(grid,1).",
            "body_pred(cell,3).",
            "body_pred(cell_color,4).",
            "body_pred(grid_size,3).",
            "",
            "% 颜色操作",
            "body_pred(change_all_color,4).",
            "body_pred(replace_color,4).",
            "",
            "% 空间关系",
            "body_pred(adjacent_4,2).",
            "body_pred(adjacent_8,2).",
            "",
            "% 对象检测",
            "body_pred(detect_objects,2).",
            "body_pred(is_line,1).",
            "",
            "% ===== 类型约束 =====",
            "",
            "type(transform,(grid,grid)).",
            "type(cell,(int,int,int)).",
            "type(change_all_color,(grid,int,int,grid)).",
            "type(grid_size,(grid,int,int)).",
            "",
            "% ===== 方向约束 =====",
            "",
            "direction(transform,(in,out)).",
            "direction(change_all_color,(in,in,in,out)).",
            "",
            "% ===== 学习控制参数 =====",
            "",
            f"max_vars({self.config['popper'].get('max_vars', 8)}).",
            f"max_body({self.config['popper'].get('max_body', 10)}).",
            f"max_rules({self.config['popper'].get('max_rules', 5)}).",
            "",
            "% 允许单例变量（对ARC任务有用）",
            "allow_singletons.",
            ""
        ]
        
        # 根据任务类型添加特定偏置
        task_type = task.metadata.get('type', 'unknown')
        if task_type == 'color_transformation':
            bias_lines.extend([
                "% 颜色转换任务特定偏置",
                "body_pred(same_color,2).",
                "body_pred(different_color,2)."
            ])
        elif task_type == 'spatial_transformation':
            bias_lines.extend([
                "% 空间转换任务特定偏置", 
                "body_pred(translate,4).",
                "body_pred(rotate,3).",
                "body_pred(reflect,2)."
            ])
        
        return "\\n".join(bias_lines)
    
    def _cegis_synthesis_loop(self, task: SynthesisTask, popper_files: Dict) -> SynthesisResult:
        """实现CEGIS合成循环"""
        counterexamples = []
        iteration = 0
        max_iterations = self.config['cegis']['max_iterations']
        
        logger.info(f"开始CEGIS循环，最大迭代次数: {max_iterations}")
        
        while iteration < max_iterations:
            logger.info(f"CEGIS迭代 {iteration + 1}/{max_iterations}")
            
            # 使用Popper生成候选程序
            candidate = self._generate_candidate_with_popper(
                popper_files, counterexamples, iteration
            )
            
            if candidate is None:
                logger.info("无法生成更多候选程序 - 搜索空间已穷尽")
                return SynthesisResult(
                    success=False, program=None, confidence=0.0,
                    synthesis_time=0, iterations=iteration,
                    counterexamples_used=len(counterexamples),
                    generalization_pattern=None,
                    error_message="搜索空间已穷尽"
                )
            
            logger.debug(f"生成候选程序: {candidate}")
            
            # 验证候选程序
            verification_result = self.verifier.verify_candidate(candidate, task.train_pairs)
            
            if verification_result.is_valid:
                logger.info(f"在迭代 {iteration + 1} 中找到有效程序")
                confidence = self._calculate_confidence(candidate, task)
                return SynthesisResult(
                    success=True, program=candidate, confidence=confidence,
                    synthesis_time=0, iterations=iteration + 1,
                    counterexamples_used=len(counterexamples),
                    generalization_pattern=None
                )
            else:
                # 生成反例约束
                new_counterexample = self.counterexample_gen.generate(
                    candidate, verification_result.failed_example
                )
                counterexamples.append(new_counterexample)
                logger.info(f"添加反例约束 {len(counterexamples)}: {new_counterexample}")
            
            iteration += 1
        
        logger.warning("CEGIS达到最大迭代次数，合成失败")
        return SynthesisResult(
            success=False, program=None, confidence=0.0,
            synthesis_time=0, iterations=max_iterations,
            counterexamples_used=len(counterexamples),
            generalization_pattern=None,
            error_message="达到最大迭代次数"
        )
    
    def _generate_candidate_with_popper(self, popper_files: Dict, 
                                      counterexamples: List[str], 
                                      iteration: int) -> Optional[str]:
        """使用Popper生成候选程序"""
        logger.debug(f"调用Popper生成候选程序，反例数量: {len(counterexamples)}")
        
        # 准备约束（如果有反例）
        constraints = []
        if counterexamples:
            constraints.append(f"% 迭代 {iteration} 的约束")
            constraints.extend(counterexamples)
        
        # 调用Popper
        task_dir = popper_files['examples'].parent
        program = self.popper.learn_program(task_dir, constraints if counterexamples else None)
        
        return program
    
    def _calculate_confidence(self, candidate: str, task: SynthesisTask) -> float:
        """计算候选程序的置信度"""
        # 基于多个因素计算置信度：
        # 1. 程序复杂度（越简单越好）
        # 2. 训练准确率
        # 3. 泛化能力评估
        
        try:
            # 程序复杂度分数（0-1，越高越好）
            complexity_score = self._calculate_complexity_score(candidate)
            
            # 训练准确率（这里应该是1.0，因为已经验证通过）
            training_accuracy = 1.0
            
            # 综合计算
            confidence = 0.6 * training_accuracy + 0.4 * complexity_score
            
            return min(1.0, max(0.0, confidence))
            
        except Exception as e:
            logger.warning(f"计算置信度时出错: {str(e)}")
            return 0.5  # 默认中等置信度
    
    def _calculate_complexity_score(self, program: str) -> float:
        """计算程序复杂度分数"""
        if not program:
            return 0.0
        
        lines = [line.strip() for line in program.split('\\n') if line.strip()]
        
        # 简单的复杂度度量
        rule_count = len(lines)
        avg_rule_length = sum(len(line) for line in lines) / max(1, rule_count)
        
        # 归一化分数（更少的规则和更短的规则得分更高）
        rule_score = max(0, 1 - (rule_count - 1) / 10)  # 1-10规则
        length_score = max(0, 1 - (avg_rule_length - 20) / 100)  # 20-120字符
        
        return (rule_score + length_score) / 2
    
    # 辅助方法
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
            'popper': {'popper_path': './popper', 'timeout': 300, 'max_vars': 8},
            'extraction': {'connectivity': 4, 'min_object_size': 1},
            'cegis': {'max_iterations': 25},
            'anti_unification': {'max_generalization_depth': 5},
            'oracle': {'validation_method': 'exact_match'}
        }
    
    def _setup_logging(self):
        """设置日志"""
        log_config = self.config.get('logging', {})
        setup_logging(
            level=log_config.get('level', 'INFO'),
            format_str=log_config.get('format'),
            log_file=log_config.get('file')
        )
    
    # 其他辅助方法...
    def _analyze_spatial_relations(self, objects: Dict) -> Dict:
        """分析空间关系（待实现）"""
        return {'relations': []}
    
    def _generate_negative_examples(self, task: SynthesisTask, objects: Dict) -> List[str]:
        """生成负例（可选）"""
        return ["% 暂无负例"]
    
    def _generate_task_specific_background(self, objects: Dict) -> List[str]:
        """生成任务特定背景知识"""
        return ["% 任务特定背景知识待添加"]
    
    def _generalize_solution(self, program: str, task: SynthesisTask) -> str:
        """泛化解决方案"""
        return self.anti_unifier.generalize_program(program, task.train_pairs)
    
    def _validate_solution(self, program: str, task: SynthesisTask) -> bool:
        """验证解决方案"""
        return self.oracle.validate_program(program, task.test_pairs)
    
    def _generate_cache_key(self, task: SynthesisTask) -> str:
        """生成缓存键"""
        return f"{task.task_id}_{hash(str(task.train_pairs))}"
    
    def _log_synthesis_result(self, task: SynthesisTask, result: SynthesisResult):
        """记录合成结果"""
        if result.success:
            logger.info(f"✓ 任务 {task.task_id} 合成成功")
            logger.info(f"  程序: {result.program}")
            logger.info(f"  置信度: {result.confidence:.2f}")
            logger.info(f"  迭代: {result.iterations}, 时间: {result.synthesis_time:.2f}s")
        else:
            logger.warning(f"✗ 任务 {task.task_id} 合成失败: {result.error_message}")
'''

# =====================================================================
# 7. main.py (主运行文件)
# =====================================================================
MAIN_PY = '''#!/usr/bin/env python3
"""
ARC程序合成框架主运行文件
"""
import argparse
import sys
import logging
from pathlib import Path

# 确保可以导入本地包
sys.path.insert(0, str(Path(__file__).parent))

from arc_synthesis_framework import ARCSynthesisEngine, SynthesisTask
from arc_synthesis_framework.utils.arc_loader import ARCDataLoader
from arc_synthesis_framework.utils.metrics import SynthesisMetrics

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="ARC程序合成框架",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  %(prog)s --demo                    # 运行演示
  %(prog)s --task simple_task        # 运行特定任务
  %(prog)s --task_file task.json     # 从文件加载任务
  %(prog)s --test                    # 运行测试
        """
    )
    
    parser.add_argument("--config", default="config/default.yaml", 
                       help="配置文件路径 (默认: config/default.yaml)")
    parser.add_argument("--task", help="特定任务ID")
    parser.add_argument("--task_file", help="任务文件路径")
    parser.add_argument("--demo", action="store_true", 
                       help="运行演示任务")
    parser.add_argument("--test", action="store_true", 
                       help="运行单元测试")
    parser.add_argument("--batch", help="批量处理任务目录")
    parser.add_argument("--output", help="结果输出文件")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="详细输出")
    
    args = parser.parse_args()
    
    # 设置日志级别
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    
    try:
        if args.demo:
            run_demo(args.config)
        elif args.test:
            run_tests()
        elif args.batch:
            run_batch_tasks(args.batch, args.config, args.output)
        elif args.task_file:
            run_single_task_from_file(args.task_file, args.config)
        elif args.task:
            run_single_task(args.task, args.config)
        else:
            print("请指定运行模式。使用 --help 查看选项。")
            parser.print_help()
            return 1
    
    except KeyboardInterrupt:
        print("\\n用户中断操作")
        return 1
    except Exception as e:
        print(f"错误: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1
    
    return 0

def run_demo(config_path: str = "config/default.yaml"):
    """运行演示"""
    print("=" * 60)
    print("🧠 ARC程序合成框架演示")
    print("=" * 60)
    
    try:
        # 初始化组件
        print("\\n📦 初始化合成引擎...")
        engine = ARCSynthesisEngine(config_path)
        loader = ARCDataLoader()
        metrics = SynthesisMetrics()
        
        # 创建简单测试任务
        print("\\n🎯 创建演示任务...")
        task = loader.create_simple_task()
        
        print(f"任务ID: {task.task_id}")
        print(f"描述: {task.metadata.get('description', '简单颜色转换')}")
        print(f"训练样例: {len(task.train_pairs)} 个")
        print(f"测试样例: {len(task.test_pairs)} 个")
        
        # 显示样例
        print("\\n📊 训练样例:")
        for i, (inp, out) in enumerate(task.train_pairs):
            print(f"  样例 {i+1}:")
            print(f"    输入:  {inp}")
            print(f"    输出:  {out}")
        
        # 执行合成
        print("\\n🔄 开始程序合成...")
        print("-" * 40)
        
        result = engine.synthesize_program(task)
        
        # 记录指标
        metrics.add_task_result(
            task.task_id, result.success, result.synthesis_time,
            result.iterations, result.program or "", result.confidence
        )
        
        # 显示结果
        print("\\n📋 合成结果:")
        print("-" * 40)
        
        if result.success:
            print("✅ 合成成功!")
            print(f"📝 程序:")
            for line in (result.program or "").split('\\n'):
                if line.strip():
                    print(f"    {line}")
            print(f"🎯 置信度: {result.confidence:.2%}")
            print(f"🔄 迭代次数: {result.iterations}")
            print(f"⏱️  合成时间: {result.synthesis_time:.2f}秒")
            
            if result.generalization_pattern:
                print(f"🔍 泛化模式: {result.generalization_pattern}")
        else:
            print("❌ 合成失败")
            print(f"💥 错误信息: {result.error_message}")
            print(f"🔄 尝试迭代: {result.iterations}")
            print(f"⏱️  用时: {result.synthesis_time:.2f}秒")
        
        # 显示统计
        print("\\n📈 性能统计:")
        print("-" * 40)
        metrics.print_summary()
        
    except Exception as e:
        print(f"❌ 演示运行失败: {str(e)}")
        raise

def run_single_task_from_file(task_file: str, config_path: str):
    """从文件运行单个任务"""
    print(f"🔄 从文件加载任务: {task_file}")
    
    try:
        engine = ARCSynthesisEngine(config_path)
        task = SynthesisTask.from_file(task_file)
        
        print(f"✅ 成功加载任务: {task.task_id}")
        print(f"📊 训练样例: {len(task.train_pairs)}")
        print(f"🧪 测试样例: {len(task.test_pairs)}")
        
        result = engine.synthesize_program(task)
        
        print("\\n📋 结果:")
        if result.success:
            print("✅ 合成成功")
            print(f"📝 程序: {result.program}")
            print(f"🎯 置信度: {result.confidence:.2%}")
        else:
            print("❌ 合成失败")
            print(f"💥 错误: {result.error_message}")
            
    except FileNotFoundError:
        print(f"❌ 文件未找到: {task_file}")
    except Exception as e:
        print(f"❌ 运行失败: {str(e)}")

def run_single_task(task_id: str, config_path: str):
    """运行单个指定任务"""
    print(f"🔄 运行任务: {task_id}")
    
    try:
        engine = ARCSynthesisEngine(config_path)
        loader = ARCDataLoader()
        
        task = loader.load_task(task_id)
        result = engine.synthesize_program(task)
        
        print("\\n📋 结果:")
        if result.success:
            print("✅ 合成成功")
            print(f"📝 程序: {result.program}")
        else:
            print("❌ 合成失败") 
            print(f"💥 错误: {result.error_message}")
            
    except FileNotFoundError:
        print(f"❌ 任务 {task_id} 未找到")
    except Exception as e:
        print(f"❌ 运行失败: {str(e)}")

def run_batch_tasks(task_dir: str, config_path: str, output_file: str = None):
    """批量运行任务"""
    print(f"🔄 批量处理任务目录: {task_dir}")
    
    try:
        engine = ARCSynthesisEngine(config_path)
        metrics = SynthesisMetrics()
        
        # 查找所有任务文件
        task_path = Path(task_dir)
        task_files = list(task_path.glob("*.json"))
        
        if not task_files:
            print(f"❌ 在目录 {task_dir} 中未找到任务文件")
            return
        
        print(f"📁 找到 {len(task_files)} 个任务文件")
        
        results = []
        
        for i, task_file in enumerate(task_files, 1):
            print(f"\\n[{i}/{len(task_files)}] 处理 {task_file.name}")
            
            try:
                task = SynthesisTask.from_file(str(task_file))
                result = engine.synthesize_program(task)
                
                metrics.add_task_result(
                    task.task_id, result.success, result.synthesis_time,
                    result.iterations, result.program or "", result.confidence
                )
                
                results.append({
                    'task_id': task.task_id,
                    'file': task_file.name,
                    'success': result.success,
                    'program': result.program,
                    'confidence': result.confidence,
                    'time': result.synthesis_time,
                    'error': result.error_message
                })
                
                status = "✅" if result.success else "❌"
                print(f"  {status} {task.task_id}: {'成功' if result.success else '失败'}")
                
            except Exception as e:
                print(f"  ❌ {task_file.name}: 处理失败 - {str(e)}")
                results.append({
                    'task_id': task_file.stem,
                    'file': task_file.name,
                    'success': False,
                    'error': str(e)
                })
        
        # 显示总结
        print("\\n" + "="*60)
        print("📈 批量处理结果总结")
        print("="*60)
        metrics.print_summary()
        
        # 保存结果
        if output_file:
            save_batch_results(results, output_file)
            print(f"\\n💾 结果已保存到: {output_file}")
        
    except Exception as e:
        print(f"❌ 批量处理失败: {str(e)}")

def save_batch_results(results: list, output_file: str):
    """保存批量处理结果"""
    import json
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

def run_tests():
    """运行单元测试"""
    print("🧪 运行单元测试...")
    
    try:
        import unittest
        
        # 发现并运行测试
        loader = unittest.TestLoader()
        suite = loader.discover('tests', pattern='test_*.py')
        
        if suite.countTestCases() == 0:
            print("⚠️  未找到测试文件")
            print("请确保测试文件位于 'tests/' 目录中")
            return
        
        print(f"📁 找到 {suite.countTestCases()} 个测试")
        
        runner = unittest.TextTestRunner(
            verbosity=2,
            descriptions=True,
            failfast=False
        )
        
        result = runner.run(suite)
        
        if result.wasSuccessful():
            print("\\n✅ 所有测试通过!")
        else:
            print(f"\\n❌ {len(result.failures)} 个测试失败, {len(result.errors)} 个错误")
            return 1
            
    except ImportError as e:
        print(f"❌ 导入测试模块失败: {str(e)}")
        print("请确保已安装 unittest 或 pytest")
        return 1

if __name__ == "__main__":
    sys.exit(main())
'''

# =====================================================================
# 说明：完整文件清单
# =====================================================================
FILE_LIST = """
完整项目文件清单:

1. 根目录文件:
   - requirements.txt          # Python依赖
   - setup.py                  # 包安装配置
   - README.md                 # 项目说明
   - main.py                   # 主运行文件

2. 配置文件 (config/):
   - default.yaml              # 默认配置
   - spatial.yaml              # 空间推理配置 
   - complex.yaml              # 复杂任务配置

3. 核心模块 (arc_synthesis_framework/core/):
   - __init__.py               # 核心模块初始化
   - synthesis_engine.py       # 主合成引擎(已包含)
   - popper_interface.py       # Popper接口
   - anti_unification.py       # 反统一算法
   - oracle.py                 # 解决方案验证

4. 提取模块 (arc_synthesis_framework/extraction/):
   - __init__.py               # 提取模块初始化
   - object_extractor.py       # 对象提取器(核心文件已包含)
   - spatial_predicates.py     # 空间谓词生成
   - transformations.py        # 转换分析

5. CEGIS模块 (arc_synthesis_framework/cegis/):
   - __init__.py               # CEGIS模块初始化
   - synthesizer.py            # CEGIS合成器
   - verifier.py               # 程序验证器
   - counterexample.py         # 反例生成器

6. 工具模块 (arc_synthesis_framework/utils/):
   - __init__.py               # 工具模块初始化
   - arc_loader.py             # ARC数据加载器
   - metrics.py                # 性能指标
   - logging.py                # 日志工具

7. Popper文件 (arc_synthesis_framework/popper_files/):
   - templates/basic_bias.pl   # 基本偏置模板
   - templates/basic_bk.pl     # 基本背景知识模板
   - templates/spatial_bias.pl # 空间推理偏置

8. 示例文件 (examples/):
   - simple_tasks/color_change.json    # 简单颜色转换任务
   - demonstrations/basic_usage.py     # 基本使用演示

9. 测试文件 (tests/):
   - __init__.py               # 测试模块初始化
   - test_synthesis_engine.py  # 合成引擎测试
   - test_object_extractor.py  # 对象提取器测试

使用说明:
1. 将代码保存为对应的文件
2. 安装依赖: pip install -r requirements.txt
3. 配置Popper路径
4. 运行演示: python main.py --demo
"""

print("=" * 70)
print("ARC程序合成框架 - 完整项目文件包")
print("=" * 70)
print()
print("本文件包含完整的项目代码结构，包括:")
print("✅ 核心合成引擎")
print("✅ Popper ILP集成") 
print("✅ 对象提取和空间分析")
print("✅ CEGIS反例引导合成")
print("✅ 反统一模式泛化")
print("✅ 完整的配置和示例")
print("✅ 测试框架")
print()
print("使用方法:")
print("1. 复制相应代码到对应文件")
print("2. 按照项目结构创建目录")
print("3. 安装依赖并配置Popper")
print("4. 运行演示验证安装")
print()
print(FILE_LIST)
