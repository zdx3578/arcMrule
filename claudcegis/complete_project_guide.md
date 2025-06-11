# ARC程序合成框架 - 完整使用指南

## 🎯 项目概述

这是一个基于Popper归纳逻辑编程的ARC（抽象与推理语料库）任务程序合成框架。系统集成了对象提取、CEGIS反例引导合成、反统一模式泛化等先进技术，能够从少量示例中学习空间推理规则。

## 📁 完整项目结构

```
arc_synthesis_framework/
├── 📄 requirements.txt          # Python依赖包
├── 📄 setup.py                  # 包安装配置
├── 📄 README.md                 # 项目说明文档
├── 📄 main.py                   # 主运行文件
├── 📁 config/                   # 配置文件目录
│   ├── 📄 default.yaml          # 默认配置
│   ├── 📄 spatial.yaml          # 空间推理配置
│   └── 📄 complex.yaml          # 复杂任务配置
├── 📁 arc_synthesis_framework/  # 主包目录
│   ├── 📄 __init__.py           # 包初始化
│   ├── 📁 core/                 # 核心模块
│   │   ├── 📄 __init__.py
│   │   ├── 📄 synthesis_engine.py      # 主合成引擎 ⭐
│   │   ├── 📄 popper_interface.py      # Popper接口 ⭐
│   │   ├── 📄 anti_unification.py      # 反统一算法 ⭐
│   │   └── 📄 oracle.py               # 解决方案验证
│   ├── 📁 extraction/           # 对象提取模块
│   │   ├── 📄 __init__.py
│   │   ├── 📄 object_extractor.py      # 对象提取器 ⭐
│   │   ├── 📄 spatial_predicates.py    # 空间谓词生成
│   │   └── 📄 transformations.py       # 转换分析
│   ├── 📁 cegis/                # CEGIS模块
│   │   ├── 📄 __init__.py
│   │   ├── 📄 synthesizer.py           # CEGIS合成器 ⭐
│   │   ├── 📄 verifier.py              # 程序验证器 ⭐
│   │   └── 📄 counterexample.py        # 反例生成器 ⭐
│   ├── 📁 utils/                # 工具模块
│   │   ├── 📄 __init__.py
│   │   ├── 📄 arc_loader.py            # ARC数据加载器 ⭐
│   │   ├── 📄 metrics.py               # 性能指标 ⭐
│   │   └── 📄 logging.py               # 日志工具 ⭐
│   └── 📁 popper_files/         # Popper相关文件
│       ├── 📁 templates/        # 模板文件
│       │   ├── 📄 basic_bias.pl        # 基本偏置模板
│       │   ├── 📄 basic_bk.pl          # 基本背景知识模板
│       │   └── 📄 spatial_bias.pl      # 空间推理偏置 ⭐
│       ├── 📁 bias/             # 偏置文件目录
│       ├── 📁 background/       # 背景知识目录
│       └── 📁 examples/         # 生成的示例
├── 📁 examples/                 # 示例和演示
│   ├── 📁 simple_tasks/
│   │   └── 📄 color_change.json        # 颜色转换示例 ⭐
│   └── 📁 demonstrations/
│       └── 📄 basic_usage.py           # 基本使用演示
├── 📁 tests/                    # 测试目录
│   ├── 📄 __init__.py
│   ├── 📄 test_synthesis_engine.py     # 合成引擎测试
│   └── 📄 test_object_extractor.py     # 对象提取器测试
├── 📁 logs/                     # 日志目录（运行时创建）
├── 📁 data/                     # 数据目录（可选）
│   └── 📁 arc/                  # ARC数据集
└── 📁 docs/                     # 文档目录
    └── 📄 README.md             # 详细文档
```

⭐ 标记的文件是核心实现文件

## 🚀 快速开始

### 1. 环境准备

```bash
# 创建Python虚拟环境
python -m venv arc_env
source arc_env/bin/activate  # Linux/Mac
# 或
arc_env\Scripts\activate     # Windows

# 升级pip
pip install --upgrade pip
```

### 2. 安装依赖

```bash
# 安装Python依赖
pip install -r requirements.txt

# 安装项目包
pip install -e .
```

### 3. 安装Popper

```bash
# 克隆Popper仓库
git clone https://github.com/logic-and-learning-lab/Popper.git
cd Popper

# 按照Popper的安装说明进行安装
# 通常需要SWI-Prolog和Clingo
```

### 4. 配置路径

在 `config/default.yaml` 中更新Popper路径：

```yaml
popper:
  popper_path: "/path/to/Popper"  # 更新为实际路径
```

### 5. 运行演示

```bash
# 运行基本演示
python main.py --demo

# 运行特定任务
python main.py --task_file examples/simple_tasks/color_change.json

# 查看帮助
python main.py --help
```

## 💡 核心功能使用

### 基本程序合成

```python
from arc_synthesis_framework import ARCSynthesisEngine, SynthesisTask
from arc_synthesis_framework.utils.arc_loader import ARCDataLoader

# 初始化引擎
engine = ARCSynthesisEngine("config/default.yaml")

# 加载任务
loader = ARCDataLoader()
task = loader.create_simple_task()

# 运行合成
result = engine.synthesize_program(task)

if result.success:
    print(f"合成成功！程序: {result.program}")
else:
    print(f"合成失败: {result.error_message}")
```

### 空间推理任务

```python
# 使用空间推理配置
engine = ARCSynthesisEngine("config/spatial.yaml")

# 创建空间任务
task = loader.create_spatial_task()

# 合成空间推理程序
result = engine.synthesize_program(task)
```

### 自定义任务创建

```python
# 创建自定义任务
custom_task = SynthesisTask(
    task_id="my_task",
    train_pairs=[
        (input_grid_1, output_grid_1),
        (input_grid_2, output_grid_2),
        # 更多训练对...
    ],
    test_pairs=[
        (test_input, expected_output)
    ],
    metadata={
        "description": "我的自定义任务",
        "type": "color_transformation"
    }
)

result = engine.synthesize_program(custom_task)
```

## ⚙️ 配置说明

### 默认配置 (config/default.yaml)

适用于大多数基本ARC任务：
- 中等复杂度的Popper设置
- 4连通对象提取
- 标准CEGIS参数

### 空间推理配置 (config/spatial.yaml)

专门用于空间推理任务：
- 8连通对象提取
- 增强的空间分析功能
- 更大的搜索空间

### 复杂任务配置 (config/complex.yaml)

用于最复杂的ARC任务：
- 最大的搜索空间
- 启用所有高级功能
- 详细的调试信息

## 🔧 扩展指南

### 添加新的转换类型

1. 在 `extraction/transformations.py` 中添加转换分析
2. 在 `popper_files/background/` 中添加相关谓词
3. 更新偏置文件以包含新谓词

### 自定义对象检测

```python
from arc_synthesis_framework.extraction.object_extractor import ARCObjectExtractor

class MyObjectExtractor(ARCObjectExtractor):
    def _classify_shape(self, cells, bbox):
        # 自定义形状分类逻辑
        return "my_custom_shape"
```

### 添加新的合成策略

```python
from arc_synthesis_framework.cegis.synthesizer import CEGISSynthesizer

class MyCustomSynthesizer(CEGISSynthesizer):
    def _generate_with_strategy(self, examples, bk, bias, strategy):
        if strategy == "my_strategy":
            return self._my_custom_generation(examples)
        return super()._generate_with_strategy(examples, bk, bias, strategy)
```

## 📊 性能监控

### 使用内置指标

```python
from arc_synthesis_framework.utils.metrics import SynthesisMetrics

metrics = SynthesisMetrics()

# 运行多个任务
for task in tasks:
    result = engine.synthesize_program(task)
    metrics.add_task_result_simple(
        task.task_id, result.success, result.synthesis_time,
        result.iterations, result.program, result.confidence
    )

# 显示统计
metrics.print_summary()

# 保存到文件
metrics.save_to_file("performance_report.json")
```

### 启用详细日志

```python
# 在配置文件中设置
logging:
  level: "DEBUG"
  file: "logs/detailed.log"
  console_output: true
```

## 🧪 测试

### 运行单元测试

```bash
# 运行所有测试
python main.py --test

# 或使用pytest（如果安装）
pytest tests/

# 运行特定测试
python -m unittest tests.test_synthesis_engine
```

### 创建自定义测试

```python
import unittest
from arc_synthesis_framework import ARCSynthesisEngine

class TestMyFeature(unittest.TestCase):
    def setUp(self):
        self.engine = ARCSynthesisEngine()
    
    def test_my_functionality(self):
        # 你的测试代码
        pass
```

## 🐛 故障排除

### 常见问题

1. **Popper未找到**
   - 检查Popper路径配置
   - 确保SWI-Prolog已安装

2. **内存不足**
   - 增加配置中的内存限制
   - 减少最大变量数和规则数

3. **合成超时**
   - 增加超时时间
   - 简化任务复杂度

4. **对象提取失败**
   - 检查输入网格格式
   - 调整连通性设置

### 调试技巧

```python
# 启用详细日志
import logging
logging.basicConfig(level=logging.DEBUG)

# 使用调试配置
engine = ARCSynthesisEngine("config/complex.yaml")

# 检查中间结果
result = engine.synthesize_program(task)
print(f"对象提取结果: {result.intermediate_results}")
```

## 📚 进阶使用

### 批量处理

```bash
# 批量处理任务目录
python main.py --batch data/arc/training --output results.json
```

### 性能分析

```python
from arc_synthesis_framework.utils.logging import PerformanceLogger

perf_logger = PerformanceLogger()

perf_logger.start_timer("synthesis")
result = engine.synthesize_program(task)
elapsed = perf_logger.end_timer("synthesis")

perf_logger.log_memory_usage("after_synthesis")
```

### 自定义验证器

```python
from arc_synthesis_framework.cegis.verifier import ProgramVerifier

class MyVerifier(ProgramVerifier):
    def _execute_program(self, program, input_data):
        # 自定义程序执行逻辑
        return {"success": True, "output": transformed_data}
```

## 🤝 贡献指南

1. Fork项目仓库
2. 创建功能分支
3. 添加测试
4. 提交代码
5. 创建Pull Request

## 📄 许可证

MIT License - 详见LICENSE文件

## 🔗 相关资源

- [ARC挑战赛官网](https://arcprize.org/)
- [Popper项目](https://github.com/logic-and-learning-lab/Popper)
- [归纳逻辑编程教程](https://en.wikipedia.org/wiki/Inductive_logic_programming)

## 📞 支持和联系

如有问题或建议，请：
1. 查看文档和FAQ
2. 检查issue列表
3. 创建新issue
4. 联系维护团队

---

**祝您使用愉快！🎉**