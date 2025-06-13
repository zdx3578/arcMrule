#!/usr/bin/env python3
"""
ARC程序合成 - 真实Popper Demo
专注于第一阶段：规则发现和人工验证

特点：
- 使用真实Popper ILP系统的Python API
- 生成标准的Prolog输入文件
- 专注规则发现，不自动应用到test
- 提供详细的参数配置指南
- 支持人工检查学到的规则
"""

import os
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import json
import time

# ==================== 配置类 ====================

@dataclass
class PopperConfig:
    """Popper配置参数"""
    timeout: int = 60                      # 超时时间(秒)
    max_vars: int = 6                      # 最大变量数
    max_body: int = 4                      # 最大体部字面量数
    max_rules: int = 3                     # 最大规则数
    solver: str = "rc2"                    # SAT求解器
    noisy: bool = True                     # 是否输出详细信息
    stats: bool = True                     # 是否显示统计信息

@dataclass
class ARCTask:
    """ARC任务定义"""
    task_id: str
    examples: List[Tuple[List[List[int]], List[List[int]]]]  # 训练示例
    test_cases: List[Tuple[List[List[int]], List[List[int]]]]  # 测试用例（暂不使用）

# ==================== Popper文件生成器 ====================

class PopperFileGenerator:
    """生成Popper所需的Prolog文件"""

    def __init__(self, config: PopperConfig):
        self.config = config

    def generate_files_for_task(self, task: ARCTask, output_dir: Path) -> Dict[str, Path]:
        """为任务生成所有Popper文件"""
        output_dir.mkdir(parents=True, exist_ok=True)

        files = {
            'examples': output_dir / 'exs.pl',
            'background': output_dir / 'bk.pl',
            'bias': output_dir / 'bias.pl'
        }

        # 生成示例文件
        self._generate_examples_file(task, files['examples'])

        # 生成背景知识文件
        self._generate_background_file(files['background'])

        # 生成偏置文件
        self._generate_bias_file(files['bias'])

        print(f"✅ Popper文件已生成到: {output_dir}")
        for name, path in files.items():
            print(f"   {name}: {path}")

        return files

    def _generate_examples_file(self, task: ARCTask, file_path: Path):
        """生成示例文件 (exs.pl)"""
        content = [
            f"% ARC任务示例文件: {task.task_id}",
            f"% 生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "% 正例 - 输入输出转换对"
        ]

        for i, (input_grid, output_grid) in enumerate(task.examples):
            # 将网格转换为Prolog项
            input_term = self._grid_to_prolog_term(input_grid)
            output_term = self._grid_to_prolog_term(output_grid)

            # 创建正例
            content.append(f"pos(transform({input_term}, {output_term})).")

        content.extend([
            "",
            "% 暂无负例 (Popper可以自动生成)",
            ""
        ])

        file_path.write_text('\n'.join(content), encoding='utf-8')

        # 简化调试输出
        if self.config.noisy:
            print(f"   📄 生成exs.pl: {len(content)}行")

        # 添加调试输出
        print(f"   📄 生成的bias.pl内容:")
        for i, line in enumerate(content[-20:], len(content)-19):  # 显示最后20行
            print(f"      {i:3d}: {line}")
        print(f"   📁 文件路径: {file_path}")
        print(f"   📊 文件大小: {file_path.stat().st_size} bytes")

    def _grid_to_prolog_term(self, grid: List[List[int]]) -> str:
        """将2D网格转换为Prolog项"""
        cells = []

        for r, row in enumerate(grid):
            for c, value in enumerate(row):
                if value != 0:  # 只记录非背景色
                    cells.append(f"cell({r},{c},{value})")

        if cells:
            return f"grid([{','.join(cells)}])"
        else:
            return "grid([])"  # 空网格






    def _generate_background_file(self, file_path: Path):
        """生成背景知识文件 (bk.pl) - 完全避免is/2"""
        content = [
            "% ARC任务背景知识 - 避免is/2触发Popper点空调用错误",
            "% 只保留不含算术运算的基础谓词",
            "",
            "% ===== 网格单元格操作 =====",
            "",
            "% 获取网格中的单元格 - 纯逻辑，无算术",
            "grid_cell(grid(Cells), R, C, Color) :-",
            "    member(cell(R, C, Color), Cells).",
            "",
            "% ===== 颜色替换操作 =====",
            "",
            "% 单一颜色替换 - 纯逻辑",
            "change_color(grid(Cells), OldColor, NewColor, grid(NewCells)) :-",
            "    maplist(replace_color(OldColor, NewColor), Cells, NewCells).",
            "",
            "% 替换单元格中的颜色",
            "replace_color(OldColor, NewColor, cell(R, C, OldColor), cell(R, C, NewColor)) :- !.",
            "replace_color(_, _, Cell, Cell).",
            "",
            "% ===== 颜色常量定义 =====",
            "",
            "color(0).",
            "color(1).",
            "color(2).",
            "color(3).",
            "color(4).",
            ""
        ]

        file_path.write_text('\n'.join(content), encoding='utf-8')

        if self.config.noisy:
            print(f"   📄 生成bk.pl: {len(content)}行 (无is/2版本)")

    def _generate_bias_file(self, file_path: Path):
        """生成偏置文件 (bias.pl) - 只保留无is/2的谓词"""
        content = [
            "% ARC任务偏置文件 - 避免is/2触发的Popper错误",
            "% 只包含纯逻辑谓词，无算术运算",
            "",
            "% ===== 头谓词定义 =====",
            "% 我们要学习的目标谓词",
            "head_pred(transform,2).",
            "",
            "% ===== 体谓词定义 =====",
            "% 只保留安全的、无算术的谓词",
            "",
            "% 网格操作 - 纯逻辑",
            "body_pred(grid_cell,4).",
            "",
            "% 颜色转换操作 - 纯逻辑",
            "body_pred(change_color,4).",
            "",
            "% 颜色常量 - 1元谓词",
            "body_pred(color,1).",
            "",
            "% ===== 类型定义 =====",
            "type(transform,(grid,grid)).",
            "type(grid_cell,(grid,int,int,int)).",
            "type(change_color,(grid,int,int,grid)).",
            "type(color,(int)).",
            "",
            "% ===== 方向定义 =====",
            "direction(transform/2,(in,out)).",
            "direction(grid_cell/4,(in,out,out,out)).",
            "direction(change_color/4,(in,in,in,out)).",
            "direction(color/1,(out)).",
            "",
            "% ===== Recall定义 - 消除WARNING =====",
            "recall(grid_cell,4).",
            "recall(change_color,4).",
            "recall(color,1).",
            "",
            "% ===== 学习控制参数 =====",
            f"max_vars({self.config.max_vars}).",
            f"max_body({self.config.max_body}).",
            f"max_rules({self.config.max_rules}).",
            "",
            "% 启用单例变量（对简单变换有用）",
            "allow_singletons.",
            "",
            "% ===== 约束 =====",
            "% 输入输出必须是网格",
            ":- not body_pred(P,A), head_pred(P,A).",
            ""
        ]

        file_path.write_text('\n'.join(content), encoding='utf-8')

        if self.config.noisy:
            print(f"   📄 生成bias.pl: {len(content)}行 (纯逻辑版本)")



# ==================== 真实Popper接口 ====================

class RealPopperInterface:
    """真实的Popper ILP接口 - 使用Python API"""

    def __init__(self, config: PopperConfig):
        self.config = config
        self._setup_popper_import()

    def learn_program(self, task_dir: Path) -> Optional[str]:
        """调用Popper学习程序"""
        print(f"🔧 调用Popper学习程序...")
        print(f"   任务目录: {task_dir}")
        print(f"   超时时间: {self.config.timeout}秒")

        # 简化文件检查
        if self.config.noisy:
            print(f"   📁 检查Popper文件:")
            for filename in ['exs.pl', 'bk.pl', 'bias.pl']:
                filepath = task_dir / filename
                if filepath.exists():
                    print(f"      ✅ {filename} ({filepath.stat().st_size} bytes)")
                else:
                    print(f"      ❌ {filename} 不存在!")

        try:
            # 使用Popper API
            start_time = time.time()

            # 创建Settings对象
            settings = self._create_popper_settings(task_dir)

            # 调用Popper核心学习函数
            prog, score, stats = self.learn_solution(settings)

            execution_time = time.time() - start_time
            print(f"   执行时间: {execution_time:.2f}秒")

            if prog is not None:
                print("   ✅ Popper学习成功")

                # 显示程序和分数
                if self.config.stats and stats:
                    print("   📊 学习统计:")
                    self._print_stats(stats)

                # 格式化程序输出
                program_str = self._format_program(prog)
                print(f"   🎯 学到的程序:")
                print(self._indent_text(program_str))
                print(f"   📈 程序分数: {score}")

                return program_str
            else:
                print("   ❌ 未能学到程序 (NO SOLUTION)")
                if self.config.stats and stats:
                    print("   📊 学习统计:")
                    self._print_stats(stats)
                return None

        except ImportError as e:
            print(f"   ❌ Popper导入失败: {str(e)}")
            print("   💡 请确保已安装Popper: pip install popper")
            return None
        except AttributeError as e:
            print(f"   ❌ Popper API属性错误: {str(e)}")
            print("   💡 可能是stats对象类型不匹配")
            return None
        except Exception as e:
            print(f"   ❌ 学习失败: {str(e)}")
            print(f"   💡 错误类型: {type(e).__name__}")
            # 打印详细错误信息用于调试
            import traceback
            if self.config.noisy:
                print("   🔍 详细错误:")
                traceback.print_exc()
            return None

    def _setup_popper_import(self):
        """设置Popper导入"""
        try:
            # 直接导入Popper模块 (pip安装版本)
            from popper.util import Settings#, print_prog_score
            from popper.loop import learn_solution

            # 保存引用
            self.Settings = Settings
            self.print_prog_score = Settings.print_prog_score
            self.learn_solution = learn_solution

            print(f"   ✅ Popper API导入成功")

        except ImportError as e:
            raise ImportError(f"无法导入Popper API: {str(e)}\n请安装Popper: pip install popper")

    def _create_popper_settings(self, task_dir: Path):
        """创建Popper设置对象"""
        # 创建设置字典 - 只包含Settings接受的参数
        settings_dict = {
            'kbpath': str(task_dir),
            'timeout': self.config.timeout,
            'max_vars': self.config.max_vars,
            'max_body': self.config.max_body,
            'max_rules': self.config.max_rules,
            'solver': self.config.solver,
        }

        # 尝试添加调试参数
        if self.config.noisy:
            settings_dict['debug'] = True
            settings_dict['verbose'] = True  # 尝试添加verbose

        # 创建Settings对象（移除stats参数）
        try:
            settings = self.Settings(**settings_dict)
        except TypeError as e:
            # 如果某些参数不被接受，逐个移除
            print(f"   ⚠️ 某些参数不被支持: {str(e)}")
            print(f"   🔧 尝试使用基本参数集...")

            # 尝试不同的参数组合
            for attempt in [
                # 尝试1: 移除verbose
                {k: v for k, v in settings_dict.items() if k != 'verbose'},
                # 尝试2: 移除debug和verbose
                {k: v for k, v in settings_dict.items() if k not in ['debug', 'verbose']},
                # 尝试3: 只用最基本的参数
                {
                    'kbpath': str(task_dir),
                    'timeout': self.config.timeout,
                    'max_vars': self.config.max_vars,
                    'max_body': self.config.max_body,
                    'max_rules': self.config.max_rules,
                }
            ]:
                try:
                    settings = self.Settings(**attempt)
                    print(f"   ✅ 成功创建Settings对象")
                    settings_dict = attempt
                    break
                except TypeError:
                    continue
            else:
                raise e

        # 尝试手动设置调试选项
        if self.config.noisy:
            try:
                # 尝试设置不同的调试属性 - 移除'stats'避免问题
                for debug_attr in ['debug', 'verbose', 'show_stats']:
                    if hasattr(settings, debug_attr):
                        setattr(settings, debug_attr, True)
                        print(f"   🔧 设置{debug_attr}=True")
            except Exception as e:
                print(f"   ⚠️ 无法设置调试选项: {str(e)}")

        if self.config.noisy:
            print(f"   🔧 Popper设置已创建")

        return settings

    def _format_program(self, prog) -> str:
        """格式化程序输出"""
        if isinstance(prog, str):
            return prog
        elif hasattr(prog, '__iter__'):
            # 如果prog是规则列表
            return '\n'.join(str(rule) for rule in prog)
        else:
            return str(prog)

    def _print_stats(self, stats):
        """打印学习统计信息"""
        if not stats:
            return

        # 检查stats类型，防止'bool' object has no attribute 'duration'错误
        if isinstance(stats, bool):
            print(f"      stats: {stats} (boolean)")
            return

        if isinstance(stats, dict):
            important_stats = [
                'num_pos', 'num_neg', 'num_rules',
                'learning_time', 'total_time',
                'num_literals', 'program_size'
            ]

            for stat in important_stats:
                if stat in stats:
                    print(f"      {stat}: {stats[stat]}")
        else:
            print(f"      stats类型: {type(stats)}")
            print(f"      stats内容: {str(stats)}")

    def _indent_text(self, text: str, indent: str = "      ") -> str:
        """为文本添加缩进"""
        return '\n'.join(indent + line for line in text.split('\n'))

# ==================== 主Demo引擎 ====================

class ARCPopperDemo:
    """ARC Popper Demo主引擎"""

    def __init__(self, config: PopperConfig):
        self.config = config
        self.file_generator = PopperFileGenerator(config)
        self.popper = RealPopperInterface(config)
        self.work_dir = Path("./arc_popper_work")

    def run_rule_discovery(self, task: ARCTask) -> Dict:
        """运行规则发现流程"""
        print("=" * 60)
        print(f"🎯 ARC规则发现: {task.task_id}")
        print("=" * 60)

        # 显示任务信息
        self._display_task_info(task)

        # 创建工作目录
        task_dir = self.work_dir / task.task_id
        task_dir.mkdir(parents=True, exist_ok=True)

        try:
            # 1. 生成Popper文件
            print(f"\n📝 步骤1: 生成Popper输入文件")
            files = self.file_generator.generate_files_for_task(task, task_dir)

            # 2. 调用Popper学习
            print(f"\n🧠 步骤2: Popper程序学习")
            learned_program = self.popper.learn_program(task_dir)

            # 3. 结果分析
            print(f"\n📊 步骤3: 结果分析")
            result = self._analyze_result(task, learned_program, files)

            return result

        except Exception as e:
            print(f"\n❌ 执行失败: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'task_id': task.task_id
            }

    def _display_task_info(self, task: ARCTask):
        """显示任务信息"""
        print(f"任务ID: {task.task_id}")
        print(f"训练示例数: {len(task.examples)}")
        print(f"测试用例数: {len(task.test_cases)}")

        print(f"\n📋 训练示例:")
        for i, (input_grid, output_grid) in enumerate(task.examples, 1):
            print(f"  示例 {i}:")
            print(f"    输入:  {input_grid}")
            print(f"    输出:  {output_grid}")

    def _analyze_result(self, task: ARCTask, program: Optional[str], files: Dict) -> Dict:
        """分析学习结果"""
        result = {
            'task_id': task.task_id,
            'success': program is not None,
            'program': program,
            'files_generated': files,
            'manual_verification_needed': True
        }

        if program:
            print(f"✅ 成功学到程序:")
            print(self._format_program(program))

            print(f"\n🔍 人工验证指南:")
            print(f"1. 检查程序逻辑是否正确")
            print(f"2. 验证是否适用于所有训练示例")
            print(f"3. 考虑程序的泛化能力")

            # 提供验证建议
            verification_tips = self._generate_verification_tips(task, program)
            if verification_tips:
                print(f"\n💡 验证提示:")
                for tip in verification_tips:
                    print(f"   - {tip}")
        else:
            print(f"❌ 未能学到程序")
            print(f"\n🔧 可能的改进方案:")
            print(f"   - 增加超时时间 (当前: {self.config.timeout}秒)")
            print(f"   - 调整max_vars参数 (当前: {self.config.max_vars})")
            print(f"   - 增加更多训练示例")
            print(f"   - 简化背景知识")

        # 显示生成的文件
        print(f"\n📁 生成的文件:")
        for name, path in files.items():
            print(f"   {name}: {path}")

        return result

    def _format_program(self, program: str) -> str:
        """格式化程序显示"""
        lines = program.strip().split('\n')
        formatted = []
        for line in lines:
            formatted.append(f"    {line}")
        return '\n'.join(formatted)

    def _generate_verification_tips(self, task: ARCTask, program: str) -> List[str]:
        """生成人工验证提示"""
        tips = []

        # 基于程序内容的提示
        if 'change_color' in program:
            tips.append("程序包含颜色转换，检查颜色映射是否正确")

        if 'transform(' in program:
            tips.append("这是基本的转换规则，验证输入输出关系")

        # 基于示例数量的提示
        if len(task.examples) <= 2:
            tips.append("示例较少，需要特别注意泛化性")

        if len(task.examples) >= 3:
            tips.append("有足够示例，检查规则是否在所有示例上都成立")

        return tips

# ==================== 测试用例创建 ====================

def create_test_tasks() -> List[ARCTask]:
    """创建测试任务"""
    tasks = []

    # 任务1: 简单颜色替换
    task1 = ARCTask(
        task_id="simple_color_replace",
        examples=[
            ([[1, 0], [0, 1]], [[2, 0], [0, 2]]),
            ([[1, 1], [0, 0]], [[2, 2], [0, 0]]),
            ([[0, 1], [1, 0]], [[0, 2], [2, 0]]),
        ],
        test_cases=[
            ([[1, 0], [1, 1]], [[2, 0], [2, 2]]),  # 暂不使用
        ]
    )
    tasks.append(task1)

    # 任务2: 多颜色替换
    task2 = ARCTask(
        task_id="multi_color_replace",
        examples=[
            ([[1, 2], [3, 0]], [[4, 5], [6, 0]]),
            ([[2, 1], [0, 3]], [[5, 4], [0, 6]]),
        ],
        test_cases=[
            ([[1, 3], [2, 0]], [[4, 6], [5, 0]]),
        ]
    )
    tasks.append(task2)

    return tasks

# ==================== 配置向导 ====================

def setup_popper_config() -> PopperConfig:
    """配置向导"""
    print("🔧 Popper配置向导")
    print("-" * 30)

    # 其他参数
    print(f"参数设置 (直接按回车使用默认值):")

    timeout = input("超时时间/秒 [60]: ").strip()
    timeout = int(timeout) if timeout else 60

    max_vars = input("最大变量数 [6]: ").strip()
    max_vars = int(max_vars) if max_vars else 6

    return PopperConfig(
        timeout=timeout,
        max_vars=max_vars
    )

# ==================== 主运行函数 ====================

def main():
    """主函数"""
    print("🧠 ARC真实Popper Demo - 规则发现阶段")
    print("="*50)

    # 配置Popper
    config = setup_popper_config()

    print(f"\n📋 配置总结:")
    print(f"   超时时间: {config.timeout}秒")
    print(f"   最大变量: {config.max_vars}")
    print(f"   最大规则: {config.max_rules}")

    # 创建demo引擎
    demo = ARCPopperDemo(config)

    # 创建测试任务
    tasks = create_test_tasks()

    print(f"\n🎯 开始规则发现...")

    # 运行每个任务
    results = []
    for task in tasks:
        result = demo.run_rule_discovery(task)
        results.append(result)

        print(f"\n" + "="*40)

        # 询问是否继续
        if len(tasks) > 1:
            continue_choice = input("继续下一个任务? (y/n): ").strip().lower()
            if continue_choice != 'y':
                break

    # 总结
    print(f"\n📊 总结")
    print("="*30)
    successful = sum(1 for r in results if r['success'])
    print(f"总任务: {len(results)}")
    print(f"成功: {successful}")
    print(f"成功率: {successful/len(results)*100:.1f}%")

    # 下一步指导
    if successful > 0:
        print(f"\n🎉 恭喜！已成功发现 {successful} 个规则")
        print(f"下一步:")
        print(f"1. 人工验证学到的规则")
        print(f"2. 测试规则在新数据上的表现")
        print(f"3. 如果规则正确，可以扩展到更复杂的任务")

def quick_demo():
    """快速演示（使用默认配置）"""
    print("🚀 快速演示模式")

    # 使用默认配置
    config = PopperConfig()

    demo = ARCPopperDemo(config)

    # 运行一个简单任务
    simple_task = ARCTask(
        task_id="demo",
        examples=[([[1, 0], [0, 1]], [[2, 0], [0, 2]])],
        test_cases=[]
    )

    result = demo.run_rule_discovery(simple_task)

    if result['success']:
        print("\n🎉 演示成功！规则发现功能正常工作。")
    else:
        print("\n💡 如需调试，请运行完整配置模式")

# ==================== 额外工具函数 ====================

def install_popper_guide():
    """显示Popper安装指南"""
    print("🔧 Popper安装指南")
    print("="*40)
    print()
    print("1. 使用pip安装 (推荐):")
    print("   pip install popper")
    print()
    print("2. 安装SAT求解器:")
    print("   pip install python-sat")
    print()
    print("3. 验证安装:")
    print("   python -c \"from popper.util import Settings; print('Popper API OK')\"")
    print()
    print("4. 运行ARC Demo:")
    print("   python arc_minimal_real_popper_demo.py quick")
    print()
    print("📝 注意事项:")
    print("- 确保Python版本 >= 3.7")
    print("- 如果pip安装失败，可以从源码安装:")
    print("  git clone https://github.com/logic-and-learning-lab/Popper.git")
    print("  cd Popper && pip install -e .")
    print()
    print("🐛 常见问题:")
    print("- ImportError: 重新安装 pip install --force-reinstall popper")
    print("- 学习失败: 尝试增加timeout参数")
    print("- 内存不足: 减少max_vars参数")

def validate_popper_installation() -> bool:
    """验证Popper安装是否正确"""
    print(f"🔍 验证Popper安装")

    # 尝试导入Popper API
    try:
        from popper.util import Settings#, print_prog_score
        from popper.loop import learn_solution

        print("✅ Popper API导入成功")

        # 尝试创建Settings对象
        temp_dir = Path("./temp_test")
        temp_dir.mkdir(exist_ok=True)

        try:
            # 尝试基本参数
            settings = Settings(kbpath=str(temp_dir))
            print("✅ Settings对象创建成功")

            # 打印Settings支持的参数
            print("✅ Settings支持的参数:")
            for attr in dir(settings):
                if not attr.startswith('_'):
                    print(f"      {attr}: {getattr(settings, attr, 'N/A')}")

            temp_dir.rmdir()  # 清理
        except Exception as e:
            print(f"⚠️ Settings创建警告: {str(e)}")
            if temp_dir.exists():
                temp_dir.rmdir()  # 清理

        return True

    except ImportError as e:
        print(f"❌ Popper API导入失败: {str(e)}")
        print("💡 请安装Popper: pip install popper")
        return False
    except Exception as e:
        print(f"❌ 验证失败: {str(e)}")
        return False

def run_popper_example():
    """运行Popper测试"""
    print("🧪 运行Popper API测试")

    if not validate_popper_installation():
        print("\n请先正确安装Popper")
        install_popper_guide()
        return

    # 创建简单测试
    print(f"创建简单测试...")

    try:
        from popper.util import Settings, print_prog_score
        from popper.loop import learn_solution

        # 创建临时测试目录
        test_dir = Path("./popper_test")
        test_dir.mkdir(exist_ok=True)

        # 创建简单示例文件
        (test_dir / "exs.pl").write_text("""
% 简单测试示例
pos(even(0)).
pos(even(2)).
pos(even(4)).
neg(even(1)).
neg(even(3)).
""")

        (test_dir / "bk.pl").write_text("""
% 背景知识
succ(0,1).
succ(1,2).
succ(2,3).
succ(3,4).
""")

        (test_dir / "bias.pl").write_text("""
% 偏置
head_pred(even,1).
body_pred(succ,2).
max_vars(3).
max_body(2).
""")

        # 创建设置并学习
        settings = Settings(kbpath=str(test_dir))

        start_time = time.time()
        prog, score, stats = learn_solution(settings)
        execution_time = time.time() - start_time

        print(f"执行时间: {execution_time:.2f}秒")

        if prog is not None:
            print("✅ 测试运行成功")
            print("\n学到的程序:")
            Settings.print_prog_score(prog, score)

            if stats:
                print(f"\n统计信息:")
                for key, value in stats.items():
                    print(f"  {key}: {value}")
        else:
            print("❌ 测试运行失败 - 未学到程序")
            if stats:
                print(f"统计信息: {stats}")

        # 清理
        shutil.rmtree(test_dir)

    except ImportError as e:
        print(f"❌ 导入失败: {str(e)}")
    except Exception as e:
        print(f"❌ 运行失败: {str(e)}")
        import traceback
        # traceback.print_exc():
        #         print(f"统计信息: {stats}")

    except ImportError as e:
        print(f"❌ 导入失败: {str(e)}")
    except Exception as e:
        print(f"❌ 运行失败: {str(e)}")
        import traceback
        traceback.print_exc()

def interactive_config():
    """交互式配置模式"""
    print("🎛️ 交互式配置模式")
    print("="*30)

    config = setup_popper_config()

    print(f"\n验证配置...")
    if validate_popper_installation():
        print("✅ 配置验证成功")

        print(f"\n选择操作:")
        print("1. 运行ARC规则发现")
        print("2. 运行Popper测试")
        print("3. 显示配置信息")
        print("4. 退出")

        while True:
            choice = input("\n请选择 (1-4): ").strip()

            if choice == '1':
                demo = ARCPopperDemo(config)
                tasks = create_test_tasks()
                for task in tasks:
                    result = demo.run_rule_discovery(task)
                    print(f"\n任务 {task.task_id} 完成")

                    if len(tasks) > 1:
                        cont = input("继续下一个任务? (y/n): ").strip().lower()
                        if cont != 'y':
                            break
                break

            elif choice == '2':
                run_popper_example()
                break

            elif choice == '3':
                print(f"\n当前配置:")
                print(f"  超时时间: {config.timeout}秒")
                print(f"  最大变量: {config.max_vars}")
                print(f"  最大规则: {config.max_rules}")
                print(f"  求解器: {config.solver}")

            elif choice == '4':
                print("退出")
                break

            else:
                print("无效选择，请重新输入")
    else:
        print("❌ 配置验证失败")
        install_popper_guide()

def show_usage():
    """显示使用说明"""
    print("🔧 ARC Popper Demo 使用说明")
    print("="*40)
    print()
    print("运行模式:")
    print("  python arc_minimal_real_popper_demo.py          # 完整交互模式")
    print("  python arc_minimal_real_popper_demo.py quick    # 快速演示")
    print("  python arc_minimal_real_popper_demo.py install  # 安装指南")
    print("  python arc_minimal_real_popper_demo.py validate # 验证安装")
    print("  python arc_minimal_real_popper_demo.py example  # 运行示例")
    print("  python arc_minimal_real_popper_demo.py config   # 交互配置")
    print()
    print("文件结构:")
    print("  ./popper/                    # Popper安装目录")
    print("  ./arc_popper_work/           # 工作目录")
    print("    ├── task_id/")
    print("    │   ├── exs.pl            # 训练示例")
    print("    │   ├── bk.pl             # 背景知识")
    print("    │   └── bias.pl           # 学习偏置")
    print()
    print("调试技巧:")
    print("  - 增加timeout参数处理复杂任务")
    print("  - 调整max_vars控制复杂度")
    print("  - 检查生成的.pl文件是否正确")
    print("  - 使用--noisy查看详细输出")

# 更新主入口点
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        command = sys.argv[1]

        if command == "quick":
            quick_demo()
        elif command == "install":
            install_popper_guide()
        elif command == "validate":
            validate_popper_installation()
        elif command == "example":
            run_popper_example()
        elif command == "config":
            interactive_config()
        elif command == "help":
            show_usage()
        else:
            print(f"未知命令: {command}")
            show_usage()
    else:
        main()