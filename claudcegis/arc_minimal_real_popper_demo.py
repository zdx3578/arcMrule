#!/usr/bin/env python3
"""
ARC程序合成 - 真实Popper Demo
专注于第一阶段：规则发现和人工验证

特点：
- 使用真实Popper ILP系统
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
    popper_path: str = "./popper"          # Popper安装路径
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
        """生成背景知识文件 (bk.pl)"""
        content = [
            "% ARC任务背景知识",
            "% 定义网格操作和颜色转换的基础谓词",
            "",
            "% ===== 网格基础操作 =====",
            "",
            "% 获取网格中的单元格",
            "grid_cell(grid(Cells), R, C, Color) :-",
            "    member(cell(R, C, Color), Cells).",
            "",
            "% 检查网格是否为空",
            "empty_grid(grid([])).",
            "",
            "% 获取网格中所有颜色",
            "grid_colors(grid(Cells), Colors) :-",
            "    findall(Color, member(cell(_, _, Color), Cells), AllColors),",
            "    sort(AllColors, Colors).",
            "",
            "% ===== 颜色转换操作 =====",
            "",
            "% 单一颜色替换",
            "change_color(grid(Cells), OldColor, NewColor, grid(NewCells)) :-",
            "    maplist(replace_color(OldColor, NewColor), Cells, NewCells).",
            "",
            "% 替换单元格中的颜色",
            "replace_color(OldColor, NewColor, cell(R, C, OldColor), cell(R, C, NewColor)) :- !.",
            "replace_color(_, _, Cell, Cell).",
            "",
            "% 批量颜色替换",
            "change_colors(Grid, [], Grid).",
            "change_colors(Grid, [OldColor-NewColor|Rest], FinalGrid) :-",
            "    change_color(Grid, OldColor, NewColor, TempGrid),",
            "    change_colors(TempGrid, Rest, FinalGrid).",
            "",
            "% ===== 网格分析谓词 =====",
            "",
            "% 统计颜色出现次数",
            "color_count(grid(Cells), Color, Count) :-",
            "    include(has_color(Color), Cells, ColorCells),",
            "    length(ColorCells, Count).",
            "",
            "has_color(Color, cell(_, _, Color)).",
            "",
            "% 检查两个网格大小是否相同",
            "same_size(grid(Cells1), grid(Cells2)) :-",
            "    grid_dimensions(grid(Cells1), W1, H1),",
            "    grid_dimensions(grid(Cells2), W2, H2),",
            "    W1 = W2, H1 = H2.",
            "",
            "% 获取网格维度",
            "grid_dimensions(grid(Cells), Width, Height) :-",
            "    (Cells = [] ->",
            "        Width = 0, Height = 0",
            "    ;   findall(R, member(cell(R, _, _), Cells), Rs),",
            "        findall(C, member(cell(_, C, _), Cells), Cs),",
            "        max_list([0|Rs], MaxR), max_list([0|Cs], MaxC),",
            "        Width is MaxC + 1, Height is MaxR + 1",
            "    ).",
            "",
            "% ===== 工具谓词 =====",
            "",
            "% 获取列表中的最大值",
            "max_list([X], X) :- !.",
            "max_list([H|T], Max) :-",
            "    max_list(T, MaxT),",
            "    Max is max(H, MaxT).",
            ""
        ]

        file_path.write_text('\n'.join(content), encoding='utf-8')

    def _generate_bias_file(self, file_path: Path):
        """生成偏置文件 (bias.pl)"""
        content = [
            "% ARC任务偏置文件",
            "% 定义学习空间和约束",
            "",
            "% ===== 头谓词定义 =====",
            "% 我们要学习的目标谓词",
            "head_pred(transform,2).",
            "",
            "% ===== 体谓词定义 =====",
            "% 可以在规则体中使用的谓词",
            "",
            "% 基础网格操作",
            "body_pred(grid_cell,4).",
            "body_pred(empty_grid,1).",
            "body_pred(grid_colors,2).",
            "body_pred(same_size,2).",
            "body_pred(grid_dimensions,3).",
            "",
            "% 颜色转换操作",
            "body_pred(change_color,4).",
            "body_pred(change_colors,3).",
            "body_pred(color_count,3).",
            "",
            "% 常量定义",
            "% 允许使用的颜色值",
            "body_pred(color_0,0).",
            "body_pred(color_1,0).",
            "body_pred(color_2,0).",
            "body_pred(color_3,0).",
            "body_pred(color_4,0).",
            "",
            "% 定义常量事实",
            "color_0(0).",
            "color_1(1).",
            "color_2(2).",
            "color_3(3).",
            "color_4(4).",
            "",
            "% ===== 类型定义 =====",
            "type(transform,(grid,grid)).",
            "type(change_color,(grid,int,int,grid)).",
            "type(change_colors,(grid,list,grid)).",
            "type(grid_cell,(grid,int,int,int)).",
            "",
            "% ===== 方向定义 =====",
            "direction(transform,(in,out)).",
            "direction(change_color,(in,in,in,out)).",
            "direction(change_colors,(in,in,out)).",
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
            "",
            "% 防止生成过于复杂的规则",
            ":- max_clauses(C), C > 3.",
            ""
        ]

        file_path.write_text('\n'.join(content), encoding='utf-8')

# ==================== 真实Popper接口 ====================

class RealPopperInterface:
    """真实的Popper ILP接口"""

    def __init__(self, config: PopperConfig):
        self.config = config
        self._verify_popper_installation()

    def learn_program(self, task_dir: Path) -> Optional[str]:
        """调用Popper学习程序"""
        print(f"🔧 调用Popper学习程序...")
        print(f"   任务目录: {task_dir}")
        print(f"   超时时间: {self.config.timeout}秒")

        try:
            # 构建Popper命令
            cmd = self._build_popper_command(task_dir)
            print(f"   执行命令: {' '.join(cmd)}")

            # 执行Popper
            start_time = time.time()
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.config.timeout,
                cwd=self.config.popper_path
            )
            execution_time = time.time() - start_time

            print(f"   执行时间: {execution_time:.2f}秒")
            print(f"   返回码: {result.returncode}")

            if result.returncode == 0:
                print("   ✅ Popper执行成功")

                # 解析输出
                program = self._parse_popper_output(result.stdout)

                if program:
                    print("   ✅ 成功提取程序")
                    return program
                else:
                    print("   ⚠️ 未能提取程序")
                    print("   完整输出:")
                    print(self._indent_text(result.stdout))
            else:
                print("   ❌ Popper执行失败")
                print("   错误信息:")
                print(self._indent_text(result.stderr))

            return None

        except subprocess.TimeoutExpired:
            print(f"   ⏰ Popper执行超时 ({self.config.timeout}秒)")
            return None
        except FileNotFoundError:
            print(f"   ❌ 找不到Popper: {self.config.popper_path}")
            return None
        except Exception as e:
            print(f"   ❌ 执行错误: {str(e)}")
            return None

    def _build_popper_command(self, task_dir: Path) -> List[str]:
        """构建Popper命令"""
        cmd = [
            'python',
            str(Path(self.config.popper_path) / 'popper.py'),
            str(task_dir),
            '--timeout', str(self.config.timeout),
            '--solver', self.config.solver
        ]

        if self.config.noisy:
            cmd.append('--noisy')

        if self.config.stats:
            cmd.append('--stats')

        return cmd

    def _parse_popper_output(self, output: str) -> Optional[str]:
        """解析Popper输出，提取学到的程序"""
        lines = output.strip().split('\n')

        # 查找程序段
        program_lines = []
        in_program = False

        for line in lines:
            line = line.strip()

            # 检测程序开始
            if 'Program:' in line or 'SOLUTION' in line:
                in_program = True
                continue

            # 检测程序结束
            if in_program and (line.startswith('Precision:') or
                              line.startswith('Recall:') or
                              line.startswith('Time:') or
                              line == ''):
                break

            # 收集程序行
            if in_program and line and not line.startswith('%'):
                program_lines.append(line)

        if program_lines:
            return '\n'.join(program_lines)

        return None

    def _verify_popper_installation(self):
        """验证Popper是否正确安装"""
        popper_script = Path(self.config.popper_path) / 'popper.py'

        if not popper_script.exists():
            raise FileNotFoundError(
                f"Popper脚本未找到: {popper_script}\n"
                f"请确保已正确安装Popper到: {self.config.popper_path}"
            )

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

    # 获取Popper路径
    print("请输入Popper安装路径:")
    print("(如果已克隆到当前目录，直接按回车)")
    popper_path = input("Popper路径 [./popper]: ").strip()
    if not popper_path:
        popper_path = "./popper"

    # 验证路径
    if not Path(popper_path).exists():
        print(f"⚠️ 路径不存在: {popper_path}")
        print("请确保已正确安装Popper")
        print("安装方法: git clone https://github.com/logic-and-learning-lab/Popper.git")

    # 其他参数
    print(f"\n其他参数 (直接按回车使用默认值):")

    timeout = input("超时时间/秒 [60]: ").strip()
    timeout = int(timeout) if timeout else 60

    max_vars = input("最大变量数 [6]: ").strip()
    max_vars = int(max_vars) if max_vars else 6

    return PopperConfig(
        popper_path=popper_path,
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
    print(f"   Popper路径: {config.popper_path}")
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

    # 检查Popper是否存在
    if not Path(config.popper_path).exists():
        print(f"❌ 未找到Popper: {config.popper_path}")
        print("请先安装Popper:")
        print("git clone https://github.com/logic-and-learning-lab/Popper.git")
        return

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
    print("1. 克隆Popper仓库:")
    print("   git clone https://github.com/logic-and-learning-lab/Popper.git")
    print()
    print("2. 进入目录并安装依赖:")
    print("   cd Popper")
    print("   pip install -r requirements.txt")
    print()
    print("3. 安装SAT求解器 (推荐RC2):")
    print("   pip install python-sat")
    print()
    print("4. 验证安装:")
    print("   python popper.py examples/robots1")
    print()
    print("5. 如果成功，你会看到学到的程序输出")
    print()
    print("📝 注意事项:")
    print("- 确保Python版本 >= 3.7")
    print("- 如果遇到问题，查看Popper的README文档")
    print("- 对于复杂任务，可能需要调整timeout参数")

def validate_popper_installation(popper_path: str) -> bool:
    """验证Popper安装是否正确"""
    print(f"🔍 验证Popper安装: {popper_path}")

    popper_script = Path(popper_path) / 'popper.py'

    # 检查主脚本
    if not popper_script.exists():
        print(f"❌ 未找到popper.py: {popper_script}")
        return False

    # 检查示例目录
    examples_dir = Path(popper_path) / 'examples'
    if not examples_dir.exists():
        print(f"❌ 未找到examples目录: {examples_dir}")
        return False

    # 尝试运行简单测试
    try:
        result = subprocess.run(
            ['python', str(popper_script), '--help'],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=popper_path
        )

        if result.returncode == 0:
            print("✅ Popper安装验证成功")
            return True
        else:
            print(f"❌ Popper执行失败: {result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        print("❌ Popper响应超时")
        return False
    except FileNotFoundError:
        print("❌ Python解释器未找到")
        return False
    except Exception as e:
        print(f"❌ 验证失败: {str(e)}")
        return False

def run_popper_example():
    """运行Popper内置示例"""
    print("🧪 运行Popper内置示例")

    config = PopperConfig()

    if not validate_popper_installation(config.popper_path):
        print("\n请先正确安装Popper")
        install_popper_guide()
        return

    # 运行robots示例
    examples_dir = Path(config.popper_path) / 'examples'
    robots_example = examples_dir / 'robots1'

    if not robots_example.exists():
        print(f"❌ 未找到robots示例: {robots_example}")
        return

    print(f"运行示例: {robots_example}")

    try:
        result = subprocess.run(
            ['python', 'popper.py', str(robots_example)],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=config.popper_path
        )

        if result.returncode == 0:
            print("✅ 示例运行成功")
            print("\n学到的程序:")
            print(result.stdout)
        else:
            print(f"❌ 示例运行失败")
            print(f"错误信息: {result.stderr}")

    except subprocess.TimeoutExpired:
        print("❌ 示例运行超时")
    except Exception as e:
        print(f"❌ 运行失败: {str(e)}")

def interactive_config():
    """交互式配置模式"""
    print("🎛️ 交互式配置模式")
    print("="*30)

    config = setup_popper_config()

    print(f"\n验证配置...")
    if validate_popper_installation(config.popper_path):
        print("✅ 配置验证成功")

        print(f"\n选择操作:")
        print("1. 运行ARC规则发现")
        print("2. 运行Popper内置示例")
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
                print(f"  Popper路径: {config.popper_path}")
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
            config = PopperConfig()
            validate_popper_installation(config.popper_path)
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

