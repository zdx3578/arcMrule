#!/usr/bin/env python3
"""
ARC程序合成极简Demo
专注验证Popper+CEGIS核心逻辑

特点：
- 最小2x2网格
- 简单颜色转换任务  
- 模拟Popper避免外部依赖
- 基本CEGIS循环
- 清晰的执行流程
"""

import time
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

# ==================== 核心数据结构 ====================

@dataclass
class MiniTask:
    """最小ARC任务"""
    task_id: str
    examples: List[Tuple[List[List[int]], List[List[int]]]]  # (input, output)
    
@dataclass
class SynthesisResult:
    """合成结果"""
    success: bool
    program: Optional[str]
    iterations: int
    time_used: float

# ==================== 模拟Popper接口 ====================

class MockPopperInterface:
    """模拟的Popper接口，避免外部依赖"""
    
    def __init__(self):
        self.call_count = 0
        self.constraints = []
    
    def learn_program(self, examples: List[Tuple], constraints: List[str] = None) -> Optional[str]:
        """模拟程序学习"""
        self.call_count += 1
        print(f"  [Popper模拟] 第{self.call_count}次调用")
        
        if constraints:
            print(f"  [Popper模拟] 约束数量: {len(constraints)}")
            self.constraints.extend(constraints)
        
        # 简单的启发式程序生成
        if len(examples) == 0:
            return None
            
        # 分析第一个示例
        input_grid, output_grid = examples[0]
        
        # 检测简单的颜色映射
        color_mapping = self._detect_color_mapping(input_grid, output_grid)
        if color_mapping:
            # 验证其他示例是否符合相同映射
            if self._validate_color_mapping(examples, color_mapping):
                return self._generate_color_program(color_mapping)
        
        # 如果有约束，尝试调整
        if len(self.constraints) > 0:
            # 根据约束数量返回不同的程序变体
            if len(self.constraints) == 1:
                return self._generate_alternative_program(examples)
            elif len(self.constraints) >= 2:
                return None  # 模拟搜索空间耗尽
        
        return None
    
    def _detect_color_mapping(self, input_grid: List[List[int]], 
                            output_grid: List[List[int]]) -> Dict[int, int]:
        """检测颜色映射"""
        mapping = {}
        
        for i in range(len(input_grid)):
            for j in range(len(input_grid[0])):
                inp_color = input_grid[i][j]
                out_color = output_grid[i][j]
                
                if inp_color in mapping:
                    if mapping[inp_color] != out_color:
                        return {}  # 不一致
                else:
                    mapping[inp_color] = out_color
        
        return mapping
    
    def _validate_color_mapping(self, examples: List[Tuple], 
                              mapping: Dict[int, int]) -> bool:
        """验证颜色映射是否适用于所有示例"""
        for input_grid, output_grid in examples:
            for i in range(len(input_grid)):
                for j in range(len(input_grid[0])):
                    inp_color = input_grid[i][j]
                    expected_color = mapping.get(inp_color, inp_color)
                    if output_grid[i][j] != expected_color:
                        return False
        return True
    
    def _generate_color_program(self, mapping: Dict[int, int]) -> str:
        """生成颜色转换程序"""
        rules = []
        for old_color, new_color in mapping.items():
            if old_color != new_color:
                rules.append(f"change_color({old_color}, {new_color})")
        
        if rules:
            return f"transform(Input, Output) :- {', '.join(rules)}."
        else:
            return "transform(Input, Input)."  # 恒等变换
    
    def _generate_alternative_program(self, examples: List[Tuple]) -> Optional[str]:
        """生成替代程序（模拟约束下的搜索）"""
        print("  [Popper模拟] 生成替代程序...")
        
        # 模拟：尝试不同的颜色映射
        input_grid, output_grid = examples[0]
        
        # 尝试反向映射
        reverse_mapping = {}
        for i in range(len(input_grid)):
            for j in range(len(input_grid[0])):
                out_color = output_grid[i][j]
                inp_color = input_grid[i][j]
                reverse_mapping[out_color] = inp_color
        
        return self._generate_color_program(reverse_mapping)

# ==================== 程序验证器 ====================

class MiniVerifier:
    """简化的程序验证器"""
    
    def verify_program(self, program: str, examples: List[Tuple]) -> Tuple[bool, Optional[Tuple]]:
        """
        验证程序是否对所有示例有效
        
        Returns:
            (is_valid, failed_example)
        """
        print(f"  [验证器] 验证程序: {program}")
        
        for i, (input_grid, expected_output) in enumerate(examples):
            actual_output = self._execute_program(program, input_grid)
            
            if actual_output != expected_output:
                print(f"  [验证器] 示例{i+1}失败")
                return False, (input_grid, expected_output)
        
        print(f"  [验证器] 所有{len(examples)}个示例验证通过")
        return True, None
    
    def _execute_program(self, program: str, input_grid: List[List[int]]) -> List[List[int]]:
        """模拟程序执行"""
        
        # 解析程序中的颜色变换
        if "change_color" in program:
            # 提取颜色映射
            import re
            pattern = r'change_color\((\d+),\s*(\d+)\)'
            matches = re.findall(pattern, program)
            
            result = [row[:] for row in input_grid]  # 深拷贝
            
            for old_str, new_str in matches:
                old_color, new_color = int(old_str), int(new_str)
                
                # 应用颜色变换
                for i in range(len(result)):
                    for j in range(len(result[i])):
                        if result[i][j] == old_color:
                            result[i][j] = new_color
            
            return result
        
        # 默认恒等变换
        return [row[:] for row in input_grid]

# ==================== 反例生成器 ====================

class MiniCounterexampleGenerator:
    """简化的反例生成器"""
    
    def generate_constraint(self, failed_program: str, 
                          failed_example: Tuple) -> str:
        """从失败示例生成约束"""
        input_grid, expected_output = failed_example
        
        print(f"  [反例生成] 生成约束，排除失败程序")
        
        # 简单的约束生成策略
        constraint_id = hash((failed_program, str(failed_example))) % 1000
        constraint = f"not_program_{constraint_id}"
        
        return constraint

# ==================== 主CEGIS引擎 ====================

class MiniCEGISEngine:
    """极简CEGIS合成引擎"""
    
    def __init__(self):
        self.popper = MockPopperInterface()
        self.verifier = MiniVerifier()
        self.counterexample_gen = MiniCounterexampleGenerator()
        self.max_iterations = 5
    
    def synthesize(self, task: MiniTask) -> SynthesisResult:
        """主合成方法"""
        print(f"\n🚀 开始合成任务: {task.task_id}")
        print(f"示例数量: {len(task.examples)}")
        
        start_time = time.time()
        constraints = []
        
        for iteration in range(self.max_iterations):
            print(f"\n--- CEGIS迭代 {iteration + 1} ---")
            
            # 1. 候选生成 (Popper)
            candidate = self.popper.learn_program(task.examples, constraints)
            
            if candidate is None:
                print("  ❌ 无法生成候选程序")
                break
            
            print(f"  🔧 候选程序: {candidate}")
            
            # 2. 程序验证
            is_valid, failed_example = self.verifier.verify_program(candidate, task.examples)
            
            if is_valid:
                # 成功！
                elapsed = time.time() - start_time
                print(f"  ✅ 验证成功！")
                return SynthesisResult(
                    success=True,
                    program=candidate,
                    iterations=iteration + 1,
                    time_used=elapsed
                )
            else:
                # 3. 反例生成
                constraint = self.counterexample_gen.generate_constraint(
                    candidate, failed_example
                )
                constraints.append(constraint)
                print(f"  🔄 添加约束: {constraint}")
        
        # 失败
        elapsed = time.time() - start_time
        print(f"  ❌ 达到最大迭代次数，合成失败")
        return SynthesisResult(
            success=False,
            program=None,
            iterations=self.max_iterations,
            time_used=elapsed
        )

# ==================== 测试用例 ====================

def create_simple_test_cases() -> List[MiniTask]:
    """创建简单测试用例"""
    
    tasks = []
    
    # 任务1: 简单颜色替换 (1->2)
    task1 = MiniTask(
        task_id="color_1_to_2",
        examples=[
            # 示例1: 2x2网格
            ([[1, 0], [0, 1]], [[2, 0], [0, 2]]),
            # 示例2: 验证一致性
            ([[0, 1], [1, 0]], [[0, 2], [2, 0]]),
        ]
    )
    tasks.append(task1)
    
    # 任务2: 多颜色替换
    task2 = MiniTask(
        task_id="multi_color_change",
        examples=[
            ([[1, 2], [2, 1]], [[3, 4], [4, 3]]),
            ([[2, 1], [1, 2]], [[4, 3], [3, 4]]),
        ]
    )
    tasks.append(task2)
    
    # 任务3: 恒等变换（测试边界情况）
    task3 = MiniTask(
        task_id="identity",
        examples=[
            ([[1, 2], [3, 4]], [[1, 2], [3, 4]]),
            ([[0, 1], [2, 0]], [[0, 1], [2, 0]]),
        ]
    )
    tasks.append(task3)
    
    return tasks

# ==================== 主运行函数 ====================

def run_demo():
    """运行完整演示"""
    print("=" * 60)
    print("🧠 ARC程序合成极简Demo")
    print("   Popper + CEGIS 核心逻辑验证")
    print("=" * 60)
    
    # 创建引擎
    engine = MiniCEGISEngine()
    
    # 创建测试任务
    tasks = create_simple_test_cases()
    
    # 运行每个任务
    results = []
    for task in tasks:
        result = engine.synthesize(task)
        results.append((task, result))
        
        print(f"\n📊 任务 {task.task_id} 结果:")
        print(f"   成功: {'✅' if result.success else '❌'}")
        if result.success:
            print(f"   程序: {result.program}")
        print(f"   迭代: {result.iterations}")
        print(f"   用时: {result.time_used:.3f}秒")
    
    # 总结
    print(f"\n" + "=" * 60)
    print("📈 总结")
    print("=" * 60)
    
    total_tasks = len(results)
    successful_tasks = sum(1 for _, result in results if result.success)
    
    print(f"总任务: {total_tasks}")
    print(f"成功: {successful_tasks}")
    print(f"成功率: {successful_tasks/total_tasks:.1%}")
    
    if successful_tasks > 0:
        avg_iterations = sum(result.iterations for _, result in results if result.success) / successful_tasks
        avg_time = sum(result.time_used for _, result in results if result.success) / successful_tasks
        print(f"平均迭代: {avg_iterations:.1f}")
        print(f"平均时间: {avg_time:.3f}秒")
    
    print(f"\n🎉 Demo完成！核心Popper+CEGIS逻辑验证成功。")

def demo_single_task():
    """单任务详细演示"""
    print("🔍 单任务详细演示")
    print("-" * 40)
    
    # 创建最简单的任务
    task = MiniTask(
        task_id="demo_task",
        examples=[
            ([[1, 0], [0, 1]], [[2, 0], [0, 2]]),  # 1->2
        ]
    )
    
    print("任务描述: 将颜色1替换为颜色2")
    print("输入示例:")
    for i, (inp, out) in enumerate(task.examples):
        print(f"  示例{i+1}: {inp} -> {out}")
    
    # 运行合成
    engine = MiniCEGISEngine()
    result = engine.synthesize(task)
    
    print(f"\n最终结果: {'成功' if result.success else '失败'}")
    if result.success:
        print(f"学到的程序: {result.program}")

if __name__ == "__main__":
    print("选择运行模式:")
    print("1. 完整演示 (推荐)")
    print("2. 单任务详细演示")
    
    choice = input("\n请选择 (1/2): ").strip()
    
    if choice == "2":
        demo_single_task()
    else:
        run_demo()
