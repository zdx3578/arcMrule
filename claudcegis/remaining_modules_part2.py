# =====================================================================
# ARC程序合成框架 - 剩余模块第二部分
# =====================================================================

# =====================================================================
# 4. cegis/synthesizer.py - CEGIS合成器实现
# =====================================================================
CEGIS_SYNTHESIZER_PY = '''"""
CEGIS合成器 - 反例引导的归纳合成
"""
import logging
from typing import List, Dict, Optional, Tuple, Any, Set
from dataclasses import dataclass
import time
import itertools

logger = logging.getLogger(__name__)

@dataclass
class SynthesisConstraint:
    """合成约束"""
    constraint_type: str
    content: str
    priority: int = 1
    source: str = "cegis"

class CEGISSynthesizer:
    """
    CEGIS合成器
    实现反例引导的归纳程序合成
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.max_iterations = config.get('max_iterations', 25)
        self.synthesis_timeout = config.get('synthesis_timeout', 300)
        self.enable_parallel = config.get('enable_parallel', False)
        self.constraint_history = []
        self.synthesis_statistics = {
            'total_iterations': 0,
            'successful_synthesis': 0,
            'constraint_count': 0
        }
        
        logger.info("CEGIS合成器初始化完成")
    
    def synthesize_from_examples(self, examples: List[Tuple], 
                               background_knowledge: str,
                               bias_specification: str,
                               initial_constraints: List[str] = None) -> Optional[str]:
        """
        从示例进行CEGIS合成
        
        Args:
            examples: 训练示例对
            background_knowledge: 背景知识
            bias_specification: 偏置规范
            initial_constraints: 初始约束
            
        Returns:
            合成的程序或None
        """
        logger.info(f"开始CEGIS合成，示例数量: {len(examples)}")
        
        constraints = list(initial_constraints) if initial_constraints else []
        iteration = 0
        start_time = time.time()
        
        try:
            while iteration < self.max_iterations:
                if time.time() - start_time > self.synthesis_timeout:
                    logger.warning("CEGIS合成超时")
                    break
                
                logger.debug(f"CEGIS迭代 {iteration + 1}")
                
                # 生成候选程序
                candidate = self._generate_candidate_program(
                    examples, background_knowledge, bias_specification, constraints
                )
                
                if candidate is None:
                    logger.info("无法生成候选程序")
                    break
                
                # 验证候选程序
                verification_result = self._verify_candidate_comprehensive(candidate, examples)
                
                if verification_result['is_valid']:
                    logger.info(f"CEGIS合成成功，迭代次数: {iteration + 1}")
                    self.synthesis_statistics['successful_synthesis'] += 1
                    return candidate
                else:
                    # 生成新的约束
                    new_constraints = self._generate_constraints_from_failure(
                        candidate, verification_result
                    )
                    constraints.extend(new_constraints)
                    
                    logger.debug(f"添加 {len(new_constraints)} 个新约束")
                    self.synthesis_statistics['constraint_count'] += len(new_constraints)
                
                iteration += 1
                self.synthesis_statistics['total_iterations'] += 1
            
            logger.warning("CEGIS合成失败：达到最大迭代次数")
            return None
            
        except Exception as e:
            logger.error(f"CEGIS合成过程出错: {str(e)}")
            return None
    
    def _generate_candidate_program(self, examples: List[Tuple], 
                                  background_knowledge: str,
                                  bias_specification: str,
                                  constraints: List[str]) -> Optional[str]:
        """生成候选程序"""
        
        # 构建扩展的偏置规范
        extended_bias = self._extend_bias_with_constraints(bias_specification, constraints)
        
        # 使用多种策略生成候选
        strategies = ['simple', 'structured', 'pattern_based']
        
        for strategy in strategies:
            candidate = self._generate_with_strategy(
                examples, background_knowledge, extended_bias, strategy
            )
            
            if candidate:
                logger.debug(f"使用策略 '{strategy}' 生成候选程序")
                return candidate
        
        return None
    
    def _generate_with_strategy(self, examples: List[Tuple], 
                              background_knowledge: str,
                              bias: str, strategy: str) -> Optional[str]:
        """使用特定策略生成程序"""
        
        if strategy == 'simple':
            return self._generate_simple_program(examples)
        elif strategy == 'structured':
            return self._generate_structured_program(examples, background_knowledge)
        elif strategy == 'pattern_based':
            return self._generate_pattern_based_program(examples)
        
        return None
    
    def _generate_simple_program(self, examples: List[Tuple]) -> Optional[str]:
        """生成简单程序"""
        if not examples:
            return None
        
        # 分析示例寻找简单模式
        patterns = self._analyze_simple_patterns(examples)
        
        if patterns.get('color_mapping'):
            return self._generate_color_mapping_program(patterns['color_mapping'])
        
        if patterns.get('spatial_transformation'):
            return self._generate_spatial_program(patterns['spatial_transformation'])
        
        return None
    
    def _analyze_simple_patterns(self, examples: List[Tuple]) -> Dict[str, Any]:
        """分析简单模式"""
        patterns = {}
        
        # 检查颜色映射模式
        color_mapping = self._detect_color_mapping_pattern(examples)
        if color_mapping:
            patterns['color_mapping'] = color_mapping
        
        # 检查空间变换模式
        spatial_pattern = self._detect_spatial_pattern(examples)
        if spatial_pattern:
            patterns['spatial_transformation'] = spatial_pattern
        
        return patterns
    
    def _detect_color_mapping_pattern(self, examples: List[Tuple]) -> Optional[Dict[int, int]]:
        """检测颜色映射模式"""
        if not examples:
            return None
        
        # 检查是否所有示例都有相同的颜色映射
        first_mapping = None
        
        for input_grid, output_grid in examples:
            current_mapping = self._extract_color_mapping(input_grid, output_grid)
            
            if current_mapping is None:
                return None
            
            if first_mapping is None:
                first_mapping = current_mapping
            elif first_mapping != current_mapping:
                return None
        
        return first_mapping
    
    def _extract_color_mapping(self, input_grid: List[List[int]], 
                              output_grid: List[List[int]]) -> Optional[Dict[int, int]]:
        """从单个示例提取颜色映射"""
        if (len(input_grid) != len(output_grid) or 
            len(input_grid[0]) != len(output_grid[0])):
            return None
        
        mapping = {}
        
        for i in range(len(input_grid)):
            for j in range(len(input_grid[0])):
                input_color = input_grid[i][j]
                output_color = output_grid[i][j]
                
                if input_color in mapping:
                    if mapping[input_color] != output_color:
                        return None  # 不一致的映射
                else:
                    mapping[input_color] = output_color
        
        return mapping
    
    def _generate_color_mapping_program(self, color_mapping: Dict[int, int]) -> str:
        """生成颜色映射程序"""
        rules = []
        
        for old_color, new_color in color_mapping.items():
            if old_color != new_color:
                rules.append(f"change_color(Input, {old_color}, {new_color}, TempGrid)")
        
        if not rules:
            return "transform(Grid, Grid)."  # 恒等变换
        
        # 构建链式变换
        if len(rules) == 1:
            program = f"transform(Input, Output) :- {rules[0].replace('TempGrid', 'Output')}."
        else:
            program_lines = [
                "transform(Input, Output) :-"
            ]
            
            for i, rule in enumerate(rules):
                if i == 0:
                    program_lines.append(f"    {rule.replace('Input', 'Input').replace('TempGrid', 'Temp1')},")
                elif i == len(rules) - 1:
                    prev_temp = f"Temp{i}"
                    program_lines.append(f"    {rule.replace('Input', prev_temp).replace('TempGrid', 'Output')}.")
                else:
                    prev_temp = f"Temp{i}" 
                    next_temp = f"Temp{i+1}"
                    program_lines.append(f"    {rule.replace('Input', prev_temp).replace('TempGrid', next_temp)},")
            
            program = "\\n".join(program_lines)
        
        return program
    
    def _detect_spatial_pattern(self, examples: List[Tuple]) -> Optional[Dict[str, Any]]:
        """检测空间变换模式"""
        # 简化实现：检测简单的平移模式
        translations = []
        
        for input_grid, output_grid in examples:
            translation = self._detect_translation(input_grid, output_grid)
            if translation:
                translations.append(translation)
            else:
                return None  # 不是平移
        
        # 检查所有示例是否有相同的平移
        if translations and all(t == translations[0] for t in translations):
            return {'type': 'translation', 'offset': translations[0]}
        
        return None
    
    def _detect_translation(self, input_grid: List[List[int]], 
                           output_grid: List[List[int]]) -> Optional[Tuple[int, int]]:
        """检测平移变换"""
        if (len(input_grid) != len(output_grid) or 
            len(input_grid[0]) != len(output_grid[0])):
            return None
        
        # 寻找非零元素的移动
        input_nonzero = []
        output_nonzero = []
        
        for i in range(len(input_grid)):
            for j in range(len(input_grid[0])):
                if input_grid[i][j] != 0:
                    input_nonzero.append((i, j, input_grid[i][j]))
                if output_grid[i][j] != 0:
                    output_nonzero.append((i, j, output_grid[i][j]))
        
        if len(input_nonzero) != len(output_nonzero):
            return None
        
        # 尝试匹配并计算偏移
        if not input_nonzero:
            return (0, 0)  # 空网格
        
        # 简单启发式：使用第一个非零元素计算偏移
        input_pos = (input_nonzero[0][0], input_nonzero[0][1])
        
        for output_pos in output_nonzero:
            if output_pos[2] == input_nonzero[0][2]:  # 相同颜色
                offset = (output_pos[0] - input_pos[0], output_pos[1] - input_pos[1])
                
                # 验证所有元素都有相同偏移
                if self._verify_translation(input_grid, output_grid, offset):
                    return offset
        
        return None
    
    def _verify_translation(self, input_grid: List[List[int]], 
                           output_grid: List[List[int]], 
                           offset: Tuple[int, int]) -> bool:
        """验证平移假设"""
        rows, cols = len(input_grid), len(input_grid[0])
        dr, dc = offset
        
        for i in range(rows):
            for j in range(cols):
                new_i, new_j = i + dr, j + dc
                
                if 0 <= new_i < rows and 0 <= new_j < cols:
                    if input_grid[i][j] != output_grid[new_i][new_j]:
                        return False
                else:
                    # 超出边界的元素应该消失
                    if input_grid[i][j] != 0:
                        return False
        
        return True
    
    def _generate_spatial_program(self, spatial_pattern: Dict[str, Any]) -> str:
        """生成空间变换程序"""
        if spatial_pattern['type'] == 'translation':
            dr, dc = spatial_pattern['offset']
            return f"transform(Input, Output) :- translate_grid(Input, {dr}, {dc}, Output)."
        
        return "transform(Input, Input)."  # 默认恒等变换
    
    def _generate_structured_program(self, examples: List[Tuple], 
                                   background_knowledge: str) -> Optional[str]:
        """生成结构化程序"""
        # 基于背景知识中的可用谓词生成程序
        available_predicates = self._extract_predicates_from_bk(background_knowledge)
        
        # 尝试组合谓词生成程序
        for combination in self._generate_predicate_combinations(available_predicates):
            candidate = self._build_program_from_predicates(combination)
            if candidate:
                return candidate
        
        return None
    
    def _extract_predicates_from_bk(self, background_knowledge: str) -> List[str]:
        """从背景知识中提取可用谓词"""
        predicates = []
        
        for line in background_knowledge.split('\\n'):
            line = line.strip()
            if line and not line.startswith('%'):
                # 提取谓词名
                if ':-' in line:
                    head = line.split(':-')[0].strip()
                    if '(' in head:
                        pred_name = head.split('(')[0]
                        predicates.append(pred_name)
        
        return list(set(predicates))
    
    def _generate_predicate_combinations(self, predicates: List[str]) -> List[List[str]]:
        """生成谓词组合"""
        combinations = []
        
        # 单个谓词
        for pred in predicates:
            combinations.append([pred])
        
        # 两个谓词的组合
        for i, pred1 in enumerate(predicates):
            for j, pred2 in enumerate(predicates[i+1:], i+1):
                combinations.append([pred1, pred2])
        
        return combinations[:10]  # 限制组合数量
    
    def _build_program_from_predicates(self, predicates: List[str]) -> Optional[str]:
        """从谓词列表构建程序"""
        if not predicates:
            return None
        
        if len(predicates) == 1:
            pred = predicates[0]
            if pred == 'change_color':
                return "transform(Input, Output) :- change_color(Input, 1, 2, Output)."
            elif pred == 'translate_grid':
                return "transform(Input, Output) :- translate_grid(Input, 1, 0, Output)."
        
        # 多个谓词的组合（简化实现）
        body_parts = []
        for pred in predicates:
            if pred == 'change_color':
                body_parts.append("change_color(Input, 1, 2, Temp)")
            elif pred == 'translate_grid':
                body_parts.append("translate_grid(Temp, 0, 1, Output)")
        
        if body_parts:
            body = ", ".join(body_parts)
            return f"transform(Input, Output) :- {body}."
        
        return None
    
    def _generate_pattern_based_program(self, examples: List[Tuple]) -> Optional[str]:
        """基于模式生成程序"""
        # 分析示例中的高级模式
        patterns = self._analyze_high_level_patterns(examples)
        
        if patterns.get('fill_pattern'):
            return self._generate_fill_program(patterns['fill_pattern'])
        
        if patterns.get('reflection_pattern'):
            return self._generate_reflection_program(patterns['reflection_pattern'])
        
        return None
    
    def _analyze_high_level_patterns(self, examples: List[Tuple]) -> Dict[str, Any]:
        """分析高级模式"""
        patterns = {}
        
        # 检查填充模式
        if self._is_fill_pattern(examples):
            patterns['fill_pattern'] = True
        
        # 检查反射模式
        reflection = self._detect_reflection_pattern(examples)
        if reflection:
            patterns['reflection_pattern'] = reflection
        
        return patterns
    
    def _is_fill_pattern(self, examples: List[Tuple]) -> bool:
        """检查是否为填充模式"""
        for input_grid, output_grid in examples:
            # 简单检查：输出是否比输入有更多非零元素
            input_nonzero = sum(1 for row in input_grid for cell in row if cell != 0)
            output_nonzero = sum(1 for row in output_grid for cell in row if cell != 0)
            
            if output_nonzero <= input_nonzero:
                return False
        
        return True
    
    def _detect_reflection_pattern(self, examples: List[Tuple]) -> Optional[str]:
        """检测反射模式"""
        # 检查水平反射
        for input_grid, output_grid in examples:
            if len(input_grid) != len(output_grid):
                continue
            
            # 检查是否为水平翻转
            if all(output_grid[i] == input_grid[i][::-1] 
                   for i in range(len(input_grid))):
                return 'horizontal'
            
            # 检查是否为垂直翻转
            if output_grid == input_grid[::-1]:
                return 'vertical'
        
        return None
    
    def _generate_fill_program(self, pattern: bool) -> str:
        """生成填充程序"""
        return "transform(Input, Output) :- fill_holes(Input, Output)."
    
    def _generate_reflection_program(self, reflection_type: str) -> str:
        """生成反射程序"""
        if reflection_type == 'horizontal':
            return "transform(Input, Output) :- reflect_horizontal(Input, Output)."
        elif reflection_type == 'vertical':
            return "transform(Input, Output) :- reflect_vertical(Input, Output)."
        
        return "transform(Input, Input)."
    
    def _extend_bias_with_constraints(self, bias: str, constraints: List[str]) -> str:
        """用约束扩展偏置"""
        if not constraints:
            return bias
        
        extended_bias = bias + "\\n\\n% CEGIS约束\\n"
        
        for constraint in constraints:
            extended_bias += constraint + "\\n"
        
        return extended_bias
    
    def _verify_candidate_comprehensive(self, candidate: str, 
                                      examples: List[Tuple]) -> Dict[str, Any]:
        """全面验证候选程序"""
        result = {
            'is_valid': True,
            'failed_examples': [],
            'error_messages': [],
            'execution_errors': []
        }
        
        for i, (input_grid, expected_output) in enumerate(examples):
            try:
                # 执行程序（模拟）
                actual_output = self._execute_program_simulation(candidate, input_grid)
                
                if actual_output != expected_output:
                    result['is_valid'] = False
                    result['failed_examples'].append({
                        'index': i,
                        'input': input_grid,
                        'expected': expected_output,
                        'actual': actual_output
                    })
                    result['error_messages'].append(
                        f"示例{i+1}输出不匹配"
                    )
                
            except Exception as e:
                result['is_valid'] = False
                result['execution_errors'].append({
                    'index': i,
                    'error': str(e)
                })
        
        return result
    
    def _execute_program_simulation(self, program: str, input_grid: List[List[int]]) -> List[List[int]]:
        """程序执行模拟"""
        # 这是一个简化的程序执行模拟
        # 实际实现需要调用Prolog解释器
        
        if 'change_color' in program:
            # 模拟颜色变换
            if '1, 2' in program:  # 1->2的映射
                result = []
                for row in input_grid:
                    new_row = []
                    for cell in row:
                        if cell == 1:
                            new_row.append(2)
                        else:
                            new_row.append(cell)
                    result.append(new_row)
                return result
        
        # 默认返回输入（恒等变换）
        return [row[:] for row in input_grid]
    
    def _generate_constraints_from_failure(self, candidate: str, 
                                         verification_result: Dict[str, Any]) -> List[str]:
        """从失败中生成约束"""
        constraints = []
        
        # 基于失败的示例生成约束
        for failed_example in verification_result.get('failed_examples', []):
            constraint = self._create_constraint_from_failure(candidate, failed_example)
            if constraint:
                constraints.append(constraint)
        
        # 基于执行错误生成约束
        for exec_error in verification_result.get('execution_errors', []):
            constraint = self._create_constraint_from_error(candidate, exec_error)
            if constraint:
                constraints.append(constraint)
        
        return constraints
    
    def _create_constraint_from_failure(self, candidate: str, 
                                      failed_example: Dict[str, Any]) -> str:
        """从失败示例创建约束"""
        # 简化的约束生成
        index = failed_example['index']
        
        # 禁止产生错误输出的约束
        constraint = f":- program_output(example_{index}, WrongOutput), " \\
                    f"WrongOutput \\= correct_output_{index}."
        
        return constraint
    
    def _create_constraint_from_error(self, candidate: str, 
                                    exec_error: Dict[str, Any]) -> str:
        """从执行错误创建约束"""
        # 禁止导致执行错误的程序结构
        constraint = f":- program_structure_causes_error."
        
        return constraint
    
    def get_synthesis_statistics(self) -> Dict[str, Any]:
        """获取合成统计信息"""
        return self.synthesis_statistics.copy()
    
    def reset_statistics(self):
        """重置统计信息"""
        self.synthesis_statistics = {
            'total_iterations': 0,
            'successful_synthesis': 0,
            'constraint_count': 0
        }
'''

# =====================================================================
# 5. cegis/verifier.py - 程序验证器实现
# =====================================================================
CEGIS_VERIFIER_PY = '''"""
程序验证器 - 验证合成程序的正确性
"""
import logging
from typing import List, Tuple, Optional, Any, Dict
from dataclasses import dataclass
import subprocess
import tempfile
from pathlib import Path
import time

logger = logging.getLogger(__name__)

@dataclass
class VerificationResult:
    """验证结果"""
    is_valid: bool
    failed_example: Optional[Tuple[Any, Any]] = None
    error_message: Optional[str] = None
    execution_time: float = 0.0
    detailed_results: List[Dict] = None

class ProgramVerifier:
    """
    程序验证器
    支持多种验证方法和详细的错误报告
    """
    
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.timeout = self.config.get('timeout', 60)
        self.verification_method = self.config.get('method', 'simulation')
        self.enable_detailed_trace = self.config.get('detailed_trace', False)
        self.prolog_interpreter = self.config.get('prolog_interpreter', 'swipl')
        
        logger.info(f"程序验证器初始化: 方法={self.verification_method}")
    
    def verify_candidate(self, candidate: str, examples: List[Tuple]) -> VerificationResult:
        """
        验证候选程序对所有示例
        
        Args:
            candidate: 候选程序字符串
            examples: 测试示例列表
            
        Returns:
            VerificationResult: 验证结果
        """
        start_time = time.time()
        
        logger.debug(f"验证程序: {candidate[:100]}...")
        logger.debug(f"示例数量: {len(examples)}")
        
        try:
            detailed_results = []
            
            for i, (example_input, expected_output) in enumerate(examples):
                logger.debug(f"验证示例 {i+1}/{len(examples)}")
                
                # 执行程序
                execution_result = self._execute_program(candidate, example_input)
                
                if execution_result['success']:
                    actual_output = execution_result['output']
                    
                    # 比较输出
                    if self._outputs_match(actual_output, expected_output):
                        detailed_results.append({
                            'example_index': i,
                            'status': 'passed',
                            'input': example_input,
                            'expected': expected_output,
                            'actual': actual_output
                        })
                    else:
                        # 验证失败
                        execution_time = time.time() - start_time
                        
                        detailed_results.append({
                            'example_index': i,
                            'status': 'failed',
                            'input': example_input,
                            'expected': expected_output,
                            'actual': actual_output,
                            'error': 'Output mismatch'
                        })
                        
                        return VerificationResult(
                            is_valid=False,
                            failed_example=(example_input, expected_output),
                            error_message=f"示例{i+1}输出不匹配: 期望{expected_output}, 得到{actual_output}",
                            execution_time=execution_time,
                            detailed_results=detailed_results
                        )
                else:
                    # 执行错误
                    execution_time = time.time() - start_time
                    
                    detailed_results.append({
                        'example_index': i,
                        'status': 'error',
                        'input': example_input,
                        'expected': expected_output,
                        'error': execution_result['error']
                    })
                    
                    return VerificationResult(
                        is_valid=False,
                        failed_example=(example_input, expected_output),
                        error_message=f"示例{i+1}执行错误: {execution_result['error']}",
                        execution_time=execution_time,
                        detailed_results=detailed_results
                    )
            
            # 所有示例都通过
            execution_time = time.time() - start_time
            
            logger.debug(f"验证成功，用时 {execution_time:.3f}秒")
            
            return VerificationResult(
                is_valid=True,
                execution_time=execution_time,
                detailed_results=detailed_results
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"验证过程出错: {str(e)}")
            
            return VerificationResult(
                is_valid=False,
                error_message=f"验证过程出错: {str(e)}",
                execution_time=execution_time
            )
    
    def _execute_program(self, program: str, input_data: Any) -> Dict[str, Any]:
        """
        执行程序并返回结果
        
        Args:
            program: 程序字符串
            input_data: 输入数据
            
        Returns:
            包含执行结果的字典
        """
        if self.verification_method == 'simulation':
            return self._execute_by_simulation(program, input_data)
        elif self.verification_method == 'prolog':
            return self._execute_by_prolog(program, input_data)
        else:
            return {'success': False, 'error': f'未知验证方法: {self.verification_method}'}
    
    def _execute_by_simulation(self, program: str, input_data: Any) -> Dict[str, Any]:
        """通过模拟执行程序"""
        try:
            # 解析程序并执行
            result = self._simulate_program_execution(program, input_data)
            return {'success': True, 'output': result}
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _simulate_program_execution(self, program: str, input_grid: List[List[int]]) -> List[List[int]]:
        """模拟程序执行"""
        
        # 分析程序中的操作
        if 'change_color' in program or 'change_all_color' in program:
            return self._simulate_color_change(program, input_grid)
        
        elif 'translate' in program:
            return self._simulate_translation(program, input_grid)
        
        elif 'reflect' in program:
            return self._simulate_reflection(program, input_grid)
        
        elif 'fill' in program:
            return self._simulate_fill_operation(program, input_grid)
        
        else:
            # 默认恒等变换
            return [row[:] for row in input_grid]
    
    def _simulate_color_change(self, program: str, input_grid: List[List[int]]) -> List[List[int]]:
        """模拟颜色变换"""
        result = [row[:] for row in input_grid]
        
        # 简单的颜色映射检测
        import re
        
        # 查找颜色变换模式
        pattern = r'change.*?color.*?(\\d+).*?(\\d+)'
        matches = re.findall(pattern, program)
        
        if matches:
            for old_color_str, new_color_str in matches:
                old_color = int(old_color_str)
                new_color = int(new_color_str)
                
                # 应用颜色变换
                for i in range(len(result)):
                    for j in range(len(result[i])):
                        if result[i][j] == old_color:
                            result[i][j] = new_color
        
        return result
    
    def _simulate_translation(self, program: str, input_grid: List[List[int]]) -> List[List[int]]:
        """模拟平移变换"""
        import re
        
        # 查找平移参数
        pattern = r'translate.*?([-]?\\d+).*?([-]?\\d+)'
        matches = re.findall(pattern, program)
        
        if not matches:
            return [row[:] for row in input_grid]
        
        dr, dc = int(matches[0][0]), int(matches[0][1])
        
        rows, cols = len(input_grid), len(input_grid[0])
        result = [[0 for _ in range(cols)] for _ in range(rows)]
        
        for i in range(rows):
            for j in range(cols):
                new_i, new_j = i + dr, j + dc
                if 0 <= new_i < rows and 0 <= new_j < cols:
                    result[new_i][new_j] = input_grid[i][j]
        
        return result
    
    def _simulate_reflection(self, program: str, input_grid: List[List[int]]) -> List[List[int]]:
        """模拟反射变换"""
        if 'horizontal' in program:
            return [row[::-1] for row in input_grid]
        elif 'vertical' in program:
            return input_grid[::-1]
        else:
            return [row[:] for row in input_grid]
    
    def _simulate_fill_operation(self, program: str, input_grid: List[List[int]]) -> List[List[int]]:
        """模拟填充操作"""
        # 简化的填充实现：填充所有0为1
        result = []
        for row in input_grid:
            new_row = []
            for cell in row:
                if cell == 0:
                    new_row.append(1)  # 填充颜色
                else:
                    new_row.append(cell)
            result.append(new_row)
        
        return result
    
    def _execute_by_prolog(self, program: str, input_data: Any) -> Dict[str, Any]:
        """通过Prolog解释器执行程序"""
        try:
            # 创建临时文件
            with tempfile.NamedTemporaryFile(mode='w', suffix='.pl', delete=False) as temp_file:
                temp_file.write(self._prepare_prolog_program(program, input_data))
                temp_file_path = temp_file.name
            
            try:
                # 运行Prolog
                result = subprocess.run([
                    self.prolog_interpreter,
                    '-g', 'main',
                    '-t', 'halt',
                    temp_file_path
                ], capture_output=True, text=True, timeout=self.timeout)
                
                if result.returncode == 0:
                    output = self._parse_prolog_output(result.stdout)
                    return {'success': True, 'output': output}
                else:
                    return {'success': False, 'error': result.stderr}
                    
            finally:
                # 清理临时文件
                Path(temp_file_path).unlink(missing_ok=True)
                
        except subprocess.TimeoutExpired:
            return {'success': False, 'error': 'Prolog执行超时'}
        except Exception as e:
            return {'success': False, 'error': f'Prolog执行错误: {str(e)}'}
    
    def _prepare_prolog_program(self, program: str, input_data: Any) -> str:
        """准备用于执行的Prolog程序"""
        # 构建完整的Prolog程序
        prolog_code = f"""
% 用户程序
{program}

% 输入数据
input_data({self._data_to_prolog_term(input_data)}).

% 主执行谓词
main :-
    input_data(Input),
    transform(Input, Output),
    write_canonical(Output),
    nl.
"""
        
        return prolog_code
    
    def _data_to_prolog_term(self, data: Any) -> str:
        """将Python数据转换为Prolog项"""
        if isinstance(data, list):
            if data and isinstance(data[0], list):
                # 2D网格
                rows = []
                for row in data:
                    row_cells = [f"cell({i}, {j}, {cell})" 
                                for j, cell in enumerate(row)]
                    rows.extend(row_cells)
                return f"grid([{', '.join(rows)}])"
            else:
                # 1D列表
                return f"[{', '.join(str(item) for item in data)}]"
        else:
            return str(data)
    
    def _parse_prolog_output(self, output: str) -> Any:
        """解析Prolog输出"""
        # 简化的输出解析
        lines = output.strip().split('\\n')
        for line in lines:
            if line.startswith('grid('):
                return self._parse_grid_term(line)
        
        return None
    
    def _parse_grid_term(self, term: str) -> List[List[int]]:
        """解析网格项"""
        # 这是一个简化的解析器
        # 实际实现需要更复杂的Prolog项解析
        
        # 提取单元格信息
        import re
        cell_pattern = r'cell\\((\\d+),\\s*(\\d+),\\s*(\\d+)\\)'
        cells = re.findall(cell_pattern, term)
        
        if not cells:
            return []
        
        # 构建网格
        max_row = max(int(cell[0]) for cell in cells)
        max_col = max(int(cell[1]) for cell in cells)
        
        grid = [[0 for _ in range(max_col + 1)] for _ in range(max_row + 1)]
        
        for row_str, col_str, color_str in cells:
            row, col, color = int(row_str), int(col_str), int(color_str)
            grid[row][col] = color
        
        return grid
    
    def _outputs_match(self, actual: Any, expected: Any) -> bool:
        """检查输出是否匹配"""
        if type(actual) != type(expected):
            return False
        
        if isinstance(actual, list):
            if len(actual) != len(expected):
                return False
            
            for a, e in zip(actual, expected):
                if not self._outputs_match(a, e):
                    return False
            
            return True
        else:
            return actual == expected
    
    def verify_program_syntax(self, program: str) -> Dict[str, Any]:
        """验证程序语法"""
        result = {
            'is_valid': True,
            'errors': [],
            'warnings': []
        }
        
        try:
            # 基本语法检查
            lines = program.split('\\n')
            
            for i, line in enumerate(lines, 1):
                line = line.strip()
                if not line or line.startswith('%'):
                    continue
                
                # 检查是否以点结尾
                if not line.endswith('.'):
                    result['errors'].append(f"第{i}行: 缺少结尾的点")
                    result['is_valid'] = False
                
                # 检查括号匹配
                if line.count('(') != line.count(')'):
                    result['errors'].append(f"第{i}行: 括号不匹配")
                    result['is_valid'] = False
                
                # 检查基本结构
                if ':-' in line:
                    parts = line.split(':-')
                    if len(parts) != 2:
                        result['errors'].append(f"第{i}行: 规则格式错误")
                        result['is_valid'] = False
        
        except Exception as e:
            result['is_valid'] = False
            result['errors'].append(f"语法检查出错: {str(e)}")
        
        return result
    
    def get_verification_trace(self, program: str, input_data: Any) -> List[Dict[str, Any]]:
        """获取验证跟踪信息"""
        if not self.enable_detailed_trace:
            return []
        
        trace = []
        
        try:
            # 记录执行步骤（简化实现）
            trace.append({
                'step': 'start',
                'action': 'begin_execution',
                'data': {'input': input_data}
            })
            
            # 执行程序
            result = self._execute_program(program, input_data)
            
            trace.append({
                'step': 'execution',
                'action': 'program_execution',
                'data': result
            })
            
            trace.append({
                'step': 'end',
                'action': 'execution_complete',
                'data': {'success': result['success']}
            })
            
        except Exception as e:
            trace.append({
                'step': 'error',
                'action': 'execution_error',
                'data': {'error': str(e)}
            })
        
        return trace
'''

# =====================================================================
# 6. cegis/counterexample.py - 反例生成器实现
# =====================================================================
CEGIS_COUNTEREXAMPLE_PY = '''"""
反例生成器 - 从失败的验证中生成有用的约束
"""
import logging
from typing import Any, Tuple, List, Dict, Optional
import random

logger = logging.getLogger(__name__)

class CounterexampleGenerator:
    """
    反例生成器
    从程序验证失败中生成有意义的约束
    """
    
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.constraint_types = self.config.get('constraint_types', [
            'output_constraint', 'input_constraint', 'structure_constraint'
        ])
        self.max_constraints_per_failure = self.config.get('max_constraints_per_failure', 3)
        self.constraint_specificity = self.config.get('constraint_specificity', 'medium')
        
        logger.info("反例生成器初始化完成")
    
    def generate(self, failed_program: str, failed_example: Tuple[Any, Any]) -> str:
        """
        从失败的程序和示例生成反例约束
        
        Args:
            failed_program: 失败的程序
            failed_example: 失败的示例 (input, expected_output)
            
        Returns:
            生成的约束字符串
        """
        example_input, expected_output = failed_example
        
        logger.debug(f"为失败程序生成反例约束")
        logger.debug(f"程序: {failed_program[:100]}...")
        
        try:
            # 分析失败原因
            failure_analysis = self._analyze_failure(failed_program, example_input, expected_output)
            
            # 根据失败原因生成约束
            constraints = self._generate_constraints_from_analysis(failure_analysis)
            
            # 选择最有效的约束
            selected_constraint = self._select_best_constraint(constraints)
            
            logger.debug(f"生成约束: {selected_constraint}")
            
            return selected_constraint
            
        except Exception as e:
            logger.error(f"反例生成失败: {str(e)}")
            return self._generate_default_constraint(failed_program, failed_example)
    
    def generate_multiple(self, failed_program: str, 
                         failed_examples: List[Tuple[Any, Any]]) -> List[str]:
        """
        从多个失败示例生成约束
        
        Args:
            failed_program: 失败的程序
            failed_examples: 失败的示例列表
            
        Returns:
            约束字符串列表
        """
        constraints = []
        
        for example in failed_examples:
            constraint = self.generate(failed_program, example)
            if constraint and constraint not in constraints:
                constraints.append(constraint)
        
        # 生成跨示例的约束
        cross_example_constraints = self._generate_cross_example_constraints(
            failed_program, failed_examples
        )
        
        constraints.extend(cross_example_constraints)
        
        return constraints[:self.max_constraints_per_failure]
    
    def _analyze_failure(self, program: str, input_data: Any, 
                        expected_output: Any) -> Dict[str, Any]:
        """分析程序失败的原因"""
        analysis = {
            'failure_type': 'unknown',
            'program_structure': self._analyze_program_structure(program),
            'input_properties': self._analyze_input_properties(input_data),
            'output_properties': self._analyze_output_properties(expected_output),
            'mismatch_details': {}
        }
        
        # 尝试执行程序以了解实际输出
        try:
            actual_output = self._simulate_execution(program, input_data)
            analysis['actual_output'] = actual_output
            analysis['mismatch_details'] = self._analyze_output_mismatch(
                actual_output, expected_output
            )
            analysis['failure_type'] = self._classify_failure_type(analysis)
        except Exception as e:
            analysis['execution_error'] = str(e)
            analysis['failure_type'] = 'execution_error'
        
        return analysis
    
    def _analyze_program_structure(self, program: str) -> Dict[str, Any]:
        """分析程序结构"""
        structure = {
            'predicates_used': [],
            'rule_count': 0,
            'has_recursion': False,
            'complexity_score': 0
        }
        
        lines = [line.strip() for line in program.split('\\n') 
                if line.strip() and not line.strip().startswith('%')]
        
        structure['rule_count'] = len(lines)
        
        # 提取使用的谓词
        import re
        predicate_pattern = r'([a-z][a-zA-Z0-9_]*)\\s*\\('
        
        for line in lines:
            predicates = re.findall(predicate_pattern, line)
            structure['predicates_used'].extend(predicates)
        
        structure['predicates_used'] = list(set(structure['predicates_used']))
        
        # 检查递归
        for line in lines:
            if ':-' in line:
                head = line.split(':-')[0].strip()
                body = line.split(':-')[1].strip()
                head_pred = re.search(r'^([a-z][a-zA-Z0-9_]*)', head)
                if head_pred and head_pred.group(1) in body:
                    structure['has_recursion'] = True
                    break
        
        # 计算复杂度分数
        structure['complexity_score'] = (
            len(structure['predicates_used']) * 2 + 
            structure['rule_count'] + 
            (5 if structure['has_recursion'] else 0)
        )
        
        return structure
    
    def _analyze_input_properties(self, input_data: Any) -> Dict[str, Any]:
        """分析输入数据属性"""
        properties = {
            'type': type(input_data).__name__,
            'size': 0,
            'unique_values': [],
            'pattern': 'unknown'
        }
        
        if isinstance(input_data, list):
            properties['size'] = len(input_data)
            
            if input_data and isinstance(input_data[0], list):
                # 2D网格
                properties['type'] = 'grid'
                properties['dimensions'] = (len(input_data), len(input_data[0]))
                
                # 收集所有值
                all_values = []
                for row in input_data:
                    all_values.extend(row)
                
                properties['unique_values'] = list(set(all_values))
                properties['background_ratio'] = all_values.count(0) / len(all_values) if all_values else 0
                
                # 检测简单模式
                if self._is_uniform_grid(input_data):
                    properties['pattern'] = 'uniform'
                elif self._has_symmetry(input_data):
                    properties['pattern'] = 'symmetric'
                else:
                    properties['pattern'] = 'irregular'
        
        return properties
    
    def _analyze_output_properties(self, output_data: Any) -> Dict[str, Any]:
        """分析输出数据属性"""
        # 与输入属性分析类似
        return self._analyze_input_properties(output_data)
    
    def _simulate_execution(self, program: str, input_data: Any) -> Any:
        """模拟程序执行"""
        # 简化的程序执行模拟
        if isinstance(input_data, list) and input_data and isinstance(input_data[0], list):
            # 2D网格处理
            if 'change_color' in program:
                return self._simulate_color_change(program, input_data)
            elif 'translate' in program:
                return self._simulate_translation(program, input_data)
            elif 'reflect' in program:
                return self._simulate_reflection(program, input_data)
        
        # 默认返回输入
        return input_data
    
    def _simulate_color_change(self, program: str, grid: List[List[int]]) -> List[List[int]]:
        """模拟颜色变换"""
        import re
        
        result = [row[:] for row in grid]
        
        # 查找颜色变换模式
        color_patterns = [
            r'change.*?color.*?(\\d+).*?(\\d+)',
            r'(\\d+).*?->.*?(\\d+)',
            r'(\\d+).*?to.*?(\\d+)'
        ]
        
        for pattern in color_patterns:
            matches = re.findall(pattern, program)
            if matches:
                old_color, new_color = int(matches[0][0]), int(matches[0][1])
                
                for i in range(len(result)):
                    for j in range(len(result[i])):
                        if result[i][j] == old_color:
                            result[i][j] = new_color
                break
        
        return result
    
    def _simulate_translation(self, program: str, grid: List[List[int]]) -> List[List[int]]:
        """模拟平移变换"""
        import re
        
        # 查找平移参数
        translate_patterns = [
            r'translate.*?([-]?\\d+).*?([-]?\\d+)',
            r'move.*?([-]?\\d+).*?([-]?\\d+)',
            r'shift.*?([-]?\\d+).*?([-]?\\d+)'
        ]
        
        dr, dc = 0, 0
        
        for pattern in translate_patterns:
            matches = re.findall(pattern, program)
            if matches:
                dr, dc = int(matches[0][0]), int(matches[0][1])
                break
        
        # 应用平移
        rows, cols = len(grid), len(grid[0])
        result = [[0 for _ in range(cols)] for _ in range(rows)]
        
        for i in range(rows):
            for j in range(cols):
                new_i, new_j = i + dr, j + dc
                if 0 <= new_i < rows and 0 <= new_j < cols:
                    result[new_i][new_j] = grid[i][j]
        
        return result
    
    def _simulate_reflection(self, program: str, grid: List[List[int]]) -> List[List[int]]:
        """模拟反射变换"""
        if 'horizontal' in program or 'left' in program or 'right' in program:
            return [row[::-1] for row in grid]
        elif 'vertical' in program or 'up' in program or 'down' in program:
            return grid[::-1]
        else:
            return [row[:] for row in grid]
    
    def _analyze_output_mismatch(self, actual: Any, expected: Any) -> Dict[str, Any]:
        """分析输出不匹配的详细信息"""
        mismatch = {
            'type': 'unknown',
            'details': {},
            'severity': 'high'
        }
        
        if type(actual) != type(expected):
            mismatch['type'] = 'type_mismatch'
            mismatch['details'] = {
                'actual_type': type(actual).__name__,
                'expected_type': type(expected).__name__
            }
            return mismatch
        
        if isinstance(actual, list) and isinstance(expected, list):
            if len(actual) != len(expected):
                mismatch['type'] = 'size_mismatch'
                mismatch['details'] = {
                    'actual_size': len(actual),
                    'expected_size': len(expected)
                }
            else:
                # 详细的元素比较
                differences = []
                
                if actual and isinstance(actual[0], list):
                    # 2D网格比较
                    for i in range(len(actual)):
                        for j in range(len(actual[i])):
                            if i < len(expected) and j < len(expected[i]):
                                if actual[i][j] != expected[i][j]:
                                    differences.append({
                                        'position': (i, j),
                                        'actual': actual[i][j],
                                        'expected': expected[i][j]
                                    })
                else:
                    # 1D列表比较
                    for i in range(len(actual)):
                        if actual[i] != expected[i]:
                            differences.append({
                                'position': i,
                                'actual': actual[i],
                                'expected': expected[i]
                            })
                
                mismatch['type'] = 'value_mismatch'
                mismatch['details'] = {
                    'difference_count': len(differences),
                    'differences': differences[:10]  # 限制显示数量
                }
                
                # 计算不匹配严重程度
                total_elements = len(actual) * len(actual[0]) if actual and isinstance(actual[0], list) else len(actual)
                mismatch['severity'] = 'high' if len(differences) > total_elements * 0.5 else 'medium'
        
        return mismatch
    
    def _classify_failure_type(self, analysis: Dict[str, Any]) -> str:
        """分类失败类型"""
        mismatch = analysis.get('mismatch_details', {})
        
        if 'execution_error' in analysis:
            return 'execution_error'
        
        mismatch_type = mismatch.get('type', 'unknown')
        
        if mismatch_type == 'type_mismatch':
            return 'type_error'
        elif mismatch_type == 'size_mismatch':
            return 'size_error'
        elif mismatch_type == 'value_mismatch':
            details = mismatch.get('details', {})
            diff_count = details.get('difference_count', 0)
            
            if diff_count == 0:
                return 'no_error'  # 这不应该发生
            elif diff_count <= 3:
                return 'minor_value_error'
            else:
                return 'major_value_error'
        
        return 'unknown_error'
    
    def _generate_constraints_from_analysis(self, analysis: Dict[str, Any]) -> List[str]:
        """根据失败分析生成约束"""
        constraints = []
        failure_type = analysis.get('failure_type', 'unknown')
        
        if failure_type == 'execution_error':
            constraints.extend(self._generate_execution_error_constraints(analysis))
        elif failure_type in ['minor_value_error', 'major_value_error']:
            constraints.extend(self._generate_value_error_constraints(analysis))
        elif failure_type == 'size_error':
            constraints.extend(self._generate_size_error_constraints(analysis))
        elif failure_type == 'type_error':
            constraints.extend(self._generate_type_error_constraints(analysis))
        
        # 生成通用约束
        constraints.extend(self._generate_general_constraints(analysis))
        
        return constraints
    
    def _generate_execution_error_constraints(self, analysis: Dict[str, Any]) -> List[str]:
        """生成执行错误约束"""
        constraints = []
        
        # 禁止导致执行错误的程序结构
        error_msg = analysis.get('execution_error', '')
        
        if 'undefined' in error_msg.lower():
            constraints.append(":- program_uses_undefined_predicate.")
        
        if 'syntax' in error_msg.lower():
            constraints.append(":- program_has_syntax_error.")
        
        # 通用执行错误约束
        constraints.append(":- program_causes_execution_error.")
        
        return constraints
    
    def _generate_value_error_constraints(self, analysis: Dict[str, Any]) -> List[str]:
        """生成值错误约束"""
        constraints = []
        
        mismatch = analysis.get('mismatch_details', {})
        details = mismatch.get('details', {})
        differences = details.get('differences', [])
        
        # 基于具体差异生成约束
        for diff in differences[:3]:  # 限制约束数量
            if 'position' in diff:
                pos = diff['position']
                actual = diff['actual']
                expected = diff['expected']
                
                if isinstance(pos, tuple):
                    # 2D位置
                    constraint = f":- transform_produces_value_at({pos[0]}, {pos[1]}, {actual}), " \\
                               f"expected_value_at({pos[0]}, {pos[1]}, {expected})."
                else:
                    # 1D位置
                    constraint = f":- transform_produces_value_at({pos}, {actual}), " \\
                               f"expected_value_at({pos}, {expected})."
                
                constraints.append(constraint)
        
        return constraints
    
    def _generate_size_error_constraints(self, analysis: Dict[str, Any]) -> List[str]:
        """生成大小错误约束"""
        constraints = []
        
        mismatch = analysis.get('mismatch_details', {})
        details = mismatch.get('details', {})
        
        actual_size = details.get('actual_size')
        expected_size = details.get('expected_size')
        
        if actual_size is not None and expected_size is not None:
            constraints.append(f":- output_size({actual_size}), required_size({expected_size}).")
        
        return constraints
    
    def _generate_type_error_constraints(self, analysis: Dict[str, Any]) -> List[str]:
        """生成类型错误约束"""
        constraints = []
        
        mismatch = analysis.get('mismatch_details', {})
        details = mismatch.get('details', {})
        
        actual_type = details.get('actual_type')
        expected_type = details.get('expected_type')
        
        if actual_type and expected_type:
            constraints.append(f":- output_type({actual_type}), required_type({expected_type}).")
        
        return constraints
    
    def _generate_general_constraints(self, analysis: Dict[str, Any]) -> List[str]:
        """生成通用约束"""
        constraints = []
        
        # 基于程序结构的约束
        structure = analysis.get('program_structure', {})
        predicates = structure.get('predicates_used', [])
        
        # 如果程序过于复杂，添加简化约束
        complexity = structure.get('complexity_score', 0)
        if complexity > 20:
            constraints.append(":- program_complexity_too_high.")
        
        # 基于使用的谓词添加约束
        if 'change_color' in predicates:
            constraints.append(":- uses_change_color, produces_incorrect_color_mapping.")
        
        if 'translate' in predicates:
            constraints.append(":- uses_translate, produces_incorrect_translation.")
        
        return constraints
    
    def _generate_cross_example_constraints(self, program: str, 
                                          failed_examples: List[Tuple[Any, Any]]) -> List[str]:
        """生成跨示例约束"""
        constraints = []
        
        if len(failed_examples) < 2:
            return constraints
        
        # 查找所有示例中的共同失败模式
        failure_patterns = []
        
        for example in failed_examples:
            analysis = self._analyze_failure(program, example[0], example[1])
            failure_patterns.append(analysis.get('failure_type', 'unknown'))
        
        # 如果所有示例都有相同的失败类型
        if len(set(failure_patterns)) == 1:
            failure_type = failure_patterns[0]
            constraints.append(f":- consistently_fails_with_{failure_type}.")
        
        return constraints
    
    def _select_best_constraint(self, constraints: List[str]) -> str:
        """选择最佳约束"""
        if not constraints:
            return ":- program_fails_verification."
        
        # 简单的启发式：选择最具体的约束
        constraint_scores = []
        
        for constraint in constraints:
            score = self._score_constraint(constraint)
            constraint_scores.append((constraint, score))
        
        # 选择分数最高的约束
        best_constraint = max(constraint_scores, key=lambda x: x[1])[0]
        
        return best_constraint
    
    def _score_constraint(self, constraint: str) -> float:
        """给约束打分"""
        score = 0.0
        
        # 更具体的约束得分更高
        if 'position' in constraint:
            score += 3.0
        
        if 'value' in constraint:
            score += 2.0
        
        if 'type' in constraint:
            score += 1.5
        
        if 'size' in constraint:
            score += 1.0
        
        # 避免过于通用的约束
        if 'program_fails' in constraint:
            score -= 1.0
        
        return score
    
    def _generate_default_constraint(self, failed_program: str, 
                                   failed_example: Tuple[Any, Any]) -> str:
        """生成默认约束"""
        return ":- program_produces_incorrect_output."
    
    # 辅助方法
    def _is_uniform_grid(self, grid: List[List[int]]) -> bool:
        """检查网格是否均匀"""
        if not grid or not grid[0]:
            return True
        
        first_value = grid[0][0]
        for row in grid:
            for cell in row:
                if cell != first_value:
                    return False
        
        return True
    
    def _has_symmetry(self, grid: List[List[int]]) -> bool:
        """检查网格是否有对称性"""
        import numpy as np
        
        np_grid = np.array(grid)
        
        # 检查水平对称
        if np.array_equal(np_grid, np.fliplr(np_grid)):
            return True
        
        # 检查垂直对称
        if np.array_equal(np_grid, np.flipud(np_grid)):
            return True
        
        return False
'''

# =====================================================================
# 7. utils/arc_loader.py - ARC数据加载器实现
# =====================================================================
UTILS_ARC_LOADER_PY = '''"""
ARC数据加载器 - 处理ARC数据集的加载和预处理
"""
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import random
import numpy as np
from ..core.synthesis_engine import SynthesisTask

logger = logging.getLogger(__name__)

class ARCDataLoader:
    """
    ARC数据集加载器
    支持标准ARC格式和自定义任务格式
    """
    
    def __init__(self, data_path: str = "data/arc"):
        self.data_path = Path(data_path)
        self.cache = {}
        self.task_metadata = {}
        
        logger.info(f"ARC数据加载器初始化: {self.data_path}")
    
    def load_task(self, task_id: str) -> SynthesisTask:
        """
        加载单个ARC任务
        
        Args:
            task_id: 任务标识符
            
        Returns:
            SynthesisTask对象
        """
        # 检查缓存
        if task_id in self.cache:
            logger.debug(f"从缓存加载任务: {task_id}")
            return self.cache[task_id]
        
        # 尝试不同的文件路径
        possible_paths = [
            self.data_path / f"{task_id}.json",
            self.data_path / "training" / f"{task_id}.json",
            self.data_path / "evaluation" / f"{task_id}.json",
            self.data_path / "test" / f"{task_id}.json"
        ]
        
        task_file = None
        for path in possible_paths:
            if path.exists():
                task_file = path
                break
        
        if not task_file:
            raise FileNotFoundError(f"任务文件未找到: {task_id}")
        
        logger.debug(f"从文件加载任务: {task_file}")
        
        try:
            with open(task_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            task = self._parse_arc_data(task_id, data)
            
            # 缓存结果
            self.cache[task_id] = task
            
            return task
            
        except Exception as e:
            logger.error(f"加载任务 {task_id} 失败: {str(e)}")
            raise
    
    def load_all_tasks(self, subset: str = "all") -> List[SynthesisTask]:
        """
        加载所有ARC任务
        
        Args:
            subset: 子集名称 ("training", "evaluation", "test", "all")
            
        Returns:
            任务列表
        """
        logger.info(f"加载任务子集: {subset}")
        
        tasks = []
        
        if subset == "all":
            search_dirs = [self.data_path]
        else:
            search_dirs = [self.data_path / subset]
        
        for search_dir in search_dirs:
            if not search_dir.exists():
                logger.warning(f"目录不存在: {search_dir}")
                continue
            
            # 递归搜索JSON文件
            for task_file in search_dir.rglob("*.json"):
                try:
                    task_id = task_file.stem
                    task = self.load_task(task_id)
                    tasks.append(task)
                    
                except Exception as e:
                    logger.warning(f"跳过无效任务文件 {task_file}: {str(e)}")
        
        logger.info(f"成功加载 {len(tasks)} 个任务")
        return tasks
    
    def _parse_arc_data(self, task_id: str, data: Dict[str, Any]) -> SynthesisTask:
        """解析ARC数据格式"""
        
        # 标准ARC格式
        if 'train' in data and 'test' in data:
            train_pairs = []
            for example in data['train']:
                train_pairs.append((example['input'], example['output']))
            
            test_pairs = []
            for example in data['test']:
                test_pairs.append((example['input'], example['output']))
            
            # 提取元数据
            metadata = {
                'source': 'arc_dataset',
                'train_count': len(train_pairs),
                'test_count': len(test_pairs)
            }
            
            # 添加任务分析
            metadata.update(self._analyze_task_properties(train_pairs, test_pairs))
            
        # 自定义格式
        elif 'train_pairs' in data:
            train_pairs = data['train_pairs']
            test_pairs = data.get('test_pairs', [])
            metadata = data.get('metadata', {})
            
        else:
            raise ValueError(f"不支持的数据格式: {task_id}")
        
        return SynthesisTask(
            task_id=task_id,
            train_pairs=train_pairs,
            test_pairs=test_pairs,
            metadata=metadata
        )
    
    def _analyze_task_properties(self, train_pairs: List[Tuple], 
                                test_pairs: List[Tuple]) -> Dict[str, Any]:
        """分析任务属性"""
        properties = {
            'task_type': 'unknown',
            'difficulty': 'medium',
            'grid_sizes': {
                'input': [],
                'output': []
            },
            'colors_used': set(),
            'transformation_hints': []
        }
        
        try:
            # 分析网格大小
            for input_grid, output_grid in train_pairs:
                properties['grid_sizes']['input'].append((len(input_grid), len(input_grid[0])))
                properties['grid_sizes']['output'].append((len(output_grid), len(output_grid[0])))
                
                # 收集使用的颜色
                for row in input_grid:
                    properties['colors_used'].update(row)
                for row in output_grid:
                    properties['colors_used'].update(row)
            
            properties['colors_used'] = list(properties['colors_used'])
            
            # 推断任务类型
            properties['task_type'] = self._infer_task_type(train_pairs)
            
            # 评估难度
            properties['difficulty'] = self._assess_difficulty(train_pairs, properties)
            
            # 生成转换提示
            properties['transformation_hints'] = self._generate_transformation_hints(train_pairs)
            
        except Exception as e:
            logger.warning(f"任务属性分析失败: {str(e)}")
        
        return properties
    
    def _infer_task_type(self, train_pairs: List[Tuple]) -> str:
        """推断任务类型"""
        if not train_pairs:
            return 'unknown'
        
        # 检查大小变化
        size_changes = []
        color_changes = []
        
        for input_grid, output_grid in train_pairs:
            input_size = (len(input_grid), len(input_grid[0]))
            output_size = (len(output_grid), len(output_grid[0]))
            
            size_changes.append(input_size != output_size)
            
            # 检查颜色变化
            input_colors = set()
            output_colors = set()
            
            for row in input_grid:
                input_colors.update(row)
            for row in output_grid:
                output_colors.update(row)
            
            color_changes.append(input_colors != output_colors)
        
        # 分类任务类型
        if all(size_changes):
            return 'size_transformation'
        elif any(color_changes) and not any(size_changes):
            return 'color_transformation'
        elif self._is_spatial_transformation(train_pairs):
            return 'spatial_transformation'
        elif self._is_pattern_completion(train_pairs):
            return 'pattern_completion'
        else:
            return 'complex_transformation'
    
    def _is_spatial_transformation(self, train_pairs: List[Tuple]) -> bool:
        """检查是否为空间变换"""
        for input_grid, output_grid in train_pairs:
            if len(input_grid) != len(output_grid) or len(input_grid[0]) != len(output_grid[0]):
                return False
            
            # 检查是否为简单的空间变换（旋转、翻转等）
            input_array = np.array(input_grid)
            output_array = np.array(output_grid)
            
            # 检查旋转
            for k in range(1, 4):
                if np.array_equal(output_array, np.rot90(input_array, k)):
                    return True
            
            # 检查翻转
            if (np.array_equal(output_array, np.fliplr(input_array)) or
                np.array_equal(output_array, np.flipud(input_array))):
                return True
        
        return False
    
    def _is_pattern_completion(self, train_pairs: List[Tuple]) -> bool:
        """检查是否为模式补全"""
        for input_grid, output_grid in train_pairs:
            # 简单检查：输出是否比输入有更多非零元素
            input_nonzero = sum(1 for row in input_grid for cell in row if cell != 0)
            output_nonzero = sum(1 for row in output_grid for cell in row if cell != 0)
            
            if output_nonzero <= input_nonzero:
                return False
        
        return True
    
    def _assess_difficulty(self, train_pairs: List[Tuple], properties: Dict[str, Any]) -> str:
        """评估任务难度"""
        difficulty_score = 0
        
        # 基于训练示例数量
        if len(train_pairs) <= 2:
            difficulty_score += 2
        elif len(train_pairs) >= 5:
            difficulty_score -= 1
        
        # 基于网格大小
        max_size = 0
        for sizes in properties['grid_sizes']['input']:
            max_size = max(max_size, sizes[0] * sizes[1])
        
        if max_size >= 100:
            difficulty_score += 2
        elif max_size >= 50:
            difficulty_score += 1
        
        # 基于颜色数量
        color_count = len(properties['colors_used'])
        if color_count >= 8:
            difficulty_score += 2
        elif color_count >= 5:
            difficulty_score += 1
        
        # 基于任务类型
        task_type = properties['task_type']
        if task_type == 'complex_transformation':
            difficulty_score += 3
        elif task_type in ['spatial_transformation', 'pattern_completion']:
            difficulty_score += 1
        
        # 转换为难度等级
        if difficulty_score <= 1:
            return 'easy'
        elif difficulty_score <= 4:
            return 'medium'
        else:
            return 'hard'
    
    def _generate_transformation_hints(self, train_pairs: List[Tuple]) -> List[str]:
        """生成转换提示"""
        hints = []
        
        if not train_pairs:
            return hints
        
        # 检查常见模式
        for input_grid, output_grid in train_pairs:
            # 颜色映射提示
            color_mapping = self._extract_color_mapping(input_grid, output_grid)
            if color_mapping:
                for old_color, new_color in color_mapping.items():
                    if old_color != new_color:
                        hints.append(f"color_{old_color}_becomes_{new_color}")
            
            # 大小变化提示
            input_size = (len(input_grid), len(input_grid[0]))
            output_size = (len(output_grid), len(output_grid[0]))
            
            if input_size != output_size:
                hints.append(f"size_change_{input_size}_to_{output_size}")
            
            # 对称性提示
            if self._has_symmetry(output_grid):
                hints.append("output_has_symmetry")
        
        return list(set(hints))  # 去重
    
    def _extract_color_mapping(self, input_grid: List[List[int]], 
                              output_grid: List[List[int]]) -> Optional[Dict[int, int]]:
        """提取颜色映射"""
        if len(input_grid) != len(output_grid) or len(input_grid[0]) != len(output_grid[0]):
            return None
        
        mapping = {}
        
        for i in range(len(input_grid)):
            for j in range(len(input_grid[0])):
                input_color = input_grid[i][j]
                output_color = output_grid[i][j]
                
                if input_color in mapping:
                    if mapping[input_color] != output_color:
                        return None  # 不一致的映射
                else:
                    mapping[input_color] = output_color
        
        return mapping
    
    def _has_symmetry(self, grid: List[List[int]]) -> bool:
        """检查网格对称性"""
        array = np.array(grid)
        
        # 水平对称
        if np.array_equal(array, np.fliplr(array)):
            return True
        
        # 垂直对称
        if np.array_equal(array, np.flipud(array)):
            return True
        
        return False
    
    def create_simple_task(self, task_id: str = "simple_color_change") -> SynthesisTask:
        """创建简单的测试任务"""
        
        train_pairs = [
            # 简单颜色转换：1->2
            ([[0, 1, 0], [1, 1, 1], [0, 1, 0]], 
             [[0, 2, 0], [2, 2, 2], [0, 2, 0]]),
            ([[0, 1, 1], [1, 0, 1], [1, 1, 0]], 
             [[0, 2, 2], [2, 0, 2], [2, 2, 0]]),
            ([[1, 0, 0], [0, 1, 0], [0, 0, 1]], 
             [[2, 0, 0], [0, 2, 0], [0, 0, 2]])
        ]
        
        test_pairs = [
            ([[1, 0, 1], [0, 1, 0], [1, 0, 1]], 
             [[2, 0, 2], [0, 2, 0], [2, 0, 2]])
        ]
        
        metadata = {
            "description": "将颜色1替换为颜色2",
            "type": "color_transformation",
            "difficulty": "easy",
            "source": "generated",
            "colors_used": [0, 1, 2],
            "transformation_hints": ["color_1_becomes_2"]
        }
        
        return SynthesisTask(
            task_id=task_id,
            train_pairs=train_pairs,
            test_pairs=test_pairs,
            metadata=metadata
        )
    
    def create_spatial_task(self, task_id: str = "simple_translation") -> SynthesisTask:
        """创建简单的空间变换任务"""
        
        train_pairs = [
            # 向右平移1格
            ([[1, 0, 0], [0, 0, 0], [0, 0, 0]], 
             [[0, 1, 0], [0, 0, 0], [0, 0, 0]]),
            ([[0, 0, 0], [1, 0, 0], [0, 0, 0]], 
             [[0, 0, 0], [0, 1, 0], [0, 0, 0]]),
            ([[0, 0, 0], [0, 0, 0], [1, 0, 0]], 
             [[0, 0, 0], [0, 0, 0], [0, 1, 0]])
        ]
        
        test_pairs = [
            ([[1, 0, 0], [1, 0, 0], [0, 0, 0]], 
             [[0, 1, 0], [0, 1, 0], [0, 0, 0]])
        ]
        
        metadata = {
            "description": "将所有对象向右平移1格",
            "type": "spatial_transformation",
            "difficulty": "easy",
            "source": "generated",
            "colors_used": [0, 1],
            "transformation_hints": ["translate_right_1"]
        }
        
        return SynthesisTask(
            task_id=task_id,
            train_pairs=train_pairs,
            test_pairs=test_pairs,
            metadata=metadata
        )
    
    def create_random_task(self, task_id: str = None, 
                          task_type: str = "color_transformation",
                          difficulty: str = "easy") -> SynthesisTask:
        """创建随机任务"""
        
        if task_id is None:
            task_id = f"random_{task_type}_{random.randint(1000, 9999)}"
        
        if task_type == "color_transformation":
            return self._create_random_color_task(task_id, difficulty)
        elif task_type == "spatial_transformation":
            return self._create_random_spatial_task(task_id, difficulty)
        else:
            return self.create_simple_task(task_id)
    
    def _create_random_color_task(self, task_id: str, difficulty: str) -> SynthesisTask:
        """创建随机颜色转换任务"""
        
        # 根据难度确定参数
        if difficulty == "easy":
            grid_size = 3
            num_colors = 3
            num_examples = 3
        elif difficulty == "medium":
            grid_size = 4
            num_colors = 4
            num_examples = 4
        else:  # hard
            grid_size = 5
            num_colors = 5
            num_examples = 5
        
        # 生成颜色映射
        colors = list(range(num_colors))
        old_color = random.choice(colors[1:])  # 不选择背景色0
        new_color = random.choice([c for c in colors if c != old_color])
        
        # 生成训练示例
        train_pairs = []
        for _ in range(num_examples):
            input_grid = self._generate_random_grid(grid_size, colors)
            output_grid = self._apply_color_mapping(input_grid, {old_color: new_color})
            train_pairs.append((input_grid, output_grid))
        
        # 生成测试示例
        test_input = self._generate_random_grid(grid_size, colors)
        test_output = self._apply_color_mapping(test_input, {old_color: new_color})
        test_pairs = [(test_input, test_output)]
        
        metadata = {
            "description": f"将颜色{old_color}替换为颜色{new_color}",
            "type": "color_transformation",
            "difficulty": difficulty,
            "source": "random_generated",
            "colors_used": colors,
            "transformation_hints": [f"color_{old_color}_becomes_{new_color}"]
        }
        
        return SynthesisTask(
            task_id=task_id,
            train_pairs=train_pairs,
            test_pairs=test_pairs,
            metadata=metadata
        )
    
    def _create_random_spatial_task(self, task_id: str, difficulty: str) -> SynthesisTask:
        """创建随机空间变换任务"""
        
        # 根据难度确定参数
        if difficulty == "easy":
            grid_size = 3
            num_examples = 3
        elif difficulty == "medium":
            grid_size = 4
            num_examples = 4
        else:  # hard
            grid_size = 5
            num_examples = 5
        
        # 随机选择变换类型
        transformations = ["translate", "rotate", "reflect"]
        transform_type = random.choice(transformations)
        
        # 生成示例
        train_pairs = []
        for _ in range(num_examples):
            input_grid = self._generate_sparse_grid(grid_size)
            output_grid = self._apply_spatial_transform(input_grid, transform_type)
            train_pairs.append((input_grid, output_grid))
        
        # 生成测试示例
        test_input = self._generate_sparse_grid(grid_size)
        test_output = self._apply_spatial_transform(test_input, transform_type)
        test_pairs = [(test_input, test_output)]
        
        metadata = {
            "description": f"应用{transform_type}变换",
            "type": "spatial_transformation",
            "difficulty": difficulty,
            "source": "random_generated",
            "colors_used": [0, 1],
            "transformation_hints": [transform_type]
        }
        
        return SynthesisTask(
            task_id=task_id,
            train_pairs=train_pairs,
            test_pairs=test_pairs,
            metadata=metadata
        )
    
    def _generate_random_grid(self, size: int, colors: List[int]) -> List[List[int]]:
        """生成随机网格"""
        grid = []
        for _ in range(size):
            row = []
            for _ in range(size):
                # 偏向于选择背景色
                if random.random() < 0.6:
                    row.append(0)  # 背景色
                else:
                    row.append(random.choice(colors[1:]))  # 非背景色
            grid.append(row)
        
        return grid
    
    def _generate_sparse_grid(self, size: int) -> List[List[int]]:
        """生成稀疏网格（用于空间变换）"""
        grid = [[0 for _ in range(size)] for _ in range(size)]
        
        # 随机放置一些非零元素
        num_elements = random.randint(1, min(3, size))
        
        for _ in range(num_elements):
            row = random.randint(0, size - 1)
            col = random.randint(0, size - 1)
            grid[row][col] = 1
        
        return grid
    
    def _apply_color_mapping(self, grid: List[List[int]], 
                           mapping: Dict[int, int]) -> List[List[int]]:
        """应用颜色映射"""
        result = []
        for row in grid:
            new_row = []
            for cell in row:
                new_row.append(mapping.get(cell, cell))
            result.append(new_row)
        
        return result
    
    def _apply_spatial_transform(self, grid: List[List[int]], 
                               transform_type: str) -> List[List[int]]:
        """应用空间变换"""
        array = np.array(grid)
        
        if transform_type == "rotate":
            result = np.rot90(array)
        elif transform_type == "reflect":
            if random.choice([True, False]):
                result = np.fliplr(array)  # 水平翻转
            else:
                result = np.flipud(array)  # 垂直翻转
        elif transform_type == "translate":
            # 简单平移（向右下移动1格）
            result = np.zeros_like(array)
            result[1:, 1:] = array[:-1, :-1]
        else:
            result = array
        
        return result.tolist()
    
    def get_task_statistics(self, tasks: List[SynthesisTask] = None) -> Dict[str, Any]:
        """获取任务统计信息"""
        if tasks is None:
            tasks = list(self.cache.values())
        
        if not tasks:
            return {}
        
        stats = {
            'total_tasks': len(tasks),
            'task_types': {},
            'difficulties': {},
            'grid_sizes': [],
            'colors_used': set(),
            'average_examples': 0
        }
        
        total_examples = 0
        
        for task in tasks:
            # 任务类型统计
            task_type = task.metadata.get('type', 'unknown')
            stats['task_types'][task_type] = stats['task_types'].get(task_type, 0) + 1
            
            # 难度统计
            difficulty = task.metadata.get('difficulty', 'unknown')
            stats['difficulties'][difficulty] = stats['difficulties'].get(difficulty, 0) + 1
            
            # 示例数量
            total_examples += len(task.train_pairs)
            
            # 网格大小和颜色
            for input_grid, output_grid in task.train_pairs:
                stats['grid_sizes'].append((len(input_grid), len(input_grid[0])))
                
                for row in input_grid + output_grid:
                    stats['colors_used'].update(row)
        
        stats['average_examples'] = total_examples / len(tasks) if tasks else 0
        stats['colors_used'] = list(stats['colors_used'])
        
        return stats
    
    def clear_cache(self):
        """清除缓存"""
        self.cache.clear()
        logger.info("缓存已清除")
'''

print("继续生成项目的剩余文件...")
print("已完成:")
print("✅ cegis/synthesizer.py - CEGIS合成器")
print("✅ cegis/verifier.py - 程序验证器") 
print("✅ cegis/counterexample.py - 反例生成器")
print("✅ utils/arc_loader.py - ARC数据加载器")
print("\\n接下来生成最后的工具模块和配置文件...")
            