# =====================================================================
# ARC程序合成框架 - 剩余核心文件
# =====================================================================

# =====================================================================
# 1. core/popper_interface.py - 完整Popper接口实现
# =====================================================================
POPPER_INTERFACE_PY = '''"""
Popper ILP系统接口 - 完整实现
"""
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
import time
import re

logger = logging.getLogger(__name__)

class PopperInterface:
    """
    Popper归纳逻辑编程系统接口
    提供完整的Popper集成功能
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.popper_path = Path(config.get('popper_path', './popper'))
        self.timeout = config.get('timeout', 600)
        self.solver = config.get('solver', 'rc2')
        self.max_vars = config.get('max_vars', 8)
        self.max_body = config.get('max_body', 10)
        self.max_rules = config.get('max_rules', 5)
        self.noisy = config.get('noisy', False)
        self.enable_recursion = config.get('enable_recursion', False)
        
        # 验证Popper安装
        self._verify_popper_installation()
        logger.info(f"Popper接口初始化完成: {self.popper_path}")
    
    def learn_program(self, task_dir: Path, constraints: List[str] = None) -> Optional[str]:
        """
        运行Popper学习程序
        
        Args:
            task_dir: 包含exs.pl, bk.pl, bias.pl的任务目录
            constraints: CEGIS反馈的额外约束
            
        Returns:
            学习到的程序字符串，失败返回None
        """
        logger.info(f"开始Popper学习: {task_dir}")
        
        try:
            # 验证输入文件
            if not self._validate_input_files(task_dir):
                logger.error("输入文件验证失败")
                return None
            
            # 如果有约束，创建修改后的任务
            working_dir = task_dir
            if constraints:
                working_dir = self._create_constrained_task(task_dir, constraints)
                logger.debug(f"创建约束任务目录: {working_dir}")
            
            # 构建Popper命令
            cmd = self._build_popper_command(working_dir)
            
            # 执行Popper
            start_time = time.time()
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout + 30,
                cwd=self.popper_path.parent
            )
            execution_time = time.time() - start_time
            
            logger.debug(f"Popper执行时间: {execution_time:.2f}秒")
            
            # 处理结果
            if result.returncode == 0:
                program = self._parse_popper_output(result.stdout)
                if program:
                    logger.info(f"Popper学习成功")
                    logger.debug(f"学到的程序: {program}")
                    return program
                else:
                    logger.warning("Popper运行成功但未找到程序")
            else:
                logger.warning(f"Popper执行失败 (返回码: {result.returncode})")
                logger.debug(f"错误输出: {result.stderr}")
                
            return None
            
        except subprocess.TimeoutExpired:
            logger.warning(f"Popper超时 ({self.timeout}秒)")
            return None
        except Exception as e:
            logger.error(f"运行Popper时出错: {str(e)}")
            return None
        finally:
            # 清理临时文件
            if constraints and working_dir != task_dir:
                try:
                    shutil.rmtree(working_dir)
                except Exception as e:
                    logger.warning(f"清理临时目录失败: {str(e)}")
    
    def _validate_input_files(self, task_dir: Path) -> bool:
        """验证Popper输入文件是否存在且有效"""
        required_files = ['exs.pl', 'bk.pl', 'bias.pl']
        
        for filename in required_files:
            file_path = task_dir / filename
            if not file_path.exists():
                logger.error(f"缺少必需文件: {file_path}")
                return False
            
            if file_path.stat().st_size == 0:
                logger.warning(f"文件为空: {file_path}")
        
        return True
    
    def _build_popper_command(self, task_dir: Path) -> List[str]:
        """构建Popper执行命令"""
        cmd = [
            'python', str(self.popper_path / 'popper.py'),
            str(task_dir),
            '--timeout', str(self.timeout),
            '--solver', self.solver
        ]
        
        # 添加可选参数
        if self.noisy:
            cmd.append('--noisy')
        
        if not self.enable_recursion:
            cmd.append('--no-recursion')
        
        # 添加统计信息
        cmd.append('--stats')
        
        logger.debug(f"Popper命令: {' '.join(cmd)}")
        return cmd
    
    def _create_constrained_task(self, original_dir: Path, constraints: List[str]) -> Path:
        """创建包含额外约束的临时任务目录"""
        temp_dir = Path(tempfile.mkdtemp(prefix="popper_constrained_"))
        
        try:
            # 复制原始文件
            for file_name in ['exs.pl', 'bk.pl', 'bias.pl']:
                src = original_dir / file_name
                dst = temp_dir / file_name
                if src.exists():
                    shutil.copy2(src, dst)
            
            # 在偏置文件中添加约束
            bias_file = temp_dir / 'bias.pl'
            if bias_file.exists():
                with open(bias_file, 'a', encoding='utf-8') as f:
                    f.write("\\n% CEGIS生成的约束\\n")
                    for constraint in constraints:
                        f.write(f"{constraint}\\n")
            
            return temp_dir
            
        except Exception as e:
            # 如果失败，清理并重新抛出异常
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise e
    
    def _parse_popper_output(self, output: str) -> Optional[str]:
        """解析Popper输出提取学习到的程序"""
        lines = output.strip().split('\\n')
        program_lines = []
        in_program_section = False
        
        for line in lines:
            line = line.strip()
            
            # 检测程序段开始
            if line.startswith('Program:') or 'SOLUTION' in line:
                in_program_section = True
                continue
            
            # 检测程序段结束
            if in_program_section and (
                line.startswith('Precision:') or 
                line.startswith('Recall:') or
                line.startswith('Time:') or
                line.startswith('Timeout') or
                line == ''
            ):
                break
            
            # 收集程序行
            if in_program_section and line and not line.startswith('%'):
                # 清理和验证Prolog语法
                cleaned_line = self._clean_prolog_line(line)
                if cleaned_line:
                    program_lines.append(cleaned_line)
        
        if program_lines:
            program = '\\n'.join(program_lines)
            # 验证程序语法
            if self._validate_prolog_syntax(program):
                return program
            else:
                logger.warning("学到的程序语法无效")
                return None
        
        return None
    
    def _clean_prolog_line(self, line: str) -> str:
        """清理Prolog行，确保格式正确"""
        line = line.strip()
        
        # 移除行号（如果有）
        line = re.sub(r'^\\d+\\s*:', '', line)
        
        # 确保每行以点结尾（如果是规则或事实）
        if line and not line.endswith('.') and not line.startswith('%'):
            if ':-' in line or any(line.startswith(pred) for pred in ['pos(', 'neg(', 'head_pred(', 'body_pred(']):
                line += '.'
        
        return line
    
    def _validate_prolog_syntax(self, program: str) -> bool:
        """基本的Prolog语法验证"""
        try:
            lines = [line.strip() for line in program.split('\\n') if line.strip()]
            
            for line in lines:
                if line.startswith('%'):  # 注释行
                    continue
                
                # 检查基本语法规则
                if not line.endswith('.'):
                    logger.debug(f"语法错误: 行不以点结尾: {line}")
                    return False
                
                # 检查括号匹配
                if line.count('(') != line.count(')'):
                    logger.debug(f"语法错误: 括号不匹配: {line}")
                    return False
            
            return True
            
        except Exception as e:
            logger.debug(f"语法验证出错: {str(e)}")
            return False
    
    def _verify_popper_installation(self) -> bool:
        """验证Popper是否正确安装"""
        try:
            popper_script = self.popper_path / 'popper.py'
            if not popper_script.exists():
                raise FileNotFoundError(f"Popper脚本未找到: {popper_script}")
            
            # 测试Popper是否可以运行
            result = subprocess.run(
                ['python', str(popper_script), '--help'],
                capture_output=True,
                timeout=10,
                cwd=self.popper_path.parent
            )
            
            if result.returncode != 0:
                raise RuntimeError(f"Popper无法正常运行: {result.stderr}")
            
            logger.debug("Popper安装验证成功")
            return True
            
        except Exception as e:
            error_msg = f"Popper安装验证失败: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def get_learning_statistics(self, output: str) -> Dict:
        """从Popper输出中提取学习统计信息"""
        stats = {
            'time': 0.0,
            'rules_generated': 0,
            'examples_processed': 0,
            'precision': 0.0,
            'recall': 0.0
        }
        
        try:
            lines = output.split('\\n')
            for line in lines:
                line = line.strip()
                
                if line.startswith('Time:'):
                    match = re.search(r'Time:\\s*([\\d.]+)', line)
                    if match:
                        stats['time'] = float(match.group(1))
                
                elif line.startswith('Precision:'):
                    match = re.search(r'Precision:\\s*([\\d.]+)', line)
                    if match:
                        stats['precision'] = float(match.group(1))
                
                elif line.startswith('Recall:'):
                    match = re.search(r'Recall:\\s*([\\d.]+)', line)
                    if match:
                        stats['recall'] = float(match.group(1))
        
        except Exception as e:
            logger.warning(f"解析统计信息失败: {str(e)}")
        
        return stats
'''

# =====================================================================
# 2. core/anti_unification.py - 完整反统一算法实现
# =====================================================================
ANTI_UNIFICATION_PY = '''"""
反统一算法实现 - 用于程序模式泛化
基于Plotkin的最小一般泛化算法
"""
import logging
from typing import List, Dict, Tuple, Any, Optional, Set
from dataclasses import dataclass
from collections import defaultdict
import re

logger = logging.getLogger(__name__)

@dataclass
class Term:
    """逻辑项表示"""
    functor: str
    args: List['Term']
    is_variable: bool = False
    term_type: Optional[str] = None
    
    def __hash__(self):
        if self.is_variable:
            return hash(('var', self.functor))
        return hash(('term', self.functor, tuple(self.args)))
    
    def __eq__(self, other):
        if not isinstance(other, Term):
            return False
        return (self.functor == other.functor and 
                self.args == other.args and 
                self.is_variable == other.is_variable)
    
    def __str__(self):
        if self.is_variable:
            return self.functor
        if not self.args:
            return self.functor
        args_str = ', '.join(str(arg) for arg in self.args)
        return f"{self.functor}({args_str})"

@dataclass
class Clause:
    """Prolog子句表示"""
    head: Term
    body: List[Term]
    
    def __str__(self):
        if not self.body:
            return f"{self.head}."
        body_str = ', '.join(str(term) for term in self.body)
        return f"{self.head} :- {body_str}."

class AntiUnifier:
    """
    反统一算法实现
    用于从多个程序中提取共同模式
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.max_depth = config.get('max_generalization_depth', 5)
        self.preserve_structure = config.get('preserve_structure', True)
        self.enable_type_constraints = config.get('enable_type_constraints', True)
        
        # 反统一状态
        self.variable_counter = 0
        self.disagreement_store = {}
        self.type_constraints = {}
        
        logger.info("反统一模块初始化完成")
    
    def generalize_programs(self, programs: List[str]) -> str:
        """
        对多个程序进行反统一泛化
        
        Args:
            programs: 程序字符串列表
            
        Returns:
            泛化后的程序模式
        """
        if not programs:
            return ""
        
        if len(programs) == 1:
            return programs[0]
        
        logger.info(f"开始对{len(programs)}个程序进行反统一")
        
        try:
            # 解析程序为结构化表示
            parsed_programs = []
            for i, program in enumerate(programs):
                parsed = self._parse_program(program)
                if parsed:
                    parsed_programs.append(parsed)
                else:
                    logger.warning(f"程序{i+1}解析失败，跳过")
            
            if len(parsed_programs) < 2:
                logger.warning("可解析的程序少于2个，无法进行反统一")
                return programs[0] if programs else ""
            
            # 执行反统一
            result = self._anti_unify_program_list(parsed_programs)
            
            # 转换回程序字符串
            generalized_program = self._clauses_to_program(result)
            
            logger.info("反统一完成")
            logger.debug(f"泛化结果: {generalized_program}")
            
            return generalized_program
            
        except Exception as e:
            logger.error(f"反统一过程出错: {str(e)}")
            return programs[0]  # 出错时返回第一个程序
    
    def generalize_program(self, program: str, examples: List[Tuple]) -> str:
        """
        基于示例对单个程序进行泛化
        
        Args:
            program: 程序字符串
            examples: 示例对列表
            
        Returns:
            泛化后的程序
        """
        logger.info("基于示例进行程序泛化")
        
        try:
            # 分析程序在不同示例上的行为模式
            behavioral_patterns = []
            
            for i, (example_input, example_output) in enumerate(examples):
                pattern = self._extract_behavioral_pattern(
                    program, example_input, example_output
                )
                if pattern:
                    behavioral_patterns.append(pattern)
                    logger.debug(f"示例{i+1}模式: {pattern}")
            
            if not behavioral_patterns:
                logger.warning("未提取到行为模式，返回原程序")
                return program
            
            # 泛化行为模式
            generalized_pattern = self._generalize_behavioral_patterns(behavioral_patterns)
            
            # 将模式转换为程序
            if generalized_pattern:
                result = self._pattern_to_program(generalized_pattern)
                logger.info("基于示例的泛化完成")
                return result
            else:
                return program
                
        except Exception as e:
            logger.error(f"基于示例的泛化失败: {str(e)}")
            return program
    
    def _anti_unify_program_list(self, programs: List[List[Clause]]) -> List[Clause]:
        """对程序列表执行反统一"""
        if len(programs) < 2:
            return programs[0] if programs else []
        
        # 重置状态
        self.disagreement_store.clear()
        self.variable_counter = 0
        
        # 两两进行反统一
        result = programs[0]
        for i in range(1, len(programs)):
            result = self._anti_unify_two_programs(result, programs[i])
            logger.debug(f"第{i+1}轮反统一完成")
        
        return result
    
    def _anti_unify_two_programs(self, prog1: List[Clause], prog2: List[Clause]) -> List[Clause]:
        """对两个程序执行反统一"""
        
        # 如果长度不同，需要对齐或选择最保守的策略
        if len(prog1) != len(prog2):
            return self._handle_different_length_programs(prog1, prog2)
        
        unified_clauses = []
        for clause1, clause2 in zip(prog1, prog2):
            unified_clause = self._anti_unify_clauses(clause1, clause2)
            if unified_clause:
                unified_clauses.append(unified_clause)
        
        return unified_clauses
    
    def _anti_unify_clauses(self, clause1: Clause, clause2: Clause) -> Optional[Clause]:
        """对两个子句执行反统一"""
        try:
            # 反统一头部
            unified_head = self._anti_unify_terms(clause1.head, clause2.head)
            
            # 反统一体部
            unified_body = []
            
            # 简单策略：只有当体部长度相同时才反统一
            if len(clause1.body) == len(clause2.body):
                for term1, term2 in zip(clause1.body, clause2.body):
                    unified_term = self._anti_unify_terms(term1, term2)
                    unified_body.append(unified_term)
            else:
                # 长度不同时，创建更一般的体部
                unified_body = self._generalize_different_bodies(clause1.body, clause2.body)
            
            return Clause(unified_head, unified_body)
            
        except Exception as e:
            logger.debug(f"子句反统一失败: {str(e)}")
            return None
    
    def _anti_unify_terms(self, term1: Term, term2: Term) -> Term:
        """
        对两个项执行反统一 - Plotkin算法核心
        
        反统一规则:
        1. 如果两项相同，返回该项
        2. 如果有相同函子和元数，递归反统一参数
        3. 否则引入新变量
        """
        # 规则1: 相同项
        if term1 == term2:
            return term1
        
        # 规则2: 相同函子和元数
        if (not term1.is_variable and not term2.is_variable and 
            term1.functor == term2.functor and len(term1.args) == len(term2.args)):
            
            unified_args = []
            for arg1, arg2 in zip(term1.args, term2.args):
                unified_arg = self._anti_unify_terms(arg1, arg2)
                unified_args.append(unified_arg)
            
            return Term(term1.functor, unified_args, term_type=term1.term_type)
        
        # 规则3: 引入新变量
        return self._create_fresh_variable(term1, term2)
    
    def _create_fresh_variable(self, term1: Term, term2: Term) -> Term:
        """创建新的变量来表示分歧"""
        disagreement_key = (term1, term2)
        
        # 检查是否已经有对应的变量
        if disagreement_key in self.disagreement_store:
            return self.disagreement_store[disagreement_key]
        
        # 创建新变量
        var_name = f"X{self.variable_counter}"
        self.variable_counter += 1
        
        # 尝试推断类型
        inferred_type = self._infer_variable_type(term1, term2)
        
        fresh_var = Term(var_name, [], is_variable=True, term_type=inferred_type)
        self.disagreement_store[disagreement_key] = fresh_var
        
        # 记录类型约束
        if self.enable_type_constraints and inferred_type:
            self.type_constraints[var_name] = inferred_type
        
        return fresh_var
    
    def _infer_variable_type(self, term1: Term, term2: Term) -> Optional[str]:
        """推断变量类型"""
        # 如果两个项都有类型信息，尝试找到最一般的公共类型
        if term1.term_type and term2.term_type:
            if term1.term_type == term2.term_type:
                return term1.term_type
            else:
                # 可以实现类型层次来找最一般公共超类型
                return "any"
        
        # 基于函子名推断类型
        if not term1.is_variable and not term2.is_variable:
            if term1.functor.isdigit() and term2.functor.isdigit():
                return "int"
            elif term1.functor in ['grid', 'cell'] or term2.functor in ['grid', 'cell']:
                return "spatial"
        
        return None
    
    def _parse_program(self, program: str) -> Optional[List[Clause]]:
        """解析程序字符串为子句列表"""
        try:
            clauses = []
            lines = [line.strip() for line in program.split('\\n') 
                    if line.strip() and not line.strip().startswith('%')]
            
            for line in lines:
                if line.endswith('.'):
                    line = line[:-1]  # 移除结尾的点
                
                clause = self._parse_clause(line)
                if clause:
                    clauses.append(clause)
            
            return clauses
            
        except Exception as e:
            logger.error(f"程序解析失败: {str(e)}")
            return None
    
    def _parse_clause(self, clause_str: str) -> Optional[Clause]:
        """解析单个子句"""
        try:
            clause_str = clause_str.strip()
            
            if ':-' in clause_str:
                # 规则
                head_str, body_str = clause_str.split(':-', 1)
                head = self._parse_term(head_str.strip())
                body_terms = []
                
                # 解析体部
                if body_str.strip():
                    body_parts = self._split_body_terms(body_str.strip())
                    for part in body_parts:
                        term = self._parse_term(part.strip())
                        if term:
                            body_terms.append(term)
                
                return Clause(head, body_terms)
            else:
                # 事实
                head = self._parse_term(clause_str)
                return Clause(head, [])
                
        except Exception as e:
            logger.debug(f"子句解析失败: {clause_str}, 错误: {str(e)}")
            return None
    
    def _parse_term(self, term_str: str) -> Optional[Term]:
        """解析项"""
        try:
            term_str = term_str.strip()
            
            # 检查是否是变量（大写字母开头）
            if term_str and term_str[0].isupper():
                return Term(term_str, [], is_variable=True)
            
            # 检查是否有参数
            if '(' not in term_str:
                return Term(term_str, [])
            
            # 解析复合项
            paren_pos = term_str.find('(')
            functor = term_str[:paren_pos]
            args_str = term_str[paren_pos+1:-1]  # 移除括号
            
            args = []
            if args_str.strip():
                arg_strs = self._split_arguments(args_str)
                for arg_str in arg_strs:
                    arg = self._parse_term(arg_str.strip())
                    if arg:
                        args.append(arg)
            
            return Term(functor, args)
            
        except Exception as e:
            logger.debug(f"项解析失败: {term_str}, 错误: {str(e)}")
            return None
    
    def _split_body_terms(self, body_str: str) -> List[str]:
        """分割体部的项"""
        # 简单的逗号分割，需要考虑嵌套括号
        terms = []
        current = ""
        paren_count = 0
        
        for char in body_str:
            if char == '(':
                paren_count += 1
            elif char == ')':
                paren_count -= 1
            elif char == ',' and paren_count == 0:
                terms.append(current.strip())
                current = ""
                continue
            
            current += char
        
        if current.strip():
            terms.append(current.strip())
        
        return terms
    
    def _split_arguments(self, args_str: str) -> List[str]:
        """分割参数"""
        # 与_split_body_terms类似的逻辑
        return self._split_body_terms(args_str)
    
    def _clauses_to_program(self, clauses: List[Clause]) -> str:
        """将子句列表转换为程序字符串"""
        program_lines = []
        
        for clause in clauses:
            clause_str = str(clause)
            program_lines.append(clause_str)
        
        return '\\n'.join(program_lines)
    
    def _handle_different_length_programs(self, prog1: List[Clause], prog2: List[Clause]) -> List[Clause]:
        """处理不同长度的程序"""
        # 保守策略：取较短程序的长度
        min_len = min(len(prog1), len(prog2))
        
        result = []
        for i in range(min_len):
            unified_clause = self._anti_unify_clauses(prog1[i], prog2[i])
            if unified_clause:
                result.append(unified_clause)
        
        return result
    
    def _generalize_different_bodies(self, body1: List[Term], body2: List[Term]) -> List[Term]:
        """泛化不同的体部"""
        # 简单策略：创建一个变量来表示体部
        if not body1 and not body2:
            return []
        
        # 可以实现更复杂的体部对齐和泛化算法
        var_name = f"Body{self.variable_counter}"
        self.variable_counter += 1
        
        return [Term(var_name, [], is_variable=True)]
    
    def _extract_behavioral_pattern(self, program: str, input_data: Any, output_data: Any) -> Optional[Dict]:
        """从程序行为中提取模式"""
        # 这里可以实现程序执行和模式提取
        # 简化实现：基于输入输出分析程序结构
        try:
            pattern = {
                'program_structure': self._analyze_program_structure(program),
                'input_properties': self._analyze_data_properties(input_data),
                'output_properties': self._analyze_data_properties(output_data),
                'transformation_type': self._infer_transformation_type(input_data, output_data)
            }
            return pattern
        except Exception as e:
            logger.debug(f"提取行为模式失败: {str(e)}")
            return None
    
    def _analyze_program_structure(self, program: str) -> Dict:
        """分析程序结构"""
        lines = [line.strip() for line in program.split('\\n') if line.strip()]
        
        return {
            'rule_count': len(lines),
            'has_recursion': any(':-' in line and line.split(':-')[0].strip() in line.split(':-')[1] 
                                for line in lines if ':-' in line),
            'predicates_used': self._extract_predicates(program)
        }
    
    def _analyze_data_properties(self, data: Any) -> Dict:
        """分析数据属性"""
        if isinstance(data, list):
            return {
                'type': 'list',
                'length': len(data),
                'element_types': list(set(type(item).__name__ for item in data))
            }
        elif isinstance(data, dict):
            return {
                'type': 'dict',
                'keys': list(data.keys()),
                'value_types': list(set(type(v).__name__ for v in data.values()))
            }
        else:
            return {
                'type': type(data).__name__,
                'value': str(data)
            }
    
    def _infer_transformation_type(self, input_data: Any, output_data: Any) -> str:
        """推断转换类型"""
        # 简单的转换类型推断
        if input_data == output_data:
            return "identity"
        elif isinstance(input_data, list) and isinstance(output_data, list):
            if len(input_data) == len(output_data):
                return "element_wise_transformation"
            else:
                return "structure_modification"
        else:
            return "unknown"
    
    def _extract_predicates(self, program: str) -> List[str]:
        """提取程序中使用的谓词"""
        predicates = set()
        
        # 使用正则表达式提取谓词名
        pattern = r'\\b([a-z][a-zA-Z0-9_]*)[\\s]*\\('
        matches = re.findall(pattern, program)
        
        for match in matches:
            predicates.add(match)
        
        return list(predicates)
    
    def _generalize_behavioral_patterns(self, patterns: List[Dict]) -> Optional[Dict]:
        """泛化行为模式"""
        if not patterns:
            return None
        
        # 简单的模式泛化：找共同特征
        common_pattern = {}
        
        # 泛化程序结构
        structures = [p.get('program_structure', {}) for p in patterns]
        common_pattern['program_structure'] = self._generalize_structures(structures)
        
        # 泛化转换类型
        transform_types = [p.get('transformation_type') for p in patterns]
        most_common_type = max(set(transform_types), key=transform_types.count)
        common_pattern['transformation_type'] = most_common_type
        
        return common_pattern
    
    def _generalize_structures(self, structures: List[Dict]) -> Dict:
        """泛化程序结构"""
        if not structures:
            return {}
        
        # 计算平均规则数
        rule_counts = [s.get('rule_count', 0) for s in structures]
        avg_rules = sum(rule_counts) / len(rule_counts)
        
        # 检查是否都有递归
        has_recursions = [s.get('has_recursion', False) for s in structures]
        common_recursion = all(has_recursions)
        
        # 找共同谓词
        all_predicates = []
        for s in structures:
            all_predicates.extend(s.get('predicates_used', []))
        
        predicate_counts = defaultdict(int)
        for pred in all_predicates:
            predicate_counts[pred] += 1
        
        # 保留出现在多数结构中的谓词
        threshold = len(structures) / 2
        common_predicates = [pred for pred, count in predicate_counts.items() 
                           if count > threshold]
        
        return {
            'avg_rule_count': avg_rules,
            'has_recursion': common_recursion,
            'common_predicates': common_predicates
        }
    
    def _pattern_to_program(self, pattern: Dict) -> str:
        """将模式转换为程序"""
        # 这是一个简化的实现
        # 实际中需要更复杂的程序生成逻辑
        
        structure = pattern.get('program_structure', {})
        transform_type = pattern.get('transformation_type', 'unknown')
        
        # 基于模式生成基本程序模板
        if transform_type == 'element_wise_transformation':
            return self._generate_element_wise_template(structure)
        elif transform_type == 'structure_modification':
            return self._generate_structure_mod_template(structure)
        else:
            return self._generate_generic_template(structure)
    
    def _generate_element_wise_template(self, structure: Dict) -> str:
        """生成元素级转换模板"""
        predicates = structure.get('common_predicates', ['transform'])
        main_pred = predicates[0] if predicates else 'transform'
        
        template = f"{main_pred}(Input, Output) :-\\n"
        template += "    apply_element_transformation(Input, Output)."
        
        return template
    
    def _generate_structure_mod_template(self, structure: Dict) -> str:
        """生成结构修改模板"""
        return "transform(Input, Output) :-\\n    modify_structure(Input, Output)."
    
    def _generate_generic_template(self, structure: Dict) -> str:
        """生成通用模板"""
        return "transform(Input, Output) :-\\n    apply_transformation(Input, Output)."
'''

# =====================================================================
# 3. extraction/object_extractor.py - 完整对象提取器实现
# =====================================================================
OBJECT_EXTRACTOR_PY = '''"""
ARC网格对象提取器 - 完整实现
使用连通组件分析和高级几何特征提取
"""
import numpy as np
from typing import List, Dict, Set, Tuple, Any, Optional
from collections import defaultdict, Counter, deque
import math
from dataclasses import dataclass, field
from scipy import ndimage
from skimage import measure, morphology
import logging

logger = logging.getLogger(__name__)

@dataclass
class ARCObject:
    """ARC网格对象表示"""
    id: int
    cells: Set[Tuple[int, int]]
    color: int
    bbox: Tuple[int, int, int, int]  # (min_row, min_col, max_row, max_col)
    centroid: Tuple[float, float]
    size: int
    shape_type: str
    convex_hull: List[Tuple[int, int]]
    holes: List[Set[Tuple[int, int]]]
    attributes: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """计算派生属性"""
        if not self.attributes:
            self.attributes = {}
        
        # 基本几何属性
        self.attributes.update({
            'width': self.bbox[3] - self.bbox[1] + 1,
            'height': self.bbox[2] - self.bbox[0] + 1,
            'area': self.size,
            'perimeter': self._calculate_perimeter(),
            'aspect_ratio': (self.bbox[3] - self.bbox[1] + 1) / (self.bbox[2] - self.bbox[0] + 1),
            'density': self.size / ((self.bbox[2] - self.bbox[0] + 1) * (self.bbox[3] - self.bbox[1] + 1)),
            'hole_count': len(self.holes),
            'is_solid': len(self.holes) == 0
        })
    
    def _calculate_perimeter(self) -> int:
        """计算对象周长"""
        perimeter = 0
        for r, c in self.cells:
            # 检查4连通邻居
            neighbors = [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]
            for nr, nc in neighbors:
                if (nr, nc) not in self.cells:
                    perimeter += 1
        return perimeter

class ARCObjectExtractor:
    """
    高级ARC对象提取器
    支持多种连通性、形状分析和空间关系提取
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.connectivity = config.get('connectivity', 4)
        self.min_object_size = config.get('min_object_size', 1)
        self.background_color = config.get('background_color', 0)
        self.analyze_shapes = config.get('analyze_shapes', True)
        self.detect_patterns = config.get('detect_patterns', True)
        self.extract_holes = config.get('extract_holes', True)
        
        logger.info(f"对象提取器初始化: {self.connectivity}连通, 最小尺寸{self.min_object_size}")
    
    def process_arc_grid(self, grid: List[List[int]]) -> Dict[str, Any]:
        """
        完整的ARC网格处理管道
        
        Args:
            grid: 2D整数网格
            
        Returns:
            包含对象、关系和分析结果的字典
        """
        if not grid or not grid[0]:
            return self._empty_result()
        
        logger.debug(f"处理网格: {len(grid)}x{len(grid[0])}")
        
        # 转换为numpy数组
        np_grid = np.array(grid, dtype=int)
        
        # 1. 提取连通组件
        components = self._extract_connected_components(np_grid)
        logger.debug(f"发现{len(components)}个连通组件")
        
        # 2. 分析每个组件
        objects = []
        for comp_id, comp_data in components.items():
            if len(comp_data['cells']) >= self.min_object_size:
                obj = self._analyze_object_comprehensive(comp_data, np_grid, comp_id)
                if obj:
                    objects.append(obj)
        
        logger.debug(f"有效对象数量: {len(objects)}")
        
        # 3. 空间关系分析
        relationships = self._analyze_spatial_relationships(objects, np_grid)
        
        # 4. 全局网格分析
        grid_analysis = self._analyze_grid_global(np_grid, objects)
        
        # 5. 模式检测（如果启用）
        patterns = []
        if self.detect_patterns:
            patterns = self._detect_patterns(objects, np_grid)
        
        return {
            'objects': objects,
            'relationships': relationships,
            'grid_analysis': grid_analysis,
            'patterns': patterns,
            'grid_shape': np_grid.shape,
            'unique_colors': list(np.unique(np_grid)),
            'metadata': {
                'connectivity': self.connectivity,
                'total_objects': len(objects),
                'extraction_config': self.config
            }
        }
    
    def _extract_connected_components(self, grid: np.ndarray) -> Dict[int, Dict]:
        """
        高效的连通组件提取
        使用Union-Find算法优化
        """
        rows, cols = grid.shape
        parent = list(range(rows * cols))
        rank = [0] * (rows * cols)
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])  # 路径压缩
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px == py:
                return
            # 按秩合并
            if rank[px] < rank[py]:
                px, py = py, px
            parent[py] = px
            if rank[px] == rank[py]:
                rank[px] += 1
        
        # 定义连通性模式
        if self.connectivity == 4:
            directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        else:  # 8连通
            directions = [(0, 1), (1, 0), (0, -1), (-1, 0), 
                         (1, 1), (1, -1), (-1, 1), (-1, -1)]
        
        # 构建并查集
        for r in range(rows):
            for c in range(cols):
                if grid[r, c] == self.background_color:
                    continue
                
                current_idx = r * cols + c
                current_color = grid[r, c]
                
                for dr, dc in directions:
                    nr, nc = r + dr, c + dc
                    if (0 <= nr < rows and 0 <= nc < cols and 
                        grid[nr, nc] == current_color):
                        neighbor_idx = nr * cols + nc
                        union(current_idx, neighbor_idx)
        
        # 分组单元格
        components = defaultdict(lambda: {'cells': set(), 'color': 0})
        for r in range(rows):
            for c in range(cols):
                if grid[r, c] != self.background_color:
                    idx = r * cols + c
                    root = find(idx)
                    components[root]['cells'].add((r, c))
                    components[root]['color'] = grid[r, c]
        
        return dict(components)
    
    def _analyze_object_comprehensive(self, component_data: Dict, 
                                    grid: np.ndarray, obj_id: int) -> Optional[ARCObject]:
        """对象的全面分析"""
        cells = component_data['cells']
        color = component_data['color']
        
        if not cells:
            return None
        
        try:
            # 基本几何属性
            rows = [r for r, c in cells]
            cols = [c for r, c in cells]
            bbox = (min(rows), min(cols), max(rows), max(cols))
            centroid = (sum(rows) / len(rows), sum(cols) / len(cols))
            size = len(cells)
            
            # 形状分类
            shape_type = self._classify_shape_advanced(cells, bbox)
            
            # 凸包计算
            convex_hull = self._compute_convex_hull(cells)
            
            # 孔洞检测
            holes = []
            if self.extract_holes:
                holes = self._detect_holes(cells, bbox)
            
            # 创建对象
            obj = ARCObject(
                id=obj_id,
                cells=cells,
                color=color,
                bbox=bbox,
                centroid=centroid,
                size=size,
                shape_type=shape_type,
                convex_hull=convex_hull,
                holes=holes
            )
            
            # 高级属性分析
            self._analyze_advanced_attributes(obj, grid)
            
            return obj
            
        except Exception as e:
            logger.warning(f"对象{obj_id}分析失败: {str(e)}")
            return None
    
    def _classify_shape_advanced(self, cells: Set[Tuple[int, int]], 
                               bbox: Tuple[int, int, int, int]) -> str:
        """高级形状分类"""
        width = bbox[3] - bbox[1] + 1
        height = bbox[2] - bbox[0] + 1
        size = len(cells)
        expected_rect_size = width * height
        
        # 基本形状检测
        if size == expected_rect_size:
            if width == height and size > 1:
                return "square"
            elif width == 1 or height == 1:
                return "line"
            else:
                return "rectangle"
        
        if size == 1:
            return "point"
        
        # 线性结构检测
        if self._is_line_pattern(cells):
            return "line"
        
        # 特殊形状检测
        if self._is_l_shape(cells, bbox):
            return "l_shape"
        
        if self._is_t_shape(cells, bbox):
            return "t_shape"
        
        if self._is_cross_shape(cells, bbox):
            return "cross"
        
        if self._is_circle_like(cells, bbox):
            return "circle"
        
        # 对称性检测
        if self._has_symmetry(cells, bbox):
            return "symmetric"
        
        return "irregular"
    
    def _is_line_pattern(self, cells: Set[Tuple[int, int]]) -> bool:
        """检测是否为线性模式"""
        if len(cells) <= 2:
            return True
        
        cells_list = list(cells)
        
        # 检查是否在同一行
        rows = [r for r, c in cells_list]
        if len(set(rows)) == 1:
            return True
        
        # 检查是否在同一列
        cols = [c for r, c in cells_list]
        if len(set(cols)) == 1:
            return True
        
        # 检查是否在对角线上
        if self._is_diagonal_line(cells_list):
            return True
        
        return False
    
    def _is_diagonal_line(self, cells_list: List[Tuple[int, int]]) -> bool:
        """检测对角线"""
        if len(cells_list) <= 2:
            return True
        
        # 排序以便检查连续性
        sorted_cells = sorted(cells_list)
        
        # 检查斜率是否一致
        if len(sorted_cells) >= 2:
            dr = sorted_cells[1][0] - sorted_cells[0][0]
            dc = sorted_cells[1][1] - sorted_cells[0][1]
            
            for i in range(2, len(sorted_cells)):
                expected_r = sorted_cells[i-1][0] + dr
                expected_c = sorted_cells[i-1][1] + dc
                if sorted_cells[i] != (expected_r, expected_c):
                    return False
            
            return True
        
        return False
    
    def _is_l_shape(self, cells: Set[Tuple[int, int]], 
                   bbox: Tuple[int, int, int, int]) -> bool:
        """检测L形"""
        # L形必须至少有3个单元格
        if len(cells) < 3:
            return False
        
        # 简化的L形检测：检查是否有明显的拐点
        cells_list = list(cells)
        
        # 检查是否可以分解为两条垂直的线段
        rows = defaultdict(list)
        cols = defaultdict(list)
        
        for r, c in cells_list:
            rows[r].append(c)
            cols[c].append(r)
        
        # L形应该有一个拐点，其中一行和一列的交集是拐点
        for r in rows:
            for c in cols:
                if (r, c) in cells:
                    # 检查是否形成L形
                    horizontal_line = set((r, col) for col in rows[r] if (r, col) in cells)
                    vertical_line = set((row, c) for row in cols[c] if (row, c) in cells)
                    
                    if len(horizontal_line) >= 2 and len(vertical_line) >= 2:
                        union = horizontal_line | vertical_line
                        if len(union) == len(cells) and len(cells) >= 3:
                            return True
        
        return False
    
    def _is_t_shape(self, cells: Set[Tuple[int, int]], 
                   bbox: Tuple[int, int, int, int]) -> bool:
        """检测T形"""
        if len(cells) < 3:
            return False
        
        # T形由一条主线和一条垂直的交叉线组成
        cells_list = list(cells)
        rows = defaultdict(list)
        cols = defaultdict(list)
        
        for r, c in cells_list:
            rows[r].append(c)
            cols[c].append(r)
        
        # 寻找可能的T形结构
        for r in rows:
            if len(rows[r]) >= 2:  # 水平线
                for c in cols:
                    if len(cols[c]) >= 2 and (r, c) in cells:  # 垂直线与水平线相交
                        horizontal = set((r, col) for col in rows[r] if (r, col) in cells)
                        vertical = set((row, c) for row in cols[c] if (row, c) in cells)
                        
                        # 检查是否形成T形
                        if len(horizontal) >= 2 and len(vertical) >= 2:
                            union = horizontal | vertical
                            if len(union) == len(cells):
                                return True
        
        return False
    
    def _is_cross_shape(self, cells: Set[Tuple[int, int]], 
                       bbox: Tuple[int, int, int, int]) -> bool:
        """检测十字形"""
        if len(cells) < 5:  # 十字形至少需要5个单元格
            return False
        
        # 寻找中心点
        for r, c in cells:
            # 检查以(r,c)为中心的十字形
            horizontal = [(r, c-1), (r, c), (r, c+1)]
            vertical = [(r-1, c), (r, c), (r+1, c)]
            
            cross_cells = set(horizontal + vertical)
            
            # 检查是否所有十字形单元格都在对象中
            if cross_cells.issubset(cells):
                # 可能的十字形，检查是否还有其他扩展
                extended_cross = set()
                
                # 水平扩展
                left_c = c - 1
                while (r, left_c) in cells:
                    extended_cross.add((r, left_c))
                    left_c -= 1
                
                right_c = c + 1
                while (r, right_c) in cells:
                    extended_cross.add((r, right_c))
                    right_c += 1
                
                # 垂直扩展
                up_r = r - 1
                while (up_r, c) in cells:
                    extended_cross.add((up_r, c))
                    up_r -= 1
                
                down_r = r + 1
                while (down_r, c) in cells:
                    extended_cross.add((down_r, c))
                    down_r += 1
                
                # 添加中心点
                extended_cross.add((r, c))
                
                if extended_cross == cells:
                    return True
        
        return False
    
    def _is_circle_like(self, cells: Set[Tuple[int, int]], 
                       bbox: Tuple[int, int, int, int]) -> bool:
        """检测圆形或椭圆形"""
        if len(cells) < 4:
            return False
        
        # 计算中心
        center_r = (bbox[0] + bbox[2]) / 2
        center_c = (bbox[1] + bbox[3]) / 2
        
        # 计算到中心的距离
        distances = []
        for r, c in cells:
            dist = math.sqrt((r - center_r)**2 + (c - center_c)**2)
            distances.append(dist)
        
        # 检查距离的方差（圆形应该方差较小）
        if distances:
            mean_dist = sum(distances) / len(distances)
            variance = sum((d - mean_dist)**2 for d in distances) / len(distances)
            std_dev = math.sqrt(variance)
            
            # 如果标准差相对较小，可能是圆形
            if mean_dist > 0 and std_dev / mean_dist < 0.3:
                return True
        
        return False
    
    def _has_symmetry(self, cells: Set[Tuple[int, int]], 
                     bbox: Tuple[int, int, int, int]) -> bool:
        """检测对称性"""
        # 检查水平对称
        center_r = (bbox[0] + bbox[2]) / 2
        horizontal_symmetric = True
        
        for r, c in cells:
            mirror_r = 2 * center_r - r
            if abs(mirror_r - round(mirror_r)) < 0.1:  # 允许小的浮点误差
                mirror_r = round(mirror_r)
                if (mirror_r, c) not in cells:
                    horizontal_symmetric = False
                    break
        
        if horizontal_symmetric:
            return True
        
        # 检查垂直对称
        center_c = (bbox[1] + bbox[3]) / 2
        vertical_symmetric = True
        
        for r, c in cells:
            mirror_c = 2 * center_c - c
            if abs(mirror_c - round(mirror_c)) < 0.1:
                mirror_c = round(mirror_c)
                if (r, mirror_c) not in cells:
                    vertical_symmetric = False
                    break
        
        return vertical_symmetric
    
    def _compute_convex_hull(self, cells: Set[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """计算凸包"""
        if len(cells) <= 2:
            return list(cells)
        
        points = list(cells)
        
        # Graham扫描算法
        def cross_product(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
        
        # 找到最低点（如果有多个，选择最右边的）
        start = min(points, key=lambda p: (p[1], p[0]))
        
        # 按极角排序
        def polar_angle(p):
            return math.atan2(p[1] - start[1], p[0] - start[0])
        
        sorted_points = sorted(points, key=polar_angle)
        
        # 构建凸包
        hull = []
        for p in sorted_points:
            while len(hull) > 1 and cross_product(hull[-2], hull[-1], p) <= 0:
                hull.pop()
            hull.append(p)
        
        return hull
    
    def _detect_holes(self, cells: Set[Tuple[int, int]], 
                     bbox: Tuple[int, int, int, int]) -> List[Set[Tuple[int, int]]]:
        """检测对象内部的孔洞"""
        holes = []
        
        # 创建边界框区域
        min_r, min_c, max_r, max_c = bbox
        
        # 使用洪水填充算法检测封闭区域
        visited = set()
        
        for r in range(min_r, max_r + 1):
            for c in range(min_c, max_c + 1):
                if (r, c) not in cells and (r, c) not in visited:
                    # 尝试洪水填充
                    hole_cells = set()
                    stack = [(r, c)]
                    is_hole = True
                    
                    while stack:
                        cr, cc = stack.pop()
                        if (cr, cc) in visited or (cr, cc) in cells:
                            continue
                        
                        # 检查是否到达边界（不是孔洞）
                        if cr == min_r or cr == max_r or cc == min_c or cc == max_c:
                            is_hole = False
                        
                        visited.add((cr, cc))
                        hole_cells.add((cr, cc))
                        
                        # 添加邻居
                        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                            nr, nc = cr + dr, cc + dc
                            if (min_r <= nr <= max_r and min_c <= nc <= max_c and
                                (nr, nc) not in visited and (nr, nc) not in cells):
                                stack.append((nr, nc))
                    
                    if is_hole and len(hole_cells) >= 1:
                        holes.append(hole_cells)
        
        return holes
    
    def _analyze_advanced_attributes(self, obj: ARCObject, grid: np.ndarray):
        """分析高级属性"""
        # 纹理分析
        obj.attributes['texture'] = self._analyze_texture(obj, grid)
        
        # 连通性分析
        obj.attributes['connectivity'] = self._analyze_connectivity_type(obj.cells)
        
        # 几何复杂度
        obj.attributes['complexity'] = self._calculate_complexity(obj)
        
        # 凸性度量
        obj.attributes['convexity'] = len(obj.cells) / len(obj.convex_hull) if obj.convex_hull else 0
        
        # 紧密度
        obj.attributes['compactness'] = self._calculate_compactness(obj)
    
    def _analyze_texture(self, obj: ARCObject, grid: np.ndarray) -> Dict[str, Any]:
        """分析对象纹理（简化版）"""
        return {
            'uniform': True,  # 在ARC中，对象通常是单色的
            'dominant_color': obj.color
        }
    
    def _analyze_connectivity_type(self, cells: Set[Tuple[int, int]]) -> str:
        """分析连通类型"""
        if len(cells) <= 1:
            return "single"
        
        # 检查是否4连通
        is_4_connected = True
        cells_list = list(cells)
        
        for r, c in cells_list:
            neighbors_4 = [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]
            has_4_neighbor = any((nr, nc) in cells for nr, nc in neighbors_4)
            
            if not has_4_neighbor and len(cells) > 1:
                is_4_connected = False
                break
        
        if is_4_connected:
            return "4_connected"
        
        # 检查是否8连通
        is_8_connected = True
        for r, c in cells_list:
            neighbors_8 = [(r-1, c-1), (r-1, c), (r-1, c+1),
                          (r, c-1),            (r, c+1),
                          (r+1, c-1), (r+1, c), (r+1, c+1)]
            has_8_neighbor = any((nr, nc) in cells for nr, nc in neighbors_8)
            
            if not has_8_neighbor and len(cells) > 1:
                is_8_connected = False
                break
        
        if is_8_connected:
            return "8_connected"
        
        return "disconnected"
    
    def _calculate_complexity(self, obj: ARCObject) -> float:
        """计算几何复杂度"""
        # 基于周长与面积的比值
        if obj.size == 0:
            return 0
        
        perimeter = obj.attributes.get('perimeter', 0)
        complexity = perimeter / math.sqrt(obj.size) if obj.size > 0 else 0
        
        # 考虑孔洞数量
        hole_penalty = len(obj.holes) * 0.5
        
        return complexity + hole_penalty
    
    def _calculate_compactness(self, obj: ARCObject) -> float:
        """计算紧密度"""
        if obj.size == 0:
            return 0
        
        perimeter = obj.attributes.get('perimeter', 0)
        if perimeter == 0:
            return 1.0
        
        # 圆形是最紧密的形状
        ideal_perimeter = 2 * math.sqrt(math.pi * obj.size)
        compactness = ideal_perimeter / perimeter if perimeter > 0 else 0
        
        return min(1.0, compactness)
    
    def _analyze_spatial_relationships(self, objects: List[ARCObject], 
                                     grid: np.ndarray) -> List[Dict[str, Any]]:
        """分析对象间的空间关系"""
        relationships = []
        
        for i, obj1 in enumerate(objects):
            for j, obj2 in enumerate(objects[i+1:], i+1):
                rel = self._compute_pairwise_relationship(obj1, obj2, grid)
                if rel:
                    relationships.append(rel)
        
        return relationships
    
    def _compute_pairwise_relationship(self, obj1: ARCObject, obj2: ARCObject, 
                                     grid: np.ndarray) -> Dict[str, Any]:
        """计算两个对象间的关系"""
        relationship = {
            'obj1_id': obj1.id,
            'obj2_id': obj2.id,
            'distance': self._compute_distance(obj1, obj2),
            'spatial_relations': self._compute_spatial_relations(obj1, obj2),
            'size_relation': self._compute_size_relation(obj1, obj2),
            'color_relation': self._compute_color_relation(obj1, obj2),
            'shape_relation': self._compute_shape_relation(obj1, obj2)
        }
        
        return relationship
    
    def _compute_distance(self, obj1: ARCObject, obj2: ARCObject) -> Dict[str, float]:
        """计算距离度量"""
        # 中心点距离
        centroid_dist = math.sqrt(
            (obj1.centroid[0] - obj2.centroid[0])**2 + 
            (obj1.centroid[1] - obj2.centroid[1])**2
        )
        
        # 最近点距离
        min_dist = float('inf')
        for r1, c1 in obj1.cells:
            for r2, c2 in obj2.cells:
                dist = math.sqrt((r1 - r2)**2 + (c1 - c2)**2)
                min_dist = min(min_dist, dist)
        
        # 边界框距离
        bbox_dist = self._compute_bbox_distance(obj1.bbox, obj2.bbox)
        
        return {
            'centroid_distance': centroid_dist,
            'minimum_distance': min_dist,
            'bbox_distance': bbox_dist
        }
    
    def _compute_spatial_relations(self, obj1: ARCObject, obj2: ARCObject) -> Dict[str, bool]:
        """计算空间关系"""
        return {
            'adjacent': self._are_adjacent(obj1, obj2),
            'overlapping': bool(obj1.cells & obj2.cells),
            'obj1_above_obj2': obj1.bbox[2] < obj2.bbox[0],
            'obj1_below_obj2': obj1.bbox[0] > obj2.bbox[2],
            'obj1_left_of_obj2': obj1.bbox[3] < obj2.bbox[1],
            'obj1_right_of_obj2': obj1.bbox[1] > obj2.bbox[3],
            'horizontally_aligned': abs(obj1.centroid[0] - obj2.centroid[0]) < 0.5,
            'vertically_aligned': abs(obj1.centroid[1] - obj2.centroid[1]) < 0.5,
            'contained': obj1.cells.issubset(obj2.cells) or obj2.cells.issubset(obj1.cells),
            'same_row': self._in_same_row(obj1, obj2),
            'same_column': self._in_same_column(obj1, obj2)
        }
    
    def _are_adjacent(self, obj1: ARCObject, obj2: ARCObject) -> bool:
        """检查两个对象是否相邻"""
        for r1, c1 in obj1.cells:
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nr, nc = r1 + dr, c1 + dc
                if (nr, nc) in obj2.cells:
                    return True
        return False
    
    def _compute_bbox_distance(self, bbox1: Tuple[int, int, int, int], 
                              bbox2: Tuple[int, int, int, int]) -> float:
        """计算边界框距离"""
        # 如果边界框重叠，距离为0
        if not (bbox1[2] < bbox2[0] or bbox2[2] < bbox1[0] or 
                bbox1[3] < bbox2[1] or bbox2[3] < bbox1[1]):
            return 0.0
        
        # 计算最短距离
        dx = max(0, max(bbox1[1] - bbox2[3], bbox2[1] - bbox1[3]))
        dy = max(0, max(bbox1[0] - bbox2[2], bbox2[0] - bbox1[2]))
        
        return math.sqrt(dx*dx + dy*dy)
    
    def _compute_size_relation(self, obj1: ARCObject, obj2: ARCObject) -> Dict[str, Any]:
        """计算大小关系"""
        ratio = obj1.size / obj2.size if obj2.size > 0 else float('inf')
        
        return {
            'size_ratio': ratio,
            'obj1_larger': obj1.size > obj2.size,
            'obj1_smaller': obj1.size < obj2.size,
            'equal_size': obj1.size == obj2.size
        }
    
    def _compute_color_relation(self, obj1: ARCObject, obj2: ARCObject) -> Dict[str, Any]:
        """计算颜色关系"""
        return {
            'same_color': obj1.color == obj2.color,
            'different_color': obj1.color != obj2.color,
            'color_diff': abs(obj1.color - obj2.color)
        }
    
    def _compute_shape_relation(self, obj1: ARCObject, obj2: ARCObject) -> Dict[str, Any]:
        """计算形状关系"""
        return {
            'same_shape': obj1.shape_type == obj2.shape_type,
            'different_shape': obj1.shape_type != obj2.shape_type,
            'obj1_shape': obj1.shape_type,
            'obj2_shape': obj2.shape_type
        }
    
    def _in_same_row(self, obj1: ARCObject, obj2: ARCObject) -> bool:
        """检查是否在同一行"""
        return not (obj1.bbox[2] < obj2.bbox[0] or obj2.bbox[2] < obj1.bbox[0])
    
    def _in_same_column(self, obj1: ARCObject, obj2: ARCObject) -> bool:
        """检查是否在同一列"""
        return not (obj1.bbox[3] < obj2.bbox[1] or obj2.bbox[3] < obj1.bbox[1])
    
    def _analyze_grid_global(self, grid: np.ndarray, objects: List[ARCObject]) -> Dict[str, Any]:
        """全局网格分析"""
        analysis = {
            'grid_shape': grid.shape,
            'total_colors': len(np.unique(grid)),
            'background_ratio': np.sum(grid == self.background_color) / grid.size,
            'object_count': len(objects),
            'object_density': len(objects) / grid.size if grid.size > 0 else 0,
            'color_distribution': self._analyze_color_distribution(grid),
            'symmetry': self._analyze_grid_symmetry(grid),
            'patterns': self._detect_grid_patterns(grid)
        }
        
        return analysis
    
    def _analyze_color_distribution(self, grid: np.ndarray) -> Dict[int, float]:
        """分析颜色分布"""
        unique, counts = np.unique(grid, return_counts=True)
        total = grid.size
        
        distribution = {}
        for color, count in zip(unique, counts):
            distribution[int(color)] = count / total
        
        return distribution
    
    def _analyze_grid_symmetry(self, grid: np.ndarray) -> Dict[str, bool]:
        """分析网格对称性"""
        return {
            'horizontal_symmetry': np.array_equal(grid, np.flipud(grid)),
            'vertical_symmetry': np.array_equal(grid, np.fliplr(grid)),
            'diagonal_symmetry': np.array_equal(grid, grid.T),
            'rotational_symmetry_90': np.array_equal(grid, np.rot90(grid)),
            'rotational_symmetry_180': np.array_equal(grid, np.rot90(grid, 2))
        }
    
    def _detect_grid_patterns(self, grid: np.ndarray) -> List[Dict[str, Any]]:
        """检测网格模式"""
        patterns = []
        
        # 检测重复模式
        repeating_patterns = self._find_repeating_patterns(grid)
        patterns.extend(repeating_patterns)
        
        # 检测边界模式
        boundary_patterns = self._analyze_boundary_patterns(grid)
        patterns.extend(boundary_patterns)
        
        return patterns
    
    def _find_repeating_patterns(self, grid: np.ndarray) -> List[Dict[str, Any]]:
        """查找重复模式"""
        patterns = []
        
        # 简化的重复模式检测
        rows, cols = grid.shape
        
        # 检查行重复
        for size in range(1, rows // 2 + 1):
            if rows % size == 0:
                chunks = [grid[i:i+size] for i in range(0, rows, size)]
                if len(set(tuple(map(tuple, chunk)) for chunk in chunks)) == 1:
                    patterns.append({
                        'type': 'row_repetition',
                        'period': size,
                        'direction': 'vertical'
                    })
        
        # 检查列重复
        for size in range(1, cols // 2 + 1):
            if cols % size == 0:
                chunks = [grid[:, i:i+size] for i in range(0, cols, size)]
                if len(set(tuple(map(tuple, chunk)) for chunk in chunks)) == 1:
                    patterns.append({
                        'type': 'column_repetition',
                        'period': size,
                        'direction': 'horizontal'
                    })
        
        return patterns
    
    def _analyze_boundary_patterns(self, grid: np.ndarray) -> List[Dict[str, Any]]:
        """分析边界模式"""
        patterns = []
        
        # 检查边界是否有特殊模式
        top_row = grid[0, :]
        bottom_row = grid[-1, :]
        left_col = grid[:, 0]
        right_col = grid[:, -1]
        
        # 检查边界是否单色
        if len(np.unique(top_row)) == 1:
            patterns.append({'type': 'uniform_top_border', 'color': int(top_row[0])})
        
        if len(np.unique(bottom_row)) == 1:
            patterns.append({'type': 'uniform_bottom_border', 'color': int(bottom_row[0])})
        
        if len(np.unique(left_col)) == 1:
            patterns.append({'type': 'uniform_left_border', 'color': int(left_col[0])})
        
        if len(np.unique(right_col)) == 1:
            patterns.append({'type': 'uniform_right_border', 'color': int(right_col[0])})
        
        return patterns
    
    def _detect_patterns(self, objects: List[ARCObject], grid: np.ndarray) -> List[Dict[str, Any]]:
        """检测对象模式"""
        patterns = []
        
        if not objects:
            return patterns
        
        # 按颜色分组
        color_groups = defaultdict(list)
        for obj in objects:
            color_groups[obj.color].append(obj)
        
        # 检测每种颜色的模式
        for color, objs in color_groups.items():
            if len(objs) > 1:
                # 检测排列模式
                arrangement_pattern = self._detect_arrangement_pattern(objs)
                if arrangement_pattern:
                    patterns.append({
                        'type': 'object_arrangement',
                        'color': color,
                        'pattern': arrangement_pattern
                    })
                
                # 检测大小模式
                size_pattern = self._detect_size_pattern(objs)
                if size_pattern:
                    patterns.append({
                        'type': 'size_pattern',
                        'color': color,
                        'pattern': size_pattern
                    })
        
        return patterns
    
    def _detect_arrangement_pattern(self, objects: List[ARCObject]) -> Optional[str]:
        """检测对象排列模式"""
        if len(objects) < 2:
            return None
        
        # 按位置排序
        sorted_by_row = sorted(objects, key=lambda obj: obj.centroid[0])
        sorted_by_col = sorted(objects, key=lambda obj: obj.centroid[1])
        
        # 检查是否在同一行
        row_positions = [obj.centroid[0] for obj in objects]
        if max(row_positions) - min(row_positions) < 1.0:  # 允许小的偏差
            return "horizontal_line"
        
        # 检查是否在同一列
        col_positions = [obj.centroid[1] for obj in objects]
        if max(col_positions) - min(col_positions) < 1.0:
            return "vertical_line"
        
        # 检查是否形成对角线
        if self._is_diagonal_arrangement(objects):
            return "diagonal_line"
        
        # 检查是否形成网格
        if self._is_grid_arrangement(objects):
            return "grid"
        
        return "irregular"
    
    def _is_diagonal_arrangement(self, objects: List[ARCObject]) -> bool:
        """检查是否为对角线排列"""
        if len(objects) < 3:
            return False
        
        sorted_objs = sorted(objects, key=lambda obj: (obj.centroid[0], obj.centroid[1]))
        
        # 检查斜率是否一致
        for i in range(2, len(sorted_objs)):
            slope1 = (sorted_objs[1].centroid[1] - sorted_objs[0].centroid[1]) / \
                    max(0.1, sorted_objs[1].centroid[0] - sorted_objs[0].centroid[0])
            slope2 = (sorted_objs[i].centroid[1] - sorted_objs[i-1].centroid[1]) / \
                    max(0.1, sorted_objs[i].centroid[0] - sorted_objs[i-1].centroid[0])
            
            if abs(slope1 - slope2) > 0.1:
                return False
        
        return True
    
    def _is_grid_arrangement(self, objects: List[ARCObject]) -> bool:
        """检查是否为网格排列"""
        if len(objects) < 4:
            return False
        
        # 收集所有行和列位置
        rows = set()
        cols = set()
        
        for obj in objects:
            rows.add(round(obj.centroid[0]))
            cols.add(round(obj.centroid[1]))
        
        # 检查是否形成完整网格
        expected_objects = len(rows) * len(cols)
        return expected_objects == len(objects)
    
    def _detect_size_pattern(self, objects: List[ARCObject]) -> Optional[str]:
        """检测大小模式"""
        if len(objects) < 2:
            return None
        
        sizes = [obj.size for obj in objects]
        
        # 检查是否所有对象大小相同
        if len(set(sizes)) == 1:
            return "uniform_size"
        
        # 检查是否有递增/递减模式
        sorted_sizes = sorted(sizes)
        if sizes == sorted_sizes:
            return "increasing_size"
        elif sizes == sorted_sizes[::-1]:
            return "decreasing_size"
        
        return "varied_size"
    
    def extract_transformation_pattern(self, input_grid: List[List[int]], 
                                     output_grid: List[List[int]]) -> Dict[str, Any]:
        """提取输入输出间的转换模式"""
        logger.debug("分析转换模式")
        
        # 处理两个网格
        input_analysis = self.process_arc_grid(input_grid)
        output_analysis = self.process_arc_grid(output_grid)
        
        # 分析转换
        transformation = {
            'type': self._classify_transformation_type(input_analysis, output_analysis),
            'color_mapping': self._extract_color_mapping(input_grid, output_grid),
            'object_changes': self._analyze_object_changes(input_analysis, output_analysis),
            'spatial_changes': self._analyze_spatial_changes(input_analysis, output_analysis),
            'size_changes': self._analyze_size_changes(input_analysis, output_analysis),
            'pattern_changes': self._analyze_pattern_changes(input_analysis, output_analysis)
        }
        
        return transformation
    
    def _classify_transformation_type(self, input_analysis: Dict, output_analysis: Dict) -> str:
        """分类转换类型"""
        input_shape = input_analysis['grid_shape']
        output_shape = output_analysis['grid_shape']
        
        if input_shape != output_shape:
            return "size_change"
        
        input_colors = set(input_analysis['unique_colors'])
        output_colors = set(output_analysis['unique_colors'])
        
        if input_colors != output_colors:
            return "color_transformation"
        
        input_objects = len(input_analysis['objects'])
        output_objects = len(output_analysis['objects'])
        
        if input_objects != output_objects:
            return "object_count_change"
        
        # 检查是否只是位置变化
        if self._only_position_changed(input_analysis, output_analysis):
            return "position_change"
        
        return "complex_transformation"
    
    def _extract_color_mapping(self, input_grid: List[List[int]], 
                              output_grid: List[List[int]]) -> Dict[int, int]:
        """提取颜色映射"""
        if len(input_grid) != len(output_grid) or \
           len(input_grid[0]) != len(output_grid[0]):
            return {}
        
        color_map = {}
        
        for i in range(len(input_grid)):
            for j in range(len(input_grid[0])):
                input_color = input_grid[i][j]
                output_color = output_grid[i][j]
                
                if input_color in color_map:
                    if color_map[input_color] != output_color:
                        # 不一致的映射，无法建立简单映射
                        return {}
                else:
                    color_map[input_color] = output_color
        
        return color_map
    
    def _analyze_object_changes(self, input_analysis: Dict, output_analysis: Dict) -> Dict[str, Any]:
        """分析对象变化"""
        input_objects = input_analysis['objects']
        output_objects = output_analysis['objects']
        
        changes = {
            'object_count_change': len(output_objects) - len(input_objects),
            'objects_added': max(0, len(output_objects) - len(input_objects)),
            'objects_removed': max(0, len(input_objects) - len(output_objects)),
            'shape_changes': [],
            'color_changes': [],
            'size_changes': []
        }
        
        # 简单的对象匹配（基于位置和大小）
        for input_obj in input_objects:
            best_match = None
            best_distance = float('inf')
            
            for output_obj in output_objects:
                distance = math.sqrt(
                    (input_obj.centroid[0] - output_obj.centroid[0])**2 +
                    (input_obj.centroid[1] - output_obj.centroid[1])**2
                )
                
                if distance < best_distance:
                    best_distance = distance
                    best_match = output_obj
            
            if best_match and best_distance < 5.0:  # 阈值
                # 检查变化
                if input_obj.shape_type != best_match.shape_type:
                    changes['shape_changes'].append({
                        'from': input_obj.shape_type,
                        'to': best_match.shape_type
                    })
                
                if input_obj.color != best_match.color:
                    changes['color_changes'].append({
                        'from': input_obj.color,
                        'to': best_match.color
                    })
                
                if input_obj.size != best_match.size:
                    changes['size_changes'].append({
                        'from': input_obj.size,
                        'to': best_match.size
                    })
        
        return changes
    
    def _analyze_spatial_changes(self, input_analysis: Dict, output_analysis: Dict) -> Dict[str, Any]:
        """分析空间变化"""
        # 简化实现
        return {
            'has_spatial_changes': len(input_analysis['objects']) != len(output_analysis['objects']),
            'symmetry_changed': input_analysis['grid_analysis'].get('symmetry') != 
                              output_analysis['grid_analysis'].get('symmetry')
        }
    
    def _analyze_size_changes(self, input_analysis: Dict, output_analysis: Dict) -> Dict[str, Any]:
        """分析大小变化"""
        return {
            'grid_size_changed': input_analysis['grid_shape'] != output_analysis['grid_shape'],
            'input_shape': input_analysis['grid_shape'],
            'output_shape': output_analysis['grid_shape']
        }
    
    def _analyze_pattern_changes(self, input_analysis: Dict, output_analysis: Dict) -> Dict[str, Any]:
        """分析模式变化"""
        input_patterns = input_analysis.get('patterns', [])
        output_patterns = output_analysis.get('patterns', [])
        
        return {
            'pattern_count_change': len(output_patterns) - len(input_patterns),
            'input_patterns': len(input_patterns),
            'output_patterns': len(output_patterns)
        }
    
    def _only_position_changed(self, input_analysis: Dict, output_analysis: Dict) -> bool:
        """检查是否只有位置发生变化"""
        input_objects = input_analysis['objects']
        output_objects = output_analysis['objects']
        
        if len(input_objects) != len(output_objects):
            return False
        
        # 检查是否所有对象只是位置变化
        # 这需要更复杂的对象匹配算法
        return False  # 简化实现
    
    def _empty_result(self) -> Dict[str, Any]:
        """返回空结果"""
        return {
            'objects': [],
            'relationships': [],
            'grid_analysis': {},
            'patterns': [],
            'grid_shape': (0, 0),
            'unique_colors': [],
            'metadata': {
                'connectivity': self.connectivity,
                'total_objects': 0,
                'extraction_config': self.config
            }
        }
'''

print("继续生成项目的核心文件...")
print("已生成:")
print("✅ core/popper_interface.py - 完整Popper接口")
print("✅ core/anti_unification.py - 反统一算法实现")
print("✅ extraction/object_extractor.py - 高级对象提取器")
print("\\n下一步将生成CEGIS模块、工具模块和配置文件...")
