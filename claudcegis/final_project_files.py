# =====================================================================
# ARC程序合成框架 - 最终文件部分
# =====================================================================

# =====================================================================
# 8. utils/metrics.py - 性能指标模块
# =====================================================================
UTILS_METRICS_PY = '''"""
合成性能指标模块 - 跟踪和分析系统性能
"""
import time
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import json
import statistics

logger = logging.getLogger(__name__)

@dataclass
class TaskMetrics:
    """单个任务的性能指标"""
    task_id: str
    success: bool
    synthesis_time: float
    iterations: int
    program_length: int
    confidence: float
    timestamp: float = field(default_factory=time.time)
    
    # 详细指标
    object_extraction_time: float = 0.0
    popper_time: float = 0.0
    verification_time: float = 0.0
    cegis_iterations: int = 0
    counterexamples_count: int = 0
    
    # 程序质量指标
    program_complexity: float = 0.0
    rule_count: int = 0
    predicate_count: int = 0
    
    # 错误信息
    error_type: Optional[str] = None
    error_message: Optional[str] = None

@dataclass
class SynthesisMetrics:
    """系统整体性能指标"""
    
    # 基本统计
    total_tasks: int = 0
    successful_tasks: int = 0
    failed_tasks: int = 0
    total_time: float = 0.0
    
    # 详细记录
    task_metrics: List[TaskMetrics] = field(default_factory=list)
    
    # 时间分布
    time_distribution: Dict[str, float] = field(default_factory=lambda: {
        'object_extraction': 0.0,
        'popper_synthesis': 0.0,
        'verification': 0.0,
        'cegis_overhead': 0.0
    })
    
    # 成功率分析
    success_by_type: Dict[str, Dict[str, int]] = field(default_factory=dict)
    success_by_difficulty: Dict[str, Dict[str, int]] = field(default_factory=dict)
    
    # 性能趋势
    performance_trend: List[Dict[str, Any]] = field(default_factory=list)
    
    def add_task_result(self, task_metrics: TaskMetrics):
        """添加任务结果"""
        self.total_tasks += 1
        
        if task_metrics.success:
            self.successful_tasks += 1
        else:
            self.failed_tasks += 1
        
        self.total_time += task_metrics.synthesis_time
        self.task_metrics.append(task_metrics)
        
        # 更新时间分布
        self.time_distribution['object_extraction'] += task_metrics.object_extraction_time
        self.time_distribution['popper_synthesis'] += task_metrics.popper_time
        self.time_distribution['verification'] += task_metrics.verification_time
        
        # 更新性能趋势
        self.performance_trend.append({
            'task_index': self.total_tasks,
            'success': task_metrics.success,
            'time': task_metrics.synthesis_time,
            'timestamp': task_metrics.timestamp
        })
    
    def add_task_result_simple(self, task_id: str, success: bool, synthesis_time: float,
                             iterations: int, program: str = "", confidence: float = 0.0,
                             error_type: str = None, error_message: str = None):
        """简化的添加任务结果方法"""
        
        metrics = TaskMetrics(
            task_id=task_id,
            success=success,
            synthesis_time=synthesis_time,
            iterations=iterations,
            program_length=len(program.split('\\n')) if program else 0,
            confidence=confidence,
            error_type=error_type,
            error_message=error_message
        )
        
        # 分析程序复杂度
        if program:
            metrics.program_complexity = self._calculate_program_complexity(program)
            metrics.rule_count = self._count_rules(program)
            metrics.predicate_count = self._count_predicates(program)
        
        self.add_task_result(metrics)
    
    def get_success_rate(self) -> float:
        """获取总体成功率"""
        if self.total_tasks == 0:
            return 0.0
        return self.successful_tasks / self.total_tasks
    
    def get_average_time(self) -> float:
        """获取平均合成时间"""
        if self.total_tasks == 0:
            return 0.0
        return self.total_time / self.total_tasks
    
    def get_success_rate_by_type(self, task_type: str) -> float:
        """按任务类型获取成功率"""
        if task_type not in self.success_by_type:
            return 0.0
        
        stats = self.success_by_type[task_type]
        total = stats.get('total', 0)
        success = stats.get('success', 0)
        
        return success / total if total > 0 else 0.0
    
    def get_time_statistics(self) -> Dict[str, float]:
        """获取时间统计"""
        if not self.task_metrics:
            return {}
        
        times = [m.synthesis_time for m in self.task_metrics]
        
        return {
            'mean': statistics.mean(times),
            'median': statistics.median(times),
            'std_dev': statistics.stdev(times) if len(times) > 1 else 0.0,
            'min': min(times),
            'max': max(times),
            'total': sum(times)
        }
    
    def get_iteration_statistics(self) -> Dict[str, float]:
        """获取迭代次数统计"""
        if not self.task_metrics:
            return {}
        
        iterations = [m.iterations for m in self.task_metrics if m.iterations > 0]
        
        if not iterations:
            return {}
        
        return {
            'mean': statistics.mean(iterations),
            'median': statistics.median(iterations),
            'std_dev': statistics.stdev(iterations) if len(iterations) > 1 else 0.0,
            'min': min(iterations),
            'max': max(iterations)
        }
    
    def get_confidence_statistics(self) -> Dict[str, float]:
        """获取置信度统计"""
        if not self.task_metrics:
            return {}
        
        confidences = [m.confidence for m in self.task_metrics 
                      if m.success and m.confidence > 0]
        
        if not confidences:
            return {}
        
        return {
            'mean': statistics.mean(confidences),
            'median': statistics.median(confidences),
            'std_dev': statistics.stdev(confidences) if len(confidences) > 1 else 0.0,
            'min': min(confidences),
            'max': max(confidences)
        }
    
    def get_error_analysis(self) -> Dict[str, Any]:
        """获取错误分析"""
        error_types = defaultdict(int)
        error_messages = defaultdict(int)
        
        for metrics in self.task_metrics:
            if not metrics.success:
                if metrics.error_type:
                    error_types[metrics.error_type] += 1
                if metrics.error_message:
                    # 简化错误消息以便分组
                    simplified_msg = self._simplify_error_message(metrics.error_message)
                    error_messages[simplified_msg] += 1
        
        return {
            'error_types': dict(error_types),
            'common_errors': dict(error_messages),
            'failure_rate': self.failed_tasks / self.total_tasks if self.total_tasks > 0 else 0
        }
    
    def get_performance_trends(self) -> Dict[str, Any]:
        """获取性能趋势分析"""
        if len(self.performance_trend) < 2:
            return {}
        
        # 计算滑动窗口成功率
        window_size = min(10, len(self.performance_trend) // 4)
        recent_success = []
        
        for i in range(len(self.performance_trend) - window_size + 1):
            window = self.performance_trend[i:i + window_size]
            success_count = sum(1 for entry in window if entry['success'])
            recent_success.append(success_count / window_size)
        
        # 时间趋势
        times = [entry['time'] for entry in self.performance_trend]
        time_trend = 'improving' if times[-5:] < times[:5] else 'stable'
        
        return {
            'success_rate_trend': recent_success,
            'time_trend': time_trend,
            'recent_performance': recent_success[-1] if recent_success else 0.0,
            'improvement_rate': (recent_success[-1] - recent_success[0]) / len(recent_success) 
                              if len(recent_success) > 1 else 0.0
        }
    
    def update_task_type_stats(self, task_id: str, task_type: str, success: bool):
        """更新按任务类型的统计"""
        if task_type not in self.success_by_type:
            self.success_by_type[task_type] = {'total': 0, 'success': 0}
        
        self.success_by_type[task_type]['total'] += 1
        if success:
            self.success_by_type[task_type]['success'] += 1
    
    def update_difficulty_stats(self, task_id: str, difficulty: str, success: bool):
        """更新按难度的统计"""
        if difficulty not in self.success_by_difficulty:
            self.success_by_difficulty[difficulty] = {'total': 0, 'success': 0}
        
        self.success_by_difficulty[difficulty]['total'] += 1
        if success:
            self.success_by_difficulty[difficulty]['success'] += 1
    
    def print_summary(self):
        """打印性能总结"""
        print("=" * 60)
        print("🔍 ARC程序合成性能总结")
        print("=" * 60)
        
        # 基本统计
        print(f"📊 基本统计:")
        print(f"   总任务数: {self.total_tasks}")
        print(f"   成功任务: {self.successful_tasks}")
        print(f"   失败任务: {self.failed_tasks}")
        print(f"   成功率: {self.get_success_rate():.2%}")
        print(f"   总时间: {self.total_time:.2f}秒")
        print(f"   平均时间: {self.get_average_time():.2f}秒")
        
        # 时间统计
        time_stats = self.get_time_statistics()
        if time_stats:
            print(f"\\n⏱️  时间分布:")
            print(f"   平均: {time_stats['mean']:.2f}秒")
            print(f"   中位数: {time_stats['median']:.2f}秒")
            print(f"   标准差: {time_stats['std_dev']:.2f}秒")
            print(f"   范围: {time_stats['min']:.2f} - {time_stats['max']:.2f}秒")
        
        # 迭代统计
        iter_stats = self.get_iteration_statistics()
        if iter_stats:
            print(f"\\n🔄 迭代统计:")
            print(f"   平均迭代: {iter_stats['mean']:.1f}")
            print(f"   中位数迭代: {iter_stats['median']:.1f}")
            print(f"   最大迭代: {iter_stats['max']}")
        
        # 置信度统计
        conf_stats = self.get_confidence_statistics()
        if conf_stats:
            print(f"\\n🎯 置信度分析:")
            print(f"   平均置信度: {conf_stats['mean']:.2%}")
            print(f"   置信度范围: {conf_stats['min']:.2%} - {conf_stats['max']:.2%}")
        
        # 按类型的成功率
        if self.success_by_type:
            print(f"\\n📋 按任务类型成功率:")
            for task_type, stats in self.success_by_type.items():
                rate = stats['success'] / stats['total'] if stats['total'] > 0 else 0
                print(f"   {task_type}: {rate:.2%} ({stats['success']}/{stats['total']})")
        
        # 错误分析
        error_analysis = self.get_error_analysis()
        if error_analysis['error_types']:
            print(f"\\n❌ 主要错误类型:")
            for error_type, count in error_analysis['error_types'].items():
                print(f"   {error_type}: {count} 次")
        
        print("=" * 60)
    
    def save_to_file(self, filename: str):
        """保存指标到文件"""
        data = {
            'summary': {
                'total_tasks': self.total_tasks,
                'successful_tasks': self.successful_tasks,
                'failed_tasks': self.failed_tasks,
                'success_rate': self.get_success_rate(),
                'total_time': self.total_time,
                'average_time': self.get_average_time()
            },
            'time_statistics': self.get_time_statistics(),
            'iteration_statistics': self.get_iteration_statistics(),
            'confidence_statistics': self.get_confidence_statistics(),
            'error_analysis': self.get_error_analysis(),
            'performance_trends': self.get_performance_trends(),
            'success_by_type': dict(self.success_by_type),
            'success_by_difficulty': dict(self.success_by_difficulty),
            'task_details': [
                {
                    'task_id': m.task_id,
                    'success': m.success,
                    'synthesis_time': m.synthesis_time,
                    'iterations': m.iterations,
                    'confidence': m.confidence,
                    'program_length': m.program_length,
                    'error_type': m.error_type
                }
                for m in self.task_metrics
            ]
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"性能指标已保存到: {filename}")
    
    def load_from_file(self, filename: str):
        """从文件加载指标"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 重建基本统计
            summary = data.get('summary', {})
            self.total_tasks = summary.get('total_tasks', 0)
            self.successful_tasks = summary.get('successful_tasks', 0)
            self.failed_tasks = summary.get('failed_tasks', 0)
            self.total_time = summary.get('total_time', 0.0)
            
            # 重建任务详情
            self.task_metrics = []
            for task_data in data.get('task_details', []):
                metrics = TaskMetrics(
                    task_id=task_data['task_id'],
                    success=task_data['success'],
                    synthesis_time=task_data['synthesis_time'],
                    iterations=task_data['iterations'],
                    program_length=task_data['program_length'],
                    confidence=task_data['confidence'],
                    error_type=task_data.get('error_type')
                )
                self.task_metrics.append(metrics)
            
            # 重建分类统计
            self.success_by_type = data.get('success_by_type', {})
            self.success_by_difficulty = data.get('success_by_difficulty', {})
            
            logger.info(f"性能指标已从文件加载: {filename}")
            
        except Exception as e:
            logger.error(f"加载性能指标失败: {str(e)}")
    
    def reset(self):
        """重置所有指标"""
        self.total_tasks = 0
        self.successful_tasks = 0
        self.failed_tasks = 0
        self.total_time = 0.0
        self.task_metrics.clear()
        self.time_distribution = {
            'object_extraction': 0.0,
            'popper_synthesis': 0.0,
            'verification': 0.0,
            'cegis_overhead': 0.0
        }
        self.success_by_type.clear()
        self.success_by_difficulty.clear()
        self.performance_trend.clear()
        
        logger.info("性能指标已重置")
    
    # 私有辅助方法
    def _calculate_program_complexity(self, program: str) -> float:
        """计算程序复杂度"""
        if not program:
            return 0.0
        
        lines = [line.strip() for line in program.split('\\n') if line.strip()]
        rule_count = len(lines)
        avg_length = sum(len(line) for line in lines) / max(1, rule_count)
        
        # 简单的复杂度计算
        complexity = rule_count * 0.5 + avg_length * 0.01
        return min(10.0, complexity)  # 限制在0-10范围
    
    def _count_rules(self, program: str) -> int:
        """统计规则数量"""
        lines = [line.strip() for line in program.split('\\n') 
                if line.strip() and not line.strip().startswith('%')]
        return len(lines)
    
    def _count_predicates(self, program: str) -> int:
        """统计谓词数量"""
        import re
        predicates = set()
        
        pattern = r'\\b([a-z][a-zA-Z0-9_]*)\\s*\\('
        matches = re.findall(pattern, program)
        
        for match in matches:
            predicates.add(match)
        
        return len(predicates)
    
    def _simplify_error_message(self, error_msg: str) -> str:
        """简化错误消息用于分组"""
        if not error_msg:
            return "unknown_error"
        
        # 移除特定的数字和路径
        import re
        simplified = re.sub(r'\\d+', 'N', error_msg)
        simplified = re.sub(r'/[^\\s]+', '/PATH', simplified)
        simplified = simplified[:100]  # 限制长度
        
        return simplified
'''

# =====================================================================
# 9. utils/logging.py - 日志工具模块
# =====================================================================
UTILS_LOGGING_PY = '''"""
日志工具模块 - 提供统一的日志配置和管理
"""
import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import datetime

def setup_logging(level: str = "INFO",
                 format_str: Optional[str] = None,
                 log_file: Optional[str] = None,
                 console_output: bool = True,
                 file_rotation: bool = True,
                 max_file_size: int = 10 * 1024 * 1024,  # 10MB
                 backup_count: int = 5) -> logging.Logger:
    """
    设置统一的日志配置
    
    Args:
        level: 日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_str: 日志格式字符串
        log_file: 日志文件路径
        console_output: 是否输出到控制台
        file_rotation: 是否启用文件轮转
        max_file_size: 最大文件大小（字节）
        backup_count: 备份文件数量
        
    Returns:
        配置好的根logger
    """
    
    # 默认格式
    if format_str is None:
        format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # 创建根logger
    root_logger = logging.getLogger()
    
    # 清除现有处理器
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # 设置日志级别
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    root_logger.setLevel(numeric_level)
    
    # 创建格式器
    formatter = logging.Formatter(format_str)
    
    handlers = []
    
    # 控制台处理器
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(formatter)
        handlers.append(console_handler)
    
    # 文件处理器
    if log_file:
        # 确保日志目录存在
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        if file_rotation:
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=max_file_size,
                backupCount=backup_count,
                encoding='utf-8'
            )
        else:
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
        
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)
    
    # 添加处理器到根logger
    for handler in handlers:
        root_logger.addHandler(handler)
    
    # 设置第三方库日志级别
    _configure_third_party_loggers()
    
    # 记录配置信息
    root_logger.info(f"日志系统已配置 - 级别: {level}, 文件: {log_file}")
    
    return root_logger

def _configure_third_party_loggers():
    """配置第三方库的日志级别"""
    third_party_configs = {
        'matplotlib': logging.WARNING,
        'PIL': logging.WARNING,
        'urllib3': logging.WARNING,
        'requests': logging.WARNING,
        'numpy': logging.WARNING,
        'scipy': logging.WARNING,
        'sklearn': logging.WARNING
    }
    
    for logger_name, level in third_party_configs.items():
        logging.getLogger(logger_name).setLevel(level)

class ContextualLogger:
    """
    上下文感知的日志记录器
    自动添加任务ID等上下文信息
    """
    
    def __init__(self, name: str, context: Optional[Dict[str, Any]] = None):
        self.logger = logging.getLogger(name)
        self.context = context or {}
    
    def update_context(self, **kwargs):
        """更新上下文信息"""
        self.context.update(kwargs)
    
    def clear_context(self):
        """清除上下文信息"""
        self.context.clear()
    
    def _format_message(self, message: str) -> str:
        """格式化消息，添加上下文信息"""
        if not self.context:
            return message
        
        context_str = " | ".join(f"{k}={v}" for k, v in self.context.items())
        return f"[{context_str}] {message}"
    
    def debug(self, message: str, **kwargs):
        """调试日志"""
        self.logger.debug(self._format_message(message), **kwargs)
    
    def info(self, message: str, **kwargs):
        """信息日志"""
        self.logger.info(self._format_message(message), **kwargs)
    
    def warning(self, message: str, **kwargs):
        """警告日志"""
        self.logger.warning(self._format_message(message), **kwargs)
    
    def error(self, message: str, **kwargs):
        """错误日志"""
        self.logger.error(self._format_message(message), **kwargs)
    
    def critical(self, message: str, **kwargs):
        """严重错误日志"""
        self.logger.critical(self._format_message(message), **kwargs)

class PerformanceLogger:
    """
    性能日志记录器
    专门用于记录性能相关信息
    """
    
    def __init__(self, name: str = "performance"):
        self.logger = logging.getLogger(name)
        self.start_times = {}
    
    def start_timer(self, operation: str):
        """开始计时"""
        import time
        self.start_times[operation] = time.time()
        self.logger.debug(f"开始操作: {operation}")
    
    def end_timer(self, operation: str) -> float:
        """结束计时并记录"""
        import time
        
        if operation not in self.start_times:
            self.logger.warning(f"未找到操作的开始时间: {operation}")
            return 0.0
        
        elapsed = time.time() - self.start_times[operation]
        del self.start_times[operation]
        
        self.logger.info(f"操作完成: {operation} - 用时: {elapsed:.3f}秒")
        return elapsed
    
    def log_memory_usage(self, operation: str = "current"):
        """记录内存使用情况"""
        try:
            import psutil
            import os
            
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            
            self.logger.info(
                f"内存使用 [{operation}]: "
                f"RSS={memory_info.rss / 1024 / 1024:.1f}MB, "
                f"VMS={memory_info.vms / 1024 / 1024:.1f}MB"
            )
            
        except ImportError:
            self.logger.debug("psutil未安装，无法记录内存使用")
        except Exception as e:
            self.logger.warning(f"记录内存使用失败: {str(e)}")
    
    def log_system_info(self):
        """记录系统信息"""
        try:
            import platform
            import psutil
            
            self.logger.info(
                f"系统信息: "
                f"OS={platform.system()} {platform.release()}, "
                f"Python={platform.python_version()}, "
                f"CPU核心={psutil.cpu_count()}, "
                f"内存={psutil.virtual_memory().total / 1024 / 1024 / 1024:.1f}GB"
            )
            
        except Exception as e:
            self.logger.warning(f"记录系统信息失败: {str(e)}")

class LogCapture:
    """
    日志捕获器
    用于在测试中捕获日志消息
    """
    
    def __init__(self, logger_name: str = None, level: int = logging.DEBUG):
        self.logger_name = logger_name
        self.level = level
        self.handler = None
        self.records = []
    
    def __enter__(self):
        """进入上下文管理器"""
        import io
        
        # 创建内存处理器
        self.stream = io.StringIO()
        self.handler = logging.StreamHandler(self.stream)
        self.handler.setLevel(self.level)
        
        # 添加到指定logger或根logger
        if self.logger_name:
            logger = logging.getLogger(self.logger_name)
        else:
            logger = logging.getLogger()
        
        logger.addHandler(self.handler)
        self.logger = logger
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出上下文管理器"""
        if self.handler:
            self.logger.removeHandler(self.handler)
    
    def get_logs(self) -> List[str]:
        """获取捕获的日志"""
        if self.handler:
            return self.stream.getvalue().strip().split('\\n')
        return []
    
    def has_log_containing(self, text: str) -> bool:
        """检查是否包含特定文本的日志"""
        logs = self.get_logs()
        return any(text in log for log in logs)

def create_file_logger(name: str, filename: str, level: str = "INFO") -> logging.Logger:
    """
    创建专用文件日志记录器
    
    Args:
        name: 日志记录器名称
        filename: 日志文件名
        level: 日志级别
        
    Returns:
        配置好的logger
    """
    logger = logging.getLogger(name)
    
    # 避免重复添加处理器
    if logger.handlers:
        return logger
    
    logger.setLevel(getattr(logging, level.upper()))
    
    # 确保目录存在
    log_path = Path(filename)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 创建文件处理器
    handler = logging.handlers.RotatingFileHandler(
        filename,
        maxBytes=5 * 1024 * 1024,  # 5MB
        backupCount=3,
        encoding='utf-8'
    )
    
    # 设置格式
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)
    
    logger.addHandler(handler)
    
    return logger

def log_function_call(func):
    """
    函数调用日志装饰器
    """
    import functools
    import time
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)
        
        # 记录函数调用
        func_name = f"{func.__module__}.{func.__name__}"
        logger.debug(f"调用函数: {func_name}")
        
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time
            logger.debug(f"函数完成: {func_name} - 用时: {elapsed:.3f}秒")
            return result
            
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(
                f"函数异常: {func_name} - 用时: {elapsed:.3f}秒 - 错误: {str(e)}"
            )
            raise
    
    return wrapper

def configure_debug_logging():
    """配置调试级别的详细日志"""
    setup_logging(
        level="DEBUG",
        format_str="%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
        log_file="logs/debug.log",
        console_output=True
    )

def configure_production_logging():
    """配置生产环境的日志"""
    setup_logging(
        level="INFO",
        format_str="%(asctime)s - %(levelname)s - %(message)s",
        log_file="logs/production.log",
        console_output=False,
        max_file_size=50 * 1024 * 1024,  # 50MB
        backup_count=10
    )
'''

# =====================================================================
# 10. 配置文件示例 - config/spatial.yaml 和 config/complex.yaml
# =====================================================================
CONFIG_SPATIAL_YAML = '''# ARC程序合成框架 - 空间推理任务配置

# Popper配置 - 优化用于空间推理
popper:
  popper_path: "./popper"
  timeout: 600  # 增加超时时间用于复杂空间推理
  solver: "rc2"
  max_vars: 12  # 增加变量数量用于空间关系
  max_body: 15  # 增加体部大小
  max_rules: 8  # 允许更多规则
  noisy: false
  enable_recursion: true  # 启用递归用于复杂空间模式

# 对象提取配置 - 专门用于空间分析
extraction:
  connectivity: 8  # 8连通用于更精确的空间分析
  min_object_size: 1
  background_color: 0
  analyze_shapes: true
  detect_patterns: true
  extract_holes: true
  
  # 空间分析特定配置
  spatial_analysis:
    detect_symmetry: true
    analyze_convexity: true
    compute_moments: true
    extract_skeletons: false  # 可选的骨架提取
    
  # 高级几何特征
  geometric_features:
    compute_convex_hull: true
    analyze_orientation: true
    detect_corners: true
    measure_compactness: true

# CEGIS配置 - 适应空间推理复杂性
cegis:
  max_iterations: 40  # 增加迭代次数
  synthesis_timeout: 600
  verification_timeout: 120
  enable_parallel: false
  
  # 空间约束生成
  spatial_constraints:
    position_constraints: true
    orientation_constraints: true
    scale_constraints: true
    topology_constraints: true

# 反统一配置 - 空间模式泛化
anti_unification:
  max_generalization_depth: 8
  preserve_structure: true
  enable_type_constraints: true
  min_pattern_support: 2
  
  # 空间模式特定配置
  spatial_patterns:
    preserve_topology: true
    generalize_positions: true
    abstract_orientations: false
    maintain_scale_relations: true

# 验证器配置
oracle:
  validation_method: "exact_match"
  tolerance: 0.0
  enable_partial_credit: false
  
  # 空间验证特定
  spatial_validation:
    check_topology: true
    verify_transformations: true
    validate_symmetries: true

# 性能配置
performance:
  cache_enabled: true
  cache_size: 2000  # 增大缓存用于复杂空间对象
  parallel_tasks: 1
  memory_limit_mb: 4096  # 增加内存限制

# 日志配置
logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: "logs/spatial_synthesis.log"
  console_output: true
  
  # 空间分析特定日志
  spatial_logging:
    log_object_details: true
    log_transformations: true
    log_spatial_relations: true
'''

CONFIG_COMPLEX_YAML = '''# ARC程序合成框架 - 复杂任务配置

# Popper配置 - 最大化表达能力
popper:
  popper_path: "./popper"
  timeout: 1200  # 20分钟超时用于复杂任务
  solver: "rc2"
  max_vars: 20   # 大量变量用于复杂逻辑
  max_body: 25   # 大型规则体
  max_rules: 15  # 允许复杂程序结构
  noisy: true    # 启用详细输出
  enable_recursion: true
  
  # 高级Popper选项
  advanced_options:
    use_metarules: true
    enable_predicate_invention: true
    max_invented_predicates: 5
    use_functional_constraints: true

# 对象提取配置 - 全功能分析
extraction:
  connectivity: 8
  min_object_size: 1
  background_color: 0
  analyze_shapes: true
  detect_patterns: true
  extract_holes: true
  
  # 高级分析功能
  advanced_analysis:
    hierarchical_decomposition: true
    multi_scale_analysis: true
    texture_analysis: true
    statistical_features: true
    topological_invariants: true
    
  # 模式识别
  pattern_recognition:
    periodic_patterns: true
    fractal_patterns: false
    symmetry_groups: true
    template_matching: true

# CEGIS配置 - 高强度合成
cegis:
  max_iterations: 100  # 大量迭代用于复杂问题
  synthesis_timeout: 1200
  verification_timeout: 300
  enable_parallel: true
  parallel_workers: 4
  
  # 高级合成策略
  synthesis_strategies:
    progressive_complexity: true
    multiple_hypotheses: true
    ensemble_methods: true
    adaptive_search: true
    
  # 复杂约束生成
  constraint_generation:
    semantic_constraints: true
    temporal_constraints: true
    causal_constraints: true
    probabilistic_constraints: false

# 反统一配置 - 深度泛化
anti_unification:
  max_generalization_depth: 15
  preserve_structure: false  # 允许更激进的泛化
  enable_type_constraints: true
  min_pattern_support: 1  # 更宽松的支持要求
  
  # 高级泛化选项
  advanced_generalization:
    hierarchical_abstraction: true
    categorical_abstraction: true
    functional_abstraction: true
    statistical_generalization: true
    
  # 泛化质量控制
  quality_control:
    overgeneralization_penalty: 0.3
    specificity_bonus: 0.1
    coherence_weight: 0.5

# 验证器配置 - 严格验证
oracle:
  validation_method: "comprehensive"
  tolerance: 0.0
  enable_partial_credit: true
  partial_credit_threshold: 0.8
  
  # 多层验证
  validation_layers:
    syntactic_validation: true
    semantic_validation: true
    pragmatic_validation: true
    performance_validation: true
    
  # 验证策略
  validation_strategies:
    cross_validation: true
    bootstrap_validation: true
    adversarial_testing: true

# 元学习配置
meta_learning:
  enable_meta_learning: true
  learn_from_failures: true
  adapt_search_strategy: true
  transfer_knowledge: true
  
  # 元特征
  meta_features:
    task_complexity_estimation: true
    solution_pattern_prediction: true
    resource_requirement_estimation: true

# 性能配置 - 高性能设置
performance:
  cache_enabled: true
  cache_size: 10000
  parallel_tasks: 4
  memory_limit_mb: 8192
  
  # 优化选项
  optimization:
    lazy_evaluation: true
    memoization: true
    pruning_strategies: ["complexity", "redundancy", "contradiction"]
    resource_monitoring: true
    
  # 负载均衡
  load_balancing:
    dynamic_task_distribution: true
    adaptive_resource_allocation: true
    priority_queue: true

# 调试和诊断
debugging:
  enable_detailed_tracing: true
  save_intermediate_results: true
  profile_performance: true
  generate_execution_reports: true
  
  # 诊断工具
  diagnostics:
    memory_profiling: true
    cpu_profiling: false
    io_monitoring: true
    bottleneck_detection: true

# 日志配置 - 详细记录
logging:
  level: "DEBUG"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
  file: "logs/complex_synthesis.log"
  console_output: true
  
  # 分层日志
  specialized_loggers:
    synthesis: "logs/synthesis_detailed.log"
    verification: "logs/verification_detailed.log"
    performance: "logs/performance_detailed.log"
    errors: "logs/errors_detailed.log"
    
  # 日志轮转
  rotation:
    max_file_size: "100MB"
    backup_count: 20
    compression: true
'''

# =====================================================================
# 11. 示例文件 - examples/simple_tasks/color_change.json
# =====================================================================
EXAMPLE_COLOR_CHANGE_JSON = '''{
  "task_id": "color_change_example",
  "description": "简单颜色转换示例 - 将蓝色(1)替换为红色(2)",
  "source": "generated",
  "difficulty": "easy",
  "type": "color_transformation",
  
  "train": [
    {
      "input": [
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0]
      ],
      "output": [
        [0, 2, 0],
        [2, 2, 2],
        [0, 2, 0]
      ]
    },
    {
      "input": [
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 0]
      ],
      "output": [
        [0, 2, 2],
        [2, 0, 2],
        [2, 2, 0]
      ]
    },
    {
      "input": [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
      ],
      "output": [
        [2, 0, 0],
        [0, 2, 0],
        [0, 0, 2]
      ]
    }
  ],
  
  "test": [
    {
      "input": [
        [1, 0, 1],
        [0, 1, 0],
        [1, 0, 1]
      ],
      "output": [
        [2, 0, 2],
        [0, 2, 0],
        [2, 0, 2]
      ]
    }
  ],
  
  "metadata": {
    "colors_used": [0, 1, 2],
    "grid_size": [3, 3],
    "transformation_rule": "replace_color(1, 2)",
    "expected_program": "transform(Input, Output) :- change_all_color(Input, 1, 2, Output).",
    "verification_notes": "所有蓝色(1)单元格应变为红色(2)，其他保持不变",
    "tags": ["color_mapping", "simple", "deterministic"],
    "creation_date": "2024-01-01",
    "author": "ARC Framework Generator"
  }
}'''

# =====================================================================
# 12. Popper模板文件 - popper_files/templates/spatial_bias.pl
# =====================================================================
SPATIAL_BIAS_PL = '''% 空间推理任务偏置文件模板
% 专门用于处理涉及位置、方向和空间关系的ARC任务

% ===== 头谓词定义 =====
head_pred(transform,2).
head_pred(spatial_pattern,1).

% ===== 基础空间谓词 =====

% 网格和单元格操作
body_pred(grid,1).
body_pred(cell,3).
body_pred(grid_cell,4).           % grid_cell(Grid,Row,Col,Color)
body_pred(grid_size,3).           % grid_size(Grid,Width,Height)
body_pred(in_bounds,3).           % in_bounds(X,Y,Grid)

% 位置和坐标
body_pred(position,3).            % position(Object,X,Y)
body_pred(coordinate,2).          % coordinate(X,Y)
body_pred(manhattan_distance,4).  % manhattan_distance(X1,Y1,X2,Y2,Dist)
body_pred(euclidean_distance,4).  % euclidean_distance(X1,Y1,X2,Y2,Dist)

% 方向关系
body_pred(north_of,2).            % north_of(Obj1,Obj2)
body_pred(south_of,2).            % south_of(Obj1,Obj2)
body_pred(east_of,2).             % east_of(Obj1,Obj2)
body_pred(west_of,2).             % west_of(Obj1,Obj2)
body_pred(northeast_of,2).        % northeast_of(Obj1,Obj2)
body_pred(northwest_of,2).        % northwest_of(Obj1,Obj2)
body_pred(southeast_of,2).        % southeast_of(Obj1,Obj2)
body_pred(southwest_of,2).        % southwest_of(Obj1,Obj2)

% 相邻关系
body_pred(adjacent_4,2).          % adjacent_4(Cell1,Cell2) - 4连通
body_pred(adjacent_8,2).          % adjacent_8(Cell1,Cell2) - 8连通
body_pred(diagonal_adjacent,2).   % diagonal_adjacent(Cell1,Cell2)
body_pred(orthogonal_adjacent,2). % orthogonal_adjacent(Cell1,Cell2)

% 对齐关系
body_pred(horizontally_aligned,2). % horizontally_aligned(Obj1,Obj2)
body_pred(vertically_aligned,2).   % vertically_aligned(Obj1,Obj2)
body_pred(diagonally_aligned,2).   % diagonally_aligned(Obj1,Obj2)
body_pred(collinear,3).            % collinear(Obj1,Obj2,Obj3)

% ===== 空间变换谓词 =====

% 基本变换
body_pred(translate,4).           % translate(Grid,DX,DY,NewGrid)
body_pred(rotate_90,2).           % rotate_90(Grid,NewGrid)
body_pred(rotate_180,2).          % rotate_180(Grid,NewGrid)
body_pred(rotate_270,2).          % rotate_270(Grid,NewGrid)
body_pred(rotate,3).              % rotate(Grid,Angle,NewGrid)

% 反射变换
body_pred(reflect_horizontal,2).  % reflect_horizontal(Grid,NewGrid)
body_pred(reflect_vertical,2).    % reflect_vertical(Grid,NewGrid)
body_pred(reflect_diagonal_main,2). % reflect_diagonal_main(Grid,NewGrid)
body_pred(reflect_diagonal_anti,2). % reflect_diagonal_anti(Grid,NewGrid)
body_pred(reflect,3).             % reflect(Grid,Axis,NewGrid)

% 缩放变换
body_pred(scale,3).               % scale(Grid,Factor,NewGrid)
body_pred(scale_up,3).            % scale_up(Grid,Factor,NewGrid)
body_pred(scale_down,3).          % scale_down(Grid,Factor,NewGrid)

% 复合变换
body_pred(compose_transforms,3).  % compose_transforms(Transform1,Transform2,Result)
body_pred(inverse_transform,2).   % inverse_transform(Transform,Inverse)

% ===== 对象和形状谓词 =====

% 对象检测
body_pred(detect_objects,2).      % detect_objects(Grid,Objects)
body_pred(object_at,3).           % object_at(Grid,Position,Object)
body_pred(object_color,2).        % object_color(Object,Color)
body_pred(object_size,2).         % object_size(Object,Size)
body_pred(object_shape,2).        % object_shape(Object,Shape)

% 形状分类
body_pred(rectangle,1).           % rectangle(Object)
body_pred(square,1).              % square(Object)
body_pred(line,1).                % line(Object)
body_pred(point,1).               % point(Object)
body_pred(l_shape,1).             % l_shape(Object)
body_pred(t_shape,1).             % t_shape(Object)
body_pred(cross,1).               % cross(Object)
body_pred(circle,1).              % circle(Object)

% 形状属性
body_pred(convex,1).              % convex(Object)
body_pred(concave,1).             % concave(Object)
body_pred(symmetric,1).           % symmetric(Object)
body_pred(regular,1).             % regular(Object)
body_pred(solid,1).               % solid(Object)
body_pred(hollow,1).              % hollow(Object)

% ===== 空间模式谓词 =====

% 排列模式
body_pred(linear_arrangement,2).  % linear_arrangement(Objects,Direction)
body_pred(grid_arrangement,2).    % grid_arrangement(Objects,Pattern)
body_pred(circular_arrangement,1). % circular_arrangement(Objects)
body_pred(random_arrangement,1).   % random_arrangement(Objects)

% 重复模式
body_pred(repeating_pattern,2).   % repeating_pattern(Grid,Period)
body_pred(alternating_pattern,1). % alternating_pattern(Objects)
body_pred(spiral_pattern,1).      % spiral_pattern(Objects)

% 对称模式
body_pred(has_symmetry,2).        % has_symmetry(Grid,SymmetryType)
body_pred(mirror_symmetry,2).     % mirror_symmetry(Grid,Axis)
body_pred(rotational_symmetry,2). % rotational_symmetry(Grid,Order)
body_pred(point_symmetry,1).      % point_symmetry(Grid)

% ===== 空间约束和关系 =====

% 拓扑关系
body_pred(contains,2).            % contains(Container,Contained)
body_pred(overlaps,2).            % overlaps(Obj1,Obj2)
body_pred(touches,2).             % touches(Obj1,Obj2)
body_pred(separates,3).           % separates(Separator,Obj1,Obj2)
body_pred(surrounds,2).           % surrounds(Outer,Inner)

% 相对位置
body_pred(between,3).             % between(Middle,End1,End2)
body_pred(closest_to,3).          % closest_to(Target,Object,Others)
body_pred(farthest_from,3).       % farthest_from(Target,Object,Others)
body_pred(center_of,2).           % center_of(Object,Container)
body_pred(corner_of,2).           % corner_of(Object,Container)
body_pred(edge_of,2).             % edge_of(Object,Container)

% ===== 类型定义 =====

% 基础类型
type(transform,(grid,grid)).
type(grid,(list)).
type(cell,(int,int,int)).
type(position,(int,int)).
type(object,(term)).

% 空间关系类型
type(adjacent_4,(cell,cell)).
type(adjacent_8,(cell,cell)).
type(north_of,(object,object)).
type(south_of,(object,object)).
type(east_of,(object,object)).
type(west_of,(object,object)).

% 变换类型
type(translate,(grid,int,int,grid)).
type(rotate,(grid,int,grid)).
type(reflect,(grid,atom,grid)).
type(scale,(grid,int,grid)).

% 形状类型
type(rectangle,(object)).
type(square,(object)).
type(line,(object)).
type(circle,(object)).

% ===== 方向定义 =====

% 主要谓词方向
direction(transform,(in,out)).
direction(translate,(in,in,in,out)).
direction(rotate,(in,in,out)).
direction(reflect,(in,in,out)).
direction(detect_objects,(in,out)).

% 关系谓词方向
direction(adjacent_4,(in,in)).
direction(north_of,(in,in)).
direction(contains,(in,in)).
direction(overlaps,(in,in)).

% ===== 控制参数 =====

max_vars(12).
max_body(15).
max_rules(8).

% 允许更复杂的空间推理
allow_singletons.
enable_recursion.

% ===== 空间推理特定约束 =====

% 距离约束
:- manhattan_distance(X1,Y1,X2,Y2,D), D < 0.
:- euclidean_distance(X1,Y1,X2,Y2,D), D < 0.

% 边界约束
:- position(Object,X,Y), X < 0.
:- position(Object,X,Y), Y < 0.

% 一致性约束
:- north_of(A,B), south_of(A,B).
:- east_of(A,B), west_of(A,B).
:- contains(A,B), contains(B,A), A \\= B.

% 传递性约束（有限）
:- north_of(A,B), north_of(B,C), south_of(A,C).
:- east_of(A,B), east_of(B,C), west_of(A,C).

% ===== 领域特定元规则 =====

% 空间变换元规则
% MetaRule: 连续变换
metarule(spatial_composition, [P,Q,R], (P(A,B) :- Q(A,C), R(C,B))).

% MetaRule: 条件空间变换
metarule(conditional_spatial, [P,Q,R], (P(A,B) :- Q(A), R(A,B))).

% MetaRule: 空间关系传播
metarule(spatial_propagation, [P,Q], (P(A,C) :- P(A,B), Q(B,C))).
'''

print("项目文件生成完成！")
print()
print("=" * 70)
print("🎉 ARC程序合成框架 - 完整项目已生成")
print("=" * 70)
print()
print("📁 生成的文件包括:")
print("✅ 核心模块: synthesis_engine, popper_interface, anti_unification")
print("✅ 提取模块: object_extractor, spatial_predicates, transformations") 
print("✅ CEGIS模块: synthesizer, verifier, counterexample")
print("✅ 工具模块: arc_loader, metrics, logging")
print("✅ 配置文件: default.yaml, spatial.yaml, complex.yaml")
print("✅ 示例文件: color_change.json, basic_usage.py")
print("✅ Popper模板: bias文件, background知识文件")
print("✅ 测试文件: 单元测试和集成测试")
print("✅ 文档文件: README.md, setup.py, requirements.txt")
print()
print("🚀 使用步骤:")
print("1. 创建项目目录结构")
print("2. 复制相应代码到对应文件")
print("3. 安装依赖: pip install -r requirements.txt")
print("4. 配置Popper路径")
print("5. 运行演示: python main.py --demo")
print()
print("📚 详细文档和扩展指南请参考README.md文件")
