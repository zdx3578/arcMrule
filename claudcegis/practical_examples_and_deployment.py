# =====================================================================
# ARC程序合成框架 - 实用示例与部署指南
# =====================================================================

# =====================================================================
# 1. 完整的端到端示例
# =====================================================================
COMPLETE_EXAMPLE_PY = '''#!/usr/bin/env python3
"""
完整的端到端ARC程序合成示例
演示从任务创建到结果分析的完整流程
"""

import sys
import time
import json
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))

from arc_synthesis_framework import ARCSynthesisEngine, SynthesisTask
from arc_synthesis_framework.utils.arc_loader import ARCDataLoader
from arc_synthesis_framework.utils.metrics import SynthesisMetrics
from arc_synthesis_framework.utils.logging import setup_logging, PerformanceLogger

def run_complete_example():
    """运行完整的ARC程序合成示例"""
    
    print("🚀 ARC程序合成框架 - 完整示例")
    print("=" * 60)
    
    # 1. 设置日志
    setup_logging(
        level="INFO",
        log_file="logs/complete_example.log",
        console_output=True
    )
    
    # 2. 初始化性能监控
    perf_logger = PerformanceLogger()
    metrics = SynthesisMetrics()
    
    # 3. 初始化合成引擎
    print("\\n📦 初始化合成引擎...")
    perf_logger.start_timer("initialization")
    
    engine = ARCSynthesisEngine("config/default.yaml")
    loader = ARCDataLoader()
    
    perf_logger.end_timer("initialization")
    perf_logger.log_system_info()
    
    # 4. 创建测试任务集
    print("\\n🎯 创建测试任务...")
    tasks = create_test_tasks(loader)
    
    print(f"创建了 {len(tasks)} 个测试任务")
    for i, task in enumerate(tasks, 1):
        print(f"  {i}. {task.task_id}: {task.metadata.get('description', '未知')}")
    
    # 5. 运行合成实验
    print("\\n🔬 开始合成实验...")
    results = run_synthesis_experiments(engine, tasks, metrics, perf_logger)
    
    # 6. 分析结果
    print("\\n📊 分析实验结果...")
    analyze_results(results, metrics)
    
    # 7. 生成报告
    print("\\n📋 生成详细报告...")
    generate_comprehensive_report(results, metrics, "reports/complete_example_report.json")
    
    print("\\n✅ 示例运行完成！")
    print("📁 检查以下文件获取详细结果:")
    print("   - logs/complete_example.log")
    print("   - reports/complete_example_report.json")

def create_test_tasks(loader):
    """创建多样化的测试任务"""
    tasks = []
    
    # 1. 简单颜色转换任务
    tasks.append(loader.create_simple_task("color_1_to_2"))
    
    # 2. 空间移动任务
    tasks.append(loader.create_spatial_task("translate_right"))
    
    # 3. 复杂模式任务
    tasks.append(create_pattern_completion_task())
    
    # 4. 对称性任务
    tasks.append(create_symmetry_task())
    
    # 5. 填充任务
    tasks.append(create_hole_filling_task())
    
    return tasks

def create_pattern_completion_task():
    """创建模式补全任务"""
    return SynthesisTask(
        task_id="pattern_completion",
        train_pairs=[
            # 十字模式补全
            ([[0, 1, 0], [1, 0, 1], [0, 1, 0]], 
             [[0, 1, 0], [1, 1, 1], [0, 1, 0]]),
            ([[1, 0, 1], [0, 0, 0], [1, 0, 1]], 
             [[1, 0, 1], [0, 1, 0], [1, 0, 1]]),
        ],
        test_pairs=[
            ([[0, 0, 0], [1, 0, 1], [0, 0, 0]], 
             [[0, 1, 0], [1, 0, 1], [0, 1, 0]])
        ],
        metadata={
            "description": "在十字模式中心填充",
            "type": "pattern_completion",
            "difficulty": "medium"
        }
    )

def create_symmetry_task():
    """创建对称性任务"""
    return SynthesisTask(
        task_id="mirror_symmetry",
        train_pairs=[
            # 水平镜像
            ([[1, 0, 0], [0, 1, 0], [0, 0, 0]], 
             [[0, 0, 1], [0, 1, 0], [0, 0, 0]]),
            ([[1, 1, 0], [1, 0, 0], [0, 0, 0]], 
             [[0, 1, 1], [0, 0, 1], [0, 0, 0]]),
        ],
        test_pairs=[
            ([[1, 0, 1], [0, 0, 0], [1, 0, 0]], 
             [[1, 0, 1], [0, 0, 0], [0, 0, 1]])
        ],
        metadata={
            "description": "水平镜像反射",
            "type": "spatial_transformation",
            "difficulty": "medium"
        }
    )

def create_hole_filling_task():
    """创建洞填充任务"""
    return SynthesisTask(
        task_id="hole_filling",
        train_pairs=[
            # 填充0为1
            ([[1, 0, 1], [0, 0, 0], [1, 0, 1]], 
             [[1, 1, 1], [1, 1, 1], [1, 1, 1]]),
            ([[2, 0, 2], [0, 2, 0], [2, 0, 2]], 
             [[2, 2, 2], [2, 2, 2], [2, 2, 2]]),
        ],
        test_pairs=[
            ([[3, 0, 3], [0, 3, 0], [3, 0, 3]], 
             [[3, 3, 3], [3, 3, 3], [3, 3, 3]])
        ],
        metadata={
            "description": "用周围颜色填充空洞",
            "type": "hole_filling",
            "difficulty": "easy"
        }
    )

def run_synthesis_experiments(engine, tasks, metrics, perf_logger):
    """运行合成实验"""
    results = []
    
    for i, task in enumerate(tasks, 1):
        print(f"\\n🔄 处理任务 {i}/{len(tasks)}: {task.task_id}")
        
        # 记录任务开始
        perf_logger.start_timer(f"task_{task.task_id}")
        
        try:
            # 运行合成
            result = engine.synthesize_program(task)
            
            # 记录结果
            synthesis_time = perf_logger.end_timer(f"task_{task.task_id}")
            
            # 更新指标
            metrics.add_task_result_simple(
                task.task_id,
                result.success,
                result.synthesis_time,
                result.iterations,
                result.program or "",
                result.confidence,
                error_message=result.error_message
            )
            
            # 更新任务类型统计
            task_type = task.metadata.get('type', 'unknown')
            metrics.update_task_type_stats(task.task_id, task_type, result.success)
            
            # 更新难度统计
            difficulty = task.metadata.get('difficulty', 'unknown')
            metrics.update_difficulty_stats(task.task_id, difficulty, result.success)
            
            # 显示结果
            status = "✅" if result.success else "❌"
            print(f"   {status} {task.task_id}: {'成功' if result.success else '失败'}")
            if result.success:
                print(f"      程序: {result.program}")
                print(f"      置信度: {result.confidence:.2%}")
                print(f"      迭代: {result.iterations}")
            else:
                print(f"      错误: {result.error_message}")
            
            print(f"      时间: {synthesis_time:.2f}秒")
            
            results.append({
                'task': task,
                'result': result,
                'synthesis_time': synthesis_time
            })
            
        except Exception as e:
            print(f"   ❌ {task.task_id}: 异常 - {str(e)}")
            metrics.add_task_result_simple(
                task.task_id, False, 0.0, 0, "", 0.0,
                error_type="exception", error_message=str(e)
            )
            
            results.append({
                'task': task,
                'result': None,
                'error': str(e)
            })
        
        # 记录内存使用
        perf_logger.log_memory_usage(f"after_task_{i}")
    
    return results

def analyze_results(results, metrics):
    """分析实验结果"""
    
    print("\\n📈 实验结果分析:")
    print("-" * 40)
    
    # 基本统计
    metrics.print_summary()
    
    # 详细分析
    successful_results = [r for r in results if r.get('result') and r['result'].success]
    failed_results = [r for r in results if not (r.get('result') and r['result'].success)]
    
    print(f"\\n🔍 详细分析:")
    print(f"   成功任务数: {len(successful_results)}")
    print(f"   失败任务数: {len(failed_results)}")
    
    if successful_results:
        print(f"\\n✅ 成功任务详情:")
        for r in successful_results:
            task = r['task']
            result = r['result']
            print(f"   - {task.task_id}: {result.confidence:.1%} 置信度, {result.iterations} 迭代")
    
    if failed_results:
        print(f"\\n❌ 失败任务详情:")
        for r in failed_results:
            task = r['task']
            if r.get('result'):
                print(f"   - {task.task_id}: {r['result'].error_message}")
            else:
                print(f"   - {task.task_id}: {r.get('error', '未知错误')}")
    
    # 性能趋势
    trends = metrics.get_performance_trends()
    if trends:
        print(f"\\n📊 性能趋势:")
        print(f"   最近成功率: {trends.get('recent_performance', 0):.1%}")
        print(f"   改进率: {trends.get('improvement_rate', 0):.3f}")

def generate_comprehensive_report(results, metrics, output_file):
    """生成综合报告"""
    
    # 确保目录存在
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    report = {
        "experiment_info": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "framework_version": "0.1.0",
            "total_tasks": len(results)
        },
        "overall_metrics": {
            "success_rate": metrics.get_success_rate(),
            "average_time": metrics.get_average_time(),
            "total_time": metrics.total_time
        },
        "detailed_statistics": {
            "time_stats": metrics.get_time_statistics(),
            "iteration_stats": metrics.get_iteration_statistics(),
            "confidence_stats": metrics.get_confidence_statistics(),
            "error_analysis": metrics.get_error_analysis()
        },
        "task_results": [],
        "recommendations": generate_recommendations(results, metrics)
    }
    
    # 添加任务详情
    for r in results:
        task = r['task']
        result = r.get('result')
        
        task_report = {
            "task_id": task.task_id,
            "description": task.metadata.get('description', ''),
            "type": task.metadata.get('type', 'unknown'),
            "difficulty": task.metadata.get('difficulty', 'unknown'),
            "success": result.success if result else False,
            "synthesis_time": r.get('synthesis_time', 0.0)
        }
        
        if result and result.success:
            task_report.update({
                "program": result.program,
                "confidence": result.confidence,
                "iterations": result.iterations
            })
        elif result:
            task_report["error"] = result.error_message
        else:
            task_report["error"] = r.get('error', '未知错误')
        
        report["task_results"].append(task_report)
    
    # 保存报告
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"📋 综合报告已保存到: {output_file}")

def generate_recommendations(results, metrics):
    """生成改进建议"""
    recommendations = []
    
    success_rate = metrics.get_success_rate()
    
    if success_rate < 0.5:
        recommendations.append({
            "type": "performance",
            "message": "成功率较低，建议检查Popper配置和背景知识",
            "action": "增加训练示例或简化任务复杂度"
        })
    
    avg_time = metrics.get_average_time()
    if avg_time > 60:
        recommendations.append({
            "type": "efficiency",
            "message": "平均合成时间较长，建议优化性能",
            "action": "减少搜索空间或增加超时限制"
        })
    
    error_analysis = metrics.get_error_analysis()
    if error_analysis.get('failure_rate', 0) > 0.3:
        recommendations.append({
            "type": "reliability",
            "message": "失败率较高，建议改进错误处理",
            "action": "检查常见错误类型并改进约束生成"
        })
    
    return recommendations

if __name__ == "__main__":
    run_complete_example()
'''

# =====================================================================
# 2. 批量处理和评估脚本
# =====================================================================
BATCH_EVALUATION_PY = '''#!/usr/bin/env python3
"""
批量评估脚本
用于在ARC数据集上进行大规模评估
"""

import argparse
import json
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

from arc_synthesis_framework import ARCSynthesisEngine
from arc_synthesis_framework.utils.arc_loader import ARCDataLoader
from arc_synthesis_framework.utils.metrics import SynthesisMetrics

def evaluate_single_task(task_file, config_path, timeout=300):
    """评估单个任务"""
    try:
        # 初始化（在子进程中）
        engine = ARCSynthesisEngine(config_path)
        loader = ARCDataLoader()
        
        # 加载任务
        task = loader.load_task(Path(task_file).stem)
        
        # 运行合成
        start_time = time.time()
        result = engine.synthesize_program(task)
        execution_time = time.time() - start_time
        
        return {
            "task_id": task.task_id,
            "success": result.success,
            "synthesis_time": execution_time,
            "iterations": result.iterations,
            "confidence": result.confidence,
            "program": result.program if result.success else None,
            "error": result.error_message if not result.success else None,
            "task_type": task.metadata.get('type', 'unknown'),
            "difficulty": task.metadata.get('difficulty', 'unknown')
        }
        
    except Exception as e:
        return {
            "task_id": Path(task_file).stem,
            "success": False,
            "error": str(e),
            "synthesis_time": 0.0,
            "iterations": 0,
            "confidence": 0.0
        }

def run_batch_evaluation(task_dir, config_path, output_file, max_workers=None, 
                        max_tasks=None):
    """运行批量评估"""
    
    print(f"🔍 扫描任务目录: {task_dir}")
    
    # 查找所有任务文件
    task_files = list(Path(task_dir).glob("*.json"))
    
    if max_tasks:
        task_files = task_files[:max_tasks]
    
    print(f"📊 找到 {len(task_files)} 个任务文件")
    
    if not task_files:
        print("❌ 未找到任务文件")
        return
    
    # 设置工作进程数
    if max_workers is None:
        max_workers = min(mp.cpu_count(), len(task_files))
    
    print(f"🚀 使用 {max_workers} 个工作进程开始评估...")
    
    results = []
    start_time = time.time()
    
    # 并行处理
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_file = {
            executor.submit(evaluate_single_task, task_file, config_path): task_file
            for task_file in task_files
        }
        
        # 收集结果
        completed = 0
        for future in as_completed(future_to_file):
            try:
                result = future.result()
                results.append(result)
                completed += 1
                
                # 显示进度
                if completed % 10 == 0 or completed == len(task_files):
                    elapsed = time.time() - start_time
                    rate = completed / elapsed if elapsed > 0 else 0
                    eta = (len(task_files) - completed) / rate if rate > 0 else 0
                    
                    print(f"进度: {completed}/{len(task_files)} "
                          f"({completed/len(task_files):.1%}) "
                          f"速度: {rate:.1f} 任务/秒 "
                          f"预计剩余: {eta:.0f}秒")
                
            except Exception as e:
                print(f"❌ 任务处理失败: {str(e)}")
                completed += 1
    
    total_time = time.time() - start_time
    print(f"\\n✅ 批量评估完成！总用时: {total_time:.1f}秒")
    
    # 分析结果
    analyze_batch_results(results)
    
    # 保存结果
    save_batch_results(results, output_file)
    
    return results

def analyze_batch_results(results):
    """分析批量结果"""
    
    print("\\n📈 批量评估结果分析:")
    print("=" * 50)
    
    total_tasks = len(results)
    successful_tasks = sum(1 for r in results if r['success'])
    
    print(f"总任务数: {total_tasks}")
    print(f"成功任务: {successful_tasks}")
    print(f"失败任务: {total_tasks - successful_tasks}")
    print(f"总体成功率: {successful_tasks/total_tasks:.2%}")
    
    # 按类型分析
    type_stats = {}
    difficulty_stats = {}
    
    for result in results:
        task_type = result.get('task_type', 'unknown')
        difficulty = result.get('difficulty', 'unknown')
        success = result['success']
        
        if task_type not in type_stats:
            type_stats[task_type] = {'total': 0, 'success': 0}
        type_stats[task_type]['total'] += 1
        if success:
            type_stats[task_type]['success'] += 1
        
        if difficulty not in difficulty_stats:
            difficulty_stats[difficulty] = {'total': 0, 'success': 0}
        difficulty_stats[difficulty]['total'] += 1
        if success:
            difficulty_stats[difficulty]['success'] += 1
    
    # 显示按类型的统计
    if type_stats:
        print(f"\\n📊 按任务类型成功率:")
        for task_type, stats in type_stats.items():
            rate = stats['success'] / stats['total'] if stats['total'] > 0 else 0
            print(f"  {task_type}: {rate:.2%} ({stats['success']}/{stats['total']})")
    
    # 显示按难度的统计
    if difficulty_stats:
        print(f"\\n📊 按难度成功率:")
        for difficulty, stats in difficulty_stats.items():
            rate = stats['success'] / stats['total'] if stats['total'] > 0 else 0
            print(f"  {difficulty}: {rate:.2%} ({stats['success']}/{stats['total']})")
    
    # 时间统计
    successful_results = [r for r in results if r['success']]
    if successful_results:
        times = [r['synthesis_time'] for r in successful_results]
        print(f"\\n⏱️  成功任务时间统计:")
        print(f"  平均时间: {sum(times)/len(times):.2f}秒")
        print(f"  最短时间: {min(times):.2f}秒")
        print(f"  最长时间: {max(times):.2f}秒")
    
    # 置信度统计
    confident_results = [r for r in successful_results if r.get('confidence', 0) > 0]
    if confident_results:
        confidences = [r['confidence'] for r in confident_results]
        print(f"\\n🎯 置信度统计:")
        print(f"  平均置信度: {sum(confidences)/len(confidences):.2%}")
        print(f"  最高置信度: {max(confidences):.2%}")
        print(f"  最低置信度: {min(confidences):.2%}")

def save_batch_results(results, output_file):
    """保存批量结果"""
    
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 准备报告数据
    report = {
        "evaluation_info": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_tasks": len(results),
            "successful_tasks": sum(1 for r in results if r['success']),
            "overall_success_rate": sum(1 for r in results if r['success']) / len(results)
        },
        "results": results,
        "summary": {
            "by_type": {},
            "by_difficulty": {}
        }
    }
    
    # 生成汇总统计
    for result in results:
        task_type = result.get('task_type', 'unknown')
        difficulty = result.get('difficulty', 'unknown')
        success = result['success']
        
        # 按类型汇总
        if task_type not in report["summary"]["by_type"]:
            report["summary"]["by_type"][task_type] = {'total': 0, 'success': 0}
        report["summary"]["by_type"][task_type]['total'] += 1
        if success:
            report["summary"]["by_type"][task_type]['success'] += 1
        
        # 按难度汇总
        if difficulty not in report["summary"]["by_difficulty"]:
            report["summary"]["by_difficulty"][difficulty] = {'total': 0, 'success': 0}
        report["summary"]["by_difficulty"][difficulty]['total'] += 1
        if success:
            report["summary"]["by_difficulty"][difficulty]['success'] += 1
    
    # 计算成功率
    for type_stats in report["summary"]["by_type"].values():
        type_stats['success_rate'] = type_stats['success'] / type_stats['total']
    
    for diff_stats in report["summary"]["by_difficulty"].values():
        diff_stats['success_rate'] = diff_stats['success'] / diff_stats['total']
    
    # 保存到文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\\n💾 批量评估结果已保存到: {output_file}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="ARC批量评估工具")
    parser.add_argument("task_dir", help="任务目录路径")
    parser.add_argument("--config", default="config/default.yaml", 
                       help="配置文件路径")
    parser.add_argument("--output", default="results/batch_evaluation.json", 
                       help="输出文件路径")
    parser.add_argument("--workers", type=int, 
                       help="并行工作进程数（默认为CPU核心数）")
    parser.add_argument("--max_tasks", type=int, 
                       help="最大任务数量限制")
    
    args = parser.parse_args()
    
    # 检查输入
    if not Path(args.task_dir).exists():
        print(f"❌ 任务目录不存在: {args.task_dir}")
        return 1
    
    if not Path(args.config).exists():
        print(f"❌ 配置文件不存在: {args.config}")
        return 1
    
    try:
        # 运行批量评估
        results = run_batch_evaluation(
            args.task_dir, 
            args.config, 
            args.output,
            max_workers=args.workers,
            max_tasks=args.max_tasks
        )
        
        print(f"\\n🎉 批量评估成功完成！")
        return 0
        
    except Exception as e:
        print(f"❌ 批量评估失败: {str(e)}")
        return 1

if __name__ == "__main__":
    exit(main())
'''

# =====================================================================
# 3. 性能基准测试脚本
# =====================================================================
BENCHMARK_PY = '''#!/usr/bin/env python3
"""
性能基准测试脚本
用于测试不同配置下的系统性能
"""

import time
import statistics
from pathlib import Path
import matplotlib.pyplot as plt
import json

from arc_synthesis_framework import ARCSynthesisEngine
from arc_synthesis_framework.utils.arc_loader import ARCDataLoader
from arc_synthesis_framework.utils.metrics import SynthesisMetrics

class PerformanceBenchmark:
    """性能基准测试类"""
    
    def __init__(self):
        self.loader = ARCDataLoader()
        self.results = {}
    
    def run_configuration_benchmark(self, configs, test_tasks=None, iterations=3):
        """运行配置基准测试"""
        
        if test_tasks is None:
            test_tasks = self._create_benchmark_tasks()
        
        print(f"🏃‍♂️ 开始性能基准测试")
        print(f"配置数量: {len(configs)}")
        print(f"测试任务: {len(test_tasks)}")
        print(f"迭代次数: {iterations}")
        print("=" * 50)
        
        for config_name, config_path in configs.items():
            print(f"\\n🔧 测试配置: {config_name}")
            
            config_results = []
            
            for iteration in range(iterations):
                print(f"  迭代 {iteration + 1}/{iterations}")
                
                iteration_results = self._run_single_iteration(
                    config_path, test_tasks
                )
                config_results.append(iteration_results)
            
            # 聚合结果
            self.results[config_name] = self._aggregate_results(config_results)
            
            # 显示初步结果
            avg_success_rate = self.results[config_name]['avg_success_rate']
            avg_time = self.results[config_name]['avg_synthesis_time']
            print(f"  平均成功率: {avg_success_rate:.2%}")
            print(f"  平均时间: {avg_time:.2f}秒")
        
        return self.results
    
    def _create_benchmark_tasks(self):
        """创建基准测试任务集"""
        tasks = []
        
        # 简单任务
        tasks.extend([
            self.loader.create_simple_task("bench_color_1"),
            self.loader.create_spatial_task("bench_translate_1"),
        ])
        
        # 中等复杂度任务
        tasks.extend([
            self.loader.create_random_task(
                "bench_medium_1", "color_transformation", "medium"
            ),
            self.loader.create_random_task(
                "bench_medium_2", "spatial_transformation", "medium"
            ),
        ])
        
        # 复杂任务
        tasks.extend([
            self.loader.create_random_task(
                "bench_hard_1", "color_transformation", "hard"
            ),
            self.loader.create_random_task(
                "bench_hard_2", "spatial_transformation", "hard"
            ),
        ])
        
        return tasks
    
    def _run_single_iteration(self, config_path, tasks):
        """运行单次迭代"""
        engine = ARCSynthesisEngine(config_path)
        results = []
        
        for task in tasks:
            start_time = time.time()
            
            try:
                result = engine.synthesize_program(task)
                synthesis_time = time.time() - start_time
                
                results.append({
                    'task_id': task.task_id,
                    'success': result.success,
                    'synthesis_time': synthesis_time,
                    'iterations': result.iterations,
                    'confidence': result.confidence
                })
                
            except Exception as e:
                synthesis_time = time.time() - start_time
                results.append({
                    'task_id': task.task_id,
                    'success': False,
                    'synthesis_time': synthesis_time,
                    'iterations': 0,
                    'confidence': 0.0,
                    'error': str(e)
                })
        
        return results
    
    def _aggregate_results(self, iteration_results):
        """聚合多次迭代的结果"""
        all_results = []
        for iteration in iteration_results:
            all_results.extend(iteration)
        
        # 计算统计信息
        successful_results = [r for r in all_results if r['success']]
        
        return {
            'total_tasks': len(all_results),
            'successful_tasks': len(successful_results),
            'avg_success_rate': len(successful_results) / len(all_results),
            'avg_synthesis_time': statistics.mean([r['synthesis_time'] for r in all_results]),
            'avg_iterations': statistics.mean([r['iterations'] for r in successful_results]) if successful_results else 0,
            'avg_confidence': statistics.mean([r['confidence'] for r in successful_results]) if successful_results else 0,
            'time_std': statistics.stdev([r['synthesis_time'] for r in all_results]) if len(all_results) > 1 else 0,
            'detailed_results': all_results
        }
    
    def generate_benchmark_report(self, output_file="reports/benchmark_report.json"):
        """生成基准测试报告"""
        
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        
        report = {
            "benchmark_info": {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "configurations_tested": list(self.results.keys())
            },
            "results": self.results,
            "comparison": self._generate_comparison(),
            "recommendations": self._generate_performance_recommendations()
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\\n📊 基准测试报告已保存到: {output_file}")
    
    def _generate_comparison(self):
        """生成配置比较"""
        if not self.results:
            return {}
        
        comparison = {}
        
        # 找出最佳配置
        best_success_rate = max(r['avg_success_rate'] for r in self.results.values())
        best_speed = min(r['avg_synthesis_time'] for r in self.results.values())
        
        for config_name, results in self.results.items():
            comparison[config_name] = {
                'success_rate_rank': self._rank_by_metric('avg_success_rate', config_name),
                'speed_rank': self._rank_by_metric('avg_synthesis_time', config_name, reverse=True),
                'is_best_success_rate': results['avg_success_rate'] == best_success_rate,
                'is_fastest': results['avg_synthesis_time'] == best_speed
            }
        
        return comparison
    
    def _rank_by_metric(self, metric, config_name, reverse=False):
        """按指标排名"""
        values = [(name, results[metric]) for name, results in self.results.items()]
        values.sort(key=lambda x: x[1], reverse=reverse)
        
        for rank, (name, _) in enumerate(values, 1):
            if name == config_name:
                return rank
        
        return len(values)
    
    def _generate_performance_recommendations(self):
        """生成性能建议"""
        recommendations = []
        
        if not self.results:
            return recommendations
        
        # 找出性能问题
        avg_success_rates = [r['avg_success_rate'] for r in self.results.values()]
        avg_times = [r['avg_synthesis_time'] for r in self.results.values()]
        
        overall_success_rate = statistics.mean(avg_success_rates)
        overall_avg_time = statistics.mean(avg_times)
        
        if overall_success_rate < 0.5:
            recommendations.append({
                "type": "success_rate",
                "message": "整体成功率较低",
                "suggestion": "考虑增加训练示例数量或简化任务复杂度"
            })
        
        if overall_avg_time > 120:
            recommendations.append({
                "type": "performance",
                "message": "平均合成时间较长",
                "suggestion": "考虑减少搜索空间或优化Popper配置"
            })
        
        # 找出最佳配置
        best_config = max(self.results.items(), 
                         key=lambda x: x[1]['avg_success_rate'])
        
        recommendations.append({
            "type": "best_configuration",
            "message": f"推荐使用配置: {best_config[0]}",
            "suggestion": f"该配置达到 {best_config[1]['avg_success_rate']:.2%} 成功率"
        })
        
        return recommendations
    
    def plot_benchmark_results(self, output_dir="reports/plots"):
        """绘制基准测试结果图表"""
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        if not self.results:
            print("❌ 没有结果可以绘制")
            return
        
        # 成功率对比图
        self._plot_success_rate_comparison(output_dir)
        
        # 时间对比图
        self._plot_time_comparison(output_dir)
        
        # 综合性能图
        self._plot_performance_scatter(output_dir)
    
    def _plot_success_rate_comparison(self, output_dir):
        """绘制成功率对比图"""
        configs = list(self.results.keys())
        success_rates = [self.results[c]['avg_success_rate'] for c in configs]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(configs, success_rates)
        plt.title('Configuration Success Rate Comparison')
        plt.ylabel('Success Rate')
        plt.ylim(0, 1)
        
        # 添加数值标签
        for bar, rate in zip(bars, success_rates):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{rate:.2%}', ha='center', va='bottom')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/success_rate_comparison.png", dpi=300)
        plt.close()
    
    def _plot_time_comparison(self, output_dir):
        """绘制时间对比图"""
        configs = list(self.results.keys())
        avg_times = [self.results[c]['avg_synthesis_time'] for c in configs]
        time_stds = [self.results[c]['time_std'] for c in configs]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(configs, avg_times, yerr=time_stds, capsize=5)
        plt.title('Configuration Time Comparison')
        plt.ylabel('Average Synthesis Time (seconds)')
        
        # 添加数值标签
        for bar, time_val in zip(bars, avg_times):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(time_stds)/10,
                    f'{time_val:.1f}s', ha='center', va='bottom')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/time_comparison.png", dpi=300)
        plt.close()
    
    def _plot_performance_scatter(self, output_dir):
        """绘制性能散点图"""
        plt.figure(figsize=(10, 8))
        
        for config_name, results in self.results.items():
            x = results['avg_synthesis_time']
            y = results['avg_success_rate']
            plt.scatter(x, y, s=100, label=config_name, alpha=0.7)
            plt.annotate(config_name, (x, y), xytext=(5, 5), 
                        textcoords='offset points', fontsize=9)
        
        plt.xlabel('Average Synthesis Time (seconds)')
        plt.ylabel('Success Rate')
        plt.title('Performance Trade-off: Success Rate vs Time')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/performance_scatter.png", dpi=300)
        plt.close()
        
        print(f"📈 性能图表已保存到: {output_dir}/")

def run_standard_benchmark():
    """运行标准基准测试"""
    
    benchmark = PerformanceBenchmark()
    
    # 定义测试配置
    configs = {
        "default": "config/default.yaml",
        "spatial": "config/spatial.yaml",
        "complex": "config/complex.yaml"
    }
    
    # 运行基准测试
    print("🚀 开始标准基准测试...")
    results = benchmark.run_configuration_benchmark(configs, iterations=2)
    
    # 生成报告和图表
    benchmark.generate_benchmark_report()
    benchmark.plot_benchmark_results()
    
    print("\\n🎉 基准测试完成！")
    print("📁 查看以下文件获取详细结果:")
    print("   - reports/benchmark_report.json")
    print("   - reports/plots/")

if __name__ == "__main__":
    run_standard_benchmark()
'''

# =====================================================================
# 4. 部署配置文件
# =====================================================================
DOCKER_COMPOSE_YML = '''# Docker Compose配置文件
# 用于部署ARC程序合成框架

version: '3.8'

services:
  arc-synthesis:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: arc-synthesis-main
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
      - ./results:/app/results
      - ./config:/app/config
    environment:
      - PYTHONPATH=/app
      - LOG_LEVEL=INFO
    ports:
      - "8000:8000"  # 如果有Web接口
    restart: unless-stopped
    
  arc-worker:
    build:
      context: .
      dockerfile: Dockerfile
    deploy:
      replicas: 3
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
      - ./config:/app/config
    environment:
      - PYTHONPATH=/app
      - LOG_LEVEL=INFO
      - WORKER_MODE=true
    depends_on:
      - redis
    restart: unless-stopped
    
  redis:
    image: redis:7-alpine
    container_name: arc-redis
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped
    
  postgres:
    image: postgres:15-alpine
    container_name: arc-postgres
    environment:
      POSTGRES_DB: arc_synthesis
      POSTGRES_USER: arc_user
      POSTGRES_PASSWORD: arc_password
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./sql/init.sql:/docker-entrypoint-initdb.d/init.sql
    ports:
      - "5432:5432"
    restart: unless-stopped

volumes:
  redis_data:
  postgres_data:

networks:
  default:
    name: arc-network
'''

DOCKERFILE = '''# Dockerfile for ARC Synthesis Framework

FROM python:3.9-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \\
    build-essential \\
    git \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# 安装SWI-Prolog（Popper依赖）
RUN apt-get update && apt-get install -y \\
    swi-prolog \\
    && rm -rf /var/lib/apt/lists/*

# 复制requirements文件
COPY requirements.txt .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目文件
COPY . .

# 安装项目
RUN pip install -e .

# 创建必要目录
RUN mkdir -p logs data results config

# 设置环境变量
ENV PYTHONPATH=/app
ENV LOG_LEVEL=INFO

# 暴露端口（如果有Web服务）
EXPOSE 8000

# 默认命令
CMD ["python", "main.py", "--demo"]
'''

# =====================================================================
# 5. 监控和日志分析脚本
# =====================================================================
LOG_ANALYZER_PY = '''#!/usr/bin/env python3
"""
日志分析和监控脚本
"""

import re
import json
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import pandas as pd

class LogAnalyzer:
    """日志分析器"""
    
    def __init__(self, log_dir="logs"):
        self.log_dir = Path(log_dir)
        self.patterns = {
            'task_start': r'开始合成任务 (\\w+)',
            'task_success': r'✓ 任务 (\\w+) 合成成功',
            'task_failure': r'✗ 任务 (\\w+) 合成失败: (.+)',
            'synthesis_time': r'合成时间: ([\\d.]+)秒',
            'iterations': r'迭代次数: (\\d+)',
            'confidence': r'置信度: ([\\d.]+)',
            'error': r'ERROR - (.+)',
            'warning': r'WARNING - (.+)'
        }
    
    def analyze_logs(self, log_file=None, days_back=7):
        """分析日志文件"""
        
        if log_file is None:
            log_file = self.log_dir / "synthesis.log"
        
        if not Path(log_file).exists():
            print(f"❌ 日志文件不存在: {log_file}")
            return {}
        
        print(f"📊 分析日志文件: {log_file}")
        
        # 读取日志
        with open(log_file, 'r', encoding='utf-8') as f:
            log_content = f.read()
        
        # 提取信息
        analysis = {
            'tasks': self._extract_task_info(log_content),
            'errors': self._extract_errors(log_content),
            'warnings': self._extract_warnings(log_content),
            'performance': self._extract_performance_metrics(log_content),
            'timeline': self._extract_timeline(log_content, days_back)
        }
        
        return analysis
    
    def _extract_task_info(self, log_content):
        """提取任务信息"""
        tasks = {}
        
        # 提取任务开始
        for match in re.finditer(self.patterns['task_start'], log_content):
            task_id = match.group(1)
            tasks[task_id] = {'status': 'started', 'start_time': match.start()}
        
        # 提取成功任务
        for match in re.finditer(self.patterns['task_success'], log_content):
            task_id = match.group(1)
            if task_id in tasks:
                tasks[task_id]['status'] = 'success'
        
        # 提取失败任务
        for match in re.finditer(self.patterns['task_failure'], log_content):
            task_id = match.group(1)
            error_msg = match.group(2)
            if task_id in tasks:
                tasks[task_id]['status'] = 'failure'
                tasks[task_id]['error'] = error_msg
        
        # 提取性能指标
        for task_id in tasks:
            tasks[task_id]['metrics'] = self._extract_task_metrics(log_content, task_id)
        
        return tasks
    
    def _extract_task_metrics(self, log_content, task_id):
        """提取特定任务的性能指标"""
        metrics = {}
        
        # 在任务相关的日志行中查找指标
        task_pattern = f'任务 {task_id}'
        task_lines = [line for line in log_content.split('\\n') if task_pattern in line]
        
        for line in task_lines:
            # 合成时间
            time_match = re.search(self.patterns['synthesis_time'], line)
            if time_match:
                metrics['synthesis_time'] = float(time_match.group(1))
            
            # 迭代次数
            iter_match = re.search(self.patterns['iterations'], line)
            if iter_match:
                metrics['iterations'] = int(iter_match.group(1))
            
            # 置信度
            conf_match = re.search(self.patterns['confidence'], line)
            if conf_match:
                metrics['confidence'] = float(conf_match.group(1))
        
        return metrics
    
    def _extract_errors(self, log_content):
        """提取错误信息"""
        errors = []
        
        for match in re.finditer(self.patterns['error'], log_content):
            error_msg = match.group(1)
            errors.append(error_msg)
        
        # 统计错误类型
        error_types = Counter()
        for error in errors:
            # 简化错误消息进行分类
            if 'timeout' in error.lower():
                error_types['timeout'] += 1
            elif 'memory' in error.lower():
                error_types['memory'] += 1
            elif 'syntax' in error.lower():
                error_types['syntax'] += 1
            elif 'popper' in error.lower():
                error_types['popper'] += 1
            else:
                error_types['other'] += 1
        
        return {
            'total_errors': len(errors),
            'error_messages': errors[:20],  # 最近20个错误
            'error_types': dict(error_types)
        }
    
    def _extract_warnings(self, log_content):
        """提取警告信息"""
        warnings = []
        
        for match in re.finditer(self.patterns['warning'], log_content):
            warning_msg = match.group(1)
            warnings.append(warning_msg)
        
        return {
            'total_warnings': len(warnings),
            'warning_messages': warnings[:10]  # 最近10个警告
        }
    
    def _extract_performance_metrics(self, log_content):
        """提取性能指标"""
        metrics = {
            'synthesis_times': [],
            'iteration_counts': [],
            'confidence_scores': []
        }
        
        # 提取所有性能数据
        for match in re.finditer(self.patterns['synthesis_time'], log_content):
            metrics['synthesis_times'].append(float(match.group(1)))
        
        for match in re.finditer(self.patterns['iterations'], log_content):
            metrics['iteration_counts'].append(int(match.group(1)))
        
        for match in re.finditer(self.patterns['confidence'], log_content):
            metrics['confidence_scores'].append(float(match.group(1)))
        
        # 计算统计信息
        if metrics['synthesis_times']:
            metrics['avg_synthesis_time'] = sum(metrics['synthesis_times']) / len(metrics['synthesis_times'])
            metrics['max_synthesis_time'] = max(metrics['synthesis_times'])
            metrics['min_synthesis_time'] = min(metrics['synthesis_times'])
        
        if metrics['confidence_scores']:
            metrics['avg_confidence'] = sum(metrics['confidence_scores']) / len(metrics['confidence_scores'])
        
        return metrics
    
    def _extract_timeline(self, log_content, days_back):
        """提取时间线数据"""
        # 这里需要更复杂的时间解析逻辑
        # 简化实现
        timeline = {
            'daily_tasks': defaultdict(int),
            'daily_successes': defaultdict(int),
            'daily_failures': defaultdict(int)
        }
        
        return timeline
    
    def generate_analysis_report(self, analysis, output_file="reports/log_analysis.json"):
        """生成分析报告"""
        
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        
        # 计算汇总统计
        tasks = analysis['tasks']
        total_tasks = len(tasks)
        successful_tasks = len([t for t in tasks.values() if t['status'] == 'success'])
        failed_tasks = len([t for t in tasks.values() if t['status'] == 'failure'])
        
        report = {
            "analysis_timestamp": datetime.now().isoformat(),
            "summary": {
                "total_tasks": total_tasks,
                "successful_tasks": successful_tasks,
                "failed_tasks": failed_tasks,
                "success_rate": successful_tasks / total_tasks if total_tasks > 0 else 0
            },
            "performance": analysis['performance'],
            "errors": analysis['errors'],
            "warnings": analysis['warnings'],
            "task_details": tasks
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"📋 日志分析报告已保存到: {output_file}")
    
    def plot_performance_trends(self, analysis, output_dir="reports/plots"):
        """绘制性能趋势图"""
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        performance = analysis['performance']
        
        # 合成时间分布
        if performance['synthesis_times']:
            plt.figure(figsize=(10, 6))
            plt.hist(performance['synthesis_times'], bins=20, alpha=0.7)
            plt.title('Synthesis Time Distribution')
            plt.xlabel('Synthesis Time (seconds)')
            plt.ylabel('Frequency')
            plt.savefig(f"{output_dir}/synthesis_time_distribution.png", dpi=300)
            plt.close()
        
        # 置信度分布
        if performance['confidence_scores']:
            plt.figure(figsize=(10, 6))
            plt.hist(performance['confidence_scores'], bins=20, alpha=0.7)
            plt.title('Confidence Score Distribution')
            plt.xlabel('Confidence Score')
            plt.ylabel('Frequency')
            plt.savefig(f"{output_dir}/confidence_distribution.png", dpi=300)
            plt.close()
        
        print(f"📈 性能趋势图已保存到: {output_dir}/")

def main():
    """主函数"""
    analyzer = LogAnalyzer()
    
    print("🔍 开始日志分析...")
    
    # 分析日志
    analysis = analyzer.analyze_logs()
    
    if not analysis:
        print("❌ 没有找到可分析的日志")
        return
    
    # 显示基本统计
    tasks = analysis['tasks']
    total_tasks = len(tasks)
    successful_tasks = len([t for t in tasks.values() if t['status'] == 'success'])
    
    print(f"\\n📊 日志分析结果:")
    print(f"总任务数: {total_tasks}")
    print(f"成功任务: {successful_tasks}")
    print(f"成功率: {successful_tasks/total_tasks:.2%}" if total_tasks > 0 else "成功率: 0%")
    
    if analysis['errors']['total_errors'] > 0:
        print(f"错误数量: {analysis['errors']['total_errors']}")
    
    if analysis['warnings']['total_warnings'] > 0:
        print(f"警告数量: {analysis['warnings']['total_warnings']}")
    
    # 生成报告
    analyzer.generate_analysis_report(analysis)
    analyzer.plot_performance_trends(analysis)

if __name__ == "__main__":
    main()
'''

print("🎉 完整的ARC程序合成框架已生成！")
print()
print("📋 补充内容包括:")
print("✅ 完整的端到端使用示例")
print("✅ 批量评估和基准测试工具")
print("✅ Docker部署配置")
print("✅ 日志分析和监控工具")
print("✅ 性能优化和调试指南")
print()
print("🚀 现在您可以:")
print("1. 按照项目结构创建文件")
print("2. 运行演示验证安装")
print("3. 使用批量评估工具测试性能")
print("4. 部署到生产环境")
print("5. 监控和分析系统性能")
print()
print("📚 所有文件都包含详细的注释和扩展指南")
print("🤝 祝您使用愉快！如有问题随时联系。")
