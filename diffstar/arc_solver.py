import json
import os
from typing import Dict, Any, List, Optional
from arc_diff_analyzer import ARCDiffAnalyzer

import sys
import logging
import traceback
import cgitb
cgitb.enable(format='text')


class ARCSolver:
    """ARC问题解决器，使用差异网格分析"""

    def __init__(self, data_dir=None, debug=False):
        """
        初始化ARC解决器

        Args:
            data_dir: ARC数据集目录，如果提供则自动加载整个数据集
            debug: 是否开启调试模式
        """
        self.diff_analyzer = ARCDiffAnalyzer()
        self.debug = debug

        # 初始化数据字典
        self.train_tasks = {}
        self.train_sols = {}
        self.eval_tasks = {}
        self.eval_sols = {}
        self.test_tasks = {}

        # 如果提供了数据目录，则加载整个数据集
        if data_dir:
            self.data_dir = data_dir
            self.load_all_data()

    def find_data_dir(self):
        """查找可用的数据目录路径"""
        # 定义多个可能的路径
        data_paths = [
            '/kaggle/input/arc-prize-2025',
            '/Users/zhangdexiang/github/ARC-AGI-2/arc-prize-2025',
            '/home/zdx/github/VSAHDC/arc-prize-2025',
            './data',
            './arc-data'
        ]

        # 遍历路径列表，找到第一个存在的路径
        for path in data_paths:
            if os.path.exists(path):
                self.data_dir = path
                if self.debug:
                    print(f"数据路径设置为: {self.data_dir}")
                print(f"数据路径设置为: {self.data_dir}")
                return path

        # 如果所有路径都不存在，抛出异常
        raise FileNotFoundError("未找到任何指定的数据路径！")

    def load_all_data(self):
        """加载所有ARC数据集文件"""
        if not hasattr(self, 'data_dir') or not self.data_dir:
            self.find_data_dir()

        # 尝试加载所有数据文件
        try:
            # 训练任务和解决方案
            train_tasks_path = os.path.join(self.data_dir, 'arc-agi_training_challenges.json')
            train_sols_path = os.path.join(self.data_dir, 'arc-agi_training_solutions.json')
            if os.path.exists(train_tasks_path):
                self.train_tasks = self.load_json(train_tasks_path)
            if os.path.exists(train_sols_path):
                self.train_sols = self.load_json(train_sols_path)

            # 评估任务和解决方案
            eval_tasks_path = os.path.join(self.data_dir, 'arc-agi_evaluation_challenges.json')
            eval_sols_path = os.path.join(self.data_dir, 'arc-agi_evaluation_solutions.json')
            if os.path.exists(eval_tasks_path):
                self.eval_tasks = self.load_json(eval_tasks_path)
            if os.path.exists(eval_sols_path):
                self.eval_sols = self.load_json(eval_sols_path)

            # 测试任务
            test_tasks_path = os.path.join(self.data_dir, 'arc-agi_test_challenges.json')
            if os.path.exists(test_tasks_path):
                self.test_tasks = self.load_json(test_tasks_path)

            if self.debug:
                print(f"加载了 {len(self.train_tasks)} 个训练任务")
                print(f"加载了 {len(self.eval_tasks)} 个评估任务")
                print(f"加载了 {len(self.test_tasks)} 个测试任务")

            return True

        except Exception as e:
            if self.debug:
                print(f"加载数据集失败: {e}")
            return False

    def load_json(self, file_path: str) -> Dict:
        """加载JSON文件"""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            return data
        except Exception as e:
            if self.debug:
                print(f"加载JSON文件失败: {file_path}")
                print(f"错误: {str(e)}")
            return {}

    def load_task(self, task_id_or_json: str) -> Optional[Dict[str, Any]]:
        """
        加载ARC任务，可以是任务ID或JSON文件路径

        Args:
            task_id_or_json: 任务ID或JSON文件路径

        Returns:
            加载的任务数据
        """
        # 判断是任务ID还是文件路径
        if os.path.exists(task_id_or_json) and task_id_or_json.endswith('.json'):
            # 加载单个JSON文件
            try:
                with open(task_id_or_json, 'r') as f:
                    task_data = json.load(f)

                # 验证任务格式
                if 'train' not in task_data or 'test' not in task_data:
                    raise ValueError("任务JSON格式不正确，缺少train或test字段")

                return {'task': task_data, 'id': os.path.basename(task_id_or_json).replace('.json', '')}
            except Exception as e:
                print(f"加载任务文件失败: {e}")
                return None
        else:
            # 作为任务ID处理
            task_id = task_id_or_json

            # 如果未加载数据集，先加载
            if not hasattr(self, 'train_tasks') or not self.train_tasks:
                self.load_all_data()

            # 在各个数据集中查找任务
            if task_id in self.train_tasks:
                return {
                    'task': self.train_tasks[task_id],
                    'solution': self.train_sols.get(task_id),
                    'id': task_id,
                    'type': 'train'
                }
            elif task_id in self.eval_tasks:
                return {
                    'task': self.eval_tasks[task_id],
                    'solution': self.eval_sols.get(task_id),
                    'id': task_id,
                    'type': 'eval'
                }
            elif task_id in self.test_tasks:
                return {
                    'task': self.test_tasks[task_id],
                    'solution': None,  # 测试任务没有解决方案
                    'id': task_id,
                    'type': 'test'
                }
            else:
                print(f"未找到任务ID: {task_id}")
                return None

    def process_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理ARC任务数据

        Args:
            task_data: 任务数据

        Returns:
            处理结果
        """
        # 获取实际任务数据
        task = task_data['task']
        task_id = task_data.get('id', 'unknown')

        if self.debug:
            print(f"\n\n\n\n\n\n处理 task 任务 {task_id}")

        # 重置差异分析器
        self.diff_analyzer = ARCDiffAnalyzer()

        # 处理训练数据
        for i, example in enumerate(task['train']):
            print(f"\n\n\n\n处理 task pair 数据对 {i} \n\n")
            input_grid = example['input']
            output_grid = example['output']

            # 添加到差异分析器
            self.diff_analyzer.add_train_pair(i, input_grid, output_grid)

        # 分析共有模式
        common_patterns = self.diff_analyzer.analyze_common_patterns()

        # 处理测试数据
        test_predictions = []
        for example in task['test']:
            input_grid = example['input']

            # 应用共有模式进行预测
            predicted_output = self.diff_analyzer.apply_common_patterns(input_grid)

            test_predictions.append({
                'input': example['input'],
                'predicted_output': predicted_output
            })

        return {
            'task_id': task_id,
            'common_patterns': common_patterns,
            'mapping_rules': self.diff_analyzer.mapping_rules,
            'test_predictions': test_predictions
        }

    def process_all_tasks(self, task_type='train', limit=None):
        """
        处理指定类型的所有任务

        Args:
            task_type: 任务类型，可以是'train', 'eval'或'test'
            limit: 处理的最大任务数，None表示处理所有任务

        Returns:
            所有任务的处理结果
        """
        # 确保数据已加载
        if not hasattr(self, 'train_tasks') or not self.train_tasks:
            self.load_all_data()

        # 选择任务集
        if task_type == 'train':
            task_dict = self.train_tasks
            solutions = self.train_sols
        elif task_type == 'eval':
            task_dict = self.eval_tasks
            solutions = self.eval_sols
        elif task_type == 'test':
            task_dict = self.test_tasks
            solutions = {}
        else:
            raise ValueError(f"未知的任务类型: {task_type}")

        # 处理所有任务
        results = {}
        count = 0

        for task_id, task in task_dict.items():
            if limit is not None and count >= limit:
                break

            task_data = {
                'task': task,
                'solution': solutions.get(task_id),
                'id': task_id,
                'type': task_type
            }

            try:
                result = self.process_task(task_data)
                results[task_id] = result
                count += 1

                if self.debug:
                    print(f"完成处理任务 {task_id} ({count}/{len(task_dict) if limit is None else min(limit, len(task_dict))})")

            except Exception as e:
                if self.debug:
                    print(f"处理任务 {task_id} 失败: {e}")
                tb_str = traceback.format_exc()
                print(f"发生错误: {e}")
                print(f"错误上下文:\n{tb_str}")
                exc_type, exc_value, exc_traceback = sys.exc_info()
                # 获取出错的文件名、行号、函数名和代码行
                filename, line_number, func_name, code_line = traceback.extract_tb(exc_traceback)[-1]
                print(f"错误: {e}")
                print(f"发生在文件 {filename}, 第 {line_number} 行, 函数 {func_name}")
                print(f"错误代码: {code_line}")

        return results

    def save_results(self, results: Dict[str, Any], output_file: str) -> bool:
        """
        保存处理结果

        Args:
            results: 处理结果
            output_file: 输出文件路径

        Returns:
            是否成功保存
        """
        try:
            # 处理结果以便JSON序列化
            serializable_results = self._make_serializable(results)

            with open(output_file, 'w') as f:
                json.dump(serializable_results, f, indent=2)
            print(f"结果已保存到 {output_file}")
            return True
        except Exception as e:
            print(f"保存结果失败: {e}")
            return False

    def _make_serializable(self, obj):
        """使对象可JSON序列化"""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return list(self._make_serializable(item) for item in obj)
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        elif isinstance(obj, frozenset):
            return list(self._make_serializable(item) for item in obj)
        else:
            # 其他类型转为字符串
            return str(obj)