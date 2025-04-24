import json
from typing import Dict, Any, List
from arc_diff_analyzer import ARCDiffAnalyzer

class ARCSolver:
    """ARC问题解决器，使用差异网格分析"""
    
    def __init__(self):
        """初始化ARC解决器"""
        self.diff_analyzer = ARCDiffAnalyzer()
    
    def load_task(self, task_json: str) -> Dict[str, Any]:
        """
        加载ARC任务
        
        Args:
            task_json: 任务JSON文件路径
            
        Returns:
            加载的任务数据
        """
        try:
            with open(task_json, 'r') as f:
                task_data = json.load(f)
            
            # 验证任务格式
            if 'train' not in task_data or 'test' not in task_data:
                raise ValueError("任务JSON格式不正确，缺少train或test字段")
            
            return task_data
        except Exception as e:
            print(f"加载任务失败: {e}")
            return None
    
    def process_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理ARC任务数据
        
        Args:
            task_data: 任务数据
            
        Returns:
            处理结果
        """
        # 处理训练数据
        for i, example in enumerate(task_data['train']):
            input_grid = example['input']
            output_grid = example['output']
            
            # 添加到差异分析器
            self.diff_analyzer.add_train_pair(i, input_grid, output_grid)
        
        # 分析共有模式
        common_patterns = self.diff_analyzer.analyze_common_patterns()
        
        # 处理测试数据
        test_predictions = []
        for example in task_data['test']:
            input_grid = example['input']
            
            # 应用共有模式进行预测
            predicted_output = self.diff_analyzer.apply_common_patterns(input_grid)
            
            test_predictions.append({
                'input': example['input'],
                'predicted_output': predicted_output
            })
        
        return {
            'common_patterns': common_patterns,
            'mapping_rules': self.diff_analyzer.mapping_rules,
            'test_predictions': test_predictions
        }
    
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