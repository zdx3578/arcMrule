import argparse
from arc_solver import ARCSolver

def main():
    """主程序入口"""
    parser = argparse.ArgumentParser(description='ARC任务解决器')
    parser.add_argument('task_json', help='ARC任务JSON文件路径')
    parser.add_argument('--output', '-o', default='results.json', help='结果输出文件路径')
    parser.add_argument('--debug', '-d', action='store_true', help='启用调试模式')
    parser.add_argument('--debug-dir', default='debug_output', help='调试输出目录')

    args = parser.parse_args()

    # 初始化解决器
    solver = ARCSolver(debug=args.debug, debug_dir=args.debug_dir)

    # 加载任务
    task_data = solver.load_task(args.task_json)
    if not task_data:
        return

    # 处理任务
    results = solver.process_task(task_data)

    # 保存结果
    solver.save_results(results, args.output)

if __name__ == '__main__':
    main()


#!  python main.py path/to/task.json --output results.json --debug