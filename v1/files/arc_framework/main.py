"""
Main application for the ARC solving framework.
"""
import numpy as np
import json
import argparse
from arc_framework.architecture import ARCFramework

def load_arc_task(task_path):
    """Load an ARC task from a JSON file."""
    with open(task_path, 'r') as f:
        task_data = json.load(f)
    
    train_pairs = []
    for example in task_data.get('train', []):
        input_grid = np.array(example['input'])
        output_grid = np.array(example['output'])
        train_pairs.append((input_grid, output_grid))
    
    test_pairs = []
    for example in task_data.get('test', []):
        input_grid = np.array(example['input'])
        if 'output' in example:
            output_grid = np.array(example['output'])
            test_pairs.append((input_grid, output_grid))
        else:
            test_pairs.append((input_grid, None))
    
    return train_pairs, test_pairs

def main():
    parser = argparse.ArgumentParser(description='ARC Solving Framework')
    parser.add_argument('--task', type=str, help='Path to ARC task JSON file')
    parser.add_argument('--prior', type=str, help='Path to prior knowledge file')
    parser.add_argument('--output', type=str, help='Path to save results')
    args = parser.parse_args()
    
    if not args.task:
        print("Please specify a task file with --task")
        return
    
    # Initialize the framework
    framework = ARCFramework(prior_knowledge_path=args.prior)
    
    # Load the task
    train_pairs, test_pairs = load_arc_task(args.task)
    
    print(f"Loaded task with {len(train_pairs)} training examples and {len(test_pairs)} test examples")
    
    # Train on the examples
    for i, (input_grid, output_grid) in enumerate(train_pairs):
        print(f"Learning from training example {i+1}...")
        framework.learn_from_samples([(input_grid, output_grid)])
    
    # Solve the test examples
    results = []
    
    for i, (input_grid, expected_output) in enumerate(test_pairs):
        print(f"Solving test example {i+1}...")
        predicted_output = framework.solve_problem(input_grid, train_pairs)
        
        if expected_output is not None:
            # Calculate accuracy
            correct = np.array_equal(predicted_output, expected_output)
            print(f"Test {i+1}: {'Correct' if correct else 'Incorrect'}")
        else:
            print(f"Test {i+1}: No ground truth available")
        
        result = {
            "test_id": i,
            "input": input_grid.tolist(),
            "predicted_output": predicted_output.tolist(),
            "expected_output": expected_output.tolist() if expected_output is not None else None
        }
        results.append(result)
    
    # Save results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output}")

if __name__ == "__main__":
    main()