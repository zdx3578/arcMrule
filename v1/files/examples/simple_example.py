"""
Simple example demonstrating how to use the ARC framework
"""
import numpy as np
from arc_framework.architecture import ARCFramework

def main():
    # Initialize the framework
    framework = ARCFramework()
    
    # Define a simple example: move all blue pixels (2) to the right by 1
    input_grid = np.array([
        [0, 0, 0, 0, 0],
        [0, 2, 0, 0, 0],
        [0, 0, 2, 0, 0],
        [0, 2, 0, 0, 0],
        [0, 0, 0, 0, 0]
    ])
    
    output_grid = np.array([
        [0, 0, 0, 0, 0],
        [0, 0, 2, 0, 0],
        [0, 0, 0, 2, 0],
        [0, 0, 2, 0, 0],
        [0, 0, 0, 0, 0]
    ])
    
    # Learn from this example
    framework.learn_from_samples([(input_grid, output_grid)])
    
    # Define a new test case with the same pattern
    test_input = np.array([
        [0, 0, 0, 0, 0],
        [0, 2, 0, 2, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 2, 0, 0],
        [0, 0, 0, 0, 0]
    ])
    
    # Solve the test case
    predicted_output = framework.solve_problem(test_input, [(input_grid, output_grid)])
    
    # Display results
    print("Input Grid:")
    print(test_input)
    
    print("\nPredicted Output Grid:")
    print(predicted_output)
    
    # Expected output: blue pixels moved right by 1
    expected_output = np.array([
        [0, 0, 0, 0, 0],
        [0, 0, 2, 0, 2],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 2, 0],
        [0, 0, 0, 0, 0]
    ])
    
    print("\nExpected Output Grid:")
    print(expected_output)
    
    # Check if prediction matches expected output
    is_correct = np.array_equal(predicted_output, expected_output)
    print(f"\nPrediction is {'correct' if is_correct else 'incorrect'}")

if __name__ == "__main__":
    main()