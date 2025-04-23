# ARC Solving Framework

A comprehensive framework for solving tasks from the Abstraction and Reasoning Corpus (ARC) using a flexible, rule-based approach with object-oriented analysis.

## Features

- **Object Extraction**: Identifies and extracts objects from grids with hierarchical decomposition
- **Pattern Recognition**: Analyzes patterns and transformations between input and output examples
- **Rule Engine**: Manages and applies transformation rules learned from examples
- **Weight Calculation**: Prioritizes objects for analysis based on various factors
- **Persistence**: Saves and loads learned patterns, rules, and knowledge
- **Human Guidance**: Accepts prior knowledge and can be guided with additional rules or actions

## Installation

```bash
pip install -r requirements.txt
```

## Basic Usage

```python
from arc_framework.architecture import ARCFramework
import numpy as np

# Initialize the framework
framework = ARCFramework()

# Define example input and output
input_grid = np.array([
    [0, 0, 0],
    [0, 1, 0],
    [0, 0, 0]
])

output_grid = np.array([
    [0, 1, 0],
    [1, 1, 1],
    [0, 1, 0]
])

# Learn from this example
framework.learn_from_samples([(input_grid, output_grid)])

# Solve a new problem
test_input = np.array([
    [0, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0]
])

predicted_output = framework.solve_problem(test_input)
```

## Working with ARC Task Files

```bash
python -m arc_framework.main --task path/to/task.json --output results.json
```

## Adding New Patterns and Actions

```python
# Add a new pattern
framework.add_pattern({
    'name': 'Symmetrical Pattern',
    'description': 'Objects form a symmetrical pattern',
    'confidence': 0.8,
    'metadata': {'type': 'symmetry', 'axis': 'vertical'}
})

# Add a new action
framework.rule_engine.add_action({
    'type': 'create_symmetry',
    'function': my_symmetry_function
})
```

## Key Components

1. **GridProcessor**: Handles grid operations and differences
2. **ObjectExtractor**: Identifies and extracts objects from grids
3. **PatternRecognizer**: Analyzes patterns and transformations
4. **RuleEngine**: Manages and applies transformation rules
5. **WeightCalculator**: Assigns weights to objects based on various factors
6. **PersistenceManager**: Saves and loads data, patterns, and rules

## Extending the Framework

The framework is designed to be extensible:

- Add new actions to the rule engine
- Define new object extraction methods
- Create custom weight calculation algorithms
- Add pattern recognition strategies

## Iterative Improvement

During runtime, the framework can:

1. Identify when current rules are insufficient
2. Search for better rules based on feedback
3. Update rule weights and confidences
4. Persist improved knowledge for future use

## Requirements

- Python 3.7+
- NumPy
- scikit-image