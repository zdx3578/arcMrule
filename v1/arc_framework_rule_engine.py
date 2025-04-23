"""
Rule engine for managing and applying transformation rules in ARC.
"""
from dataclasses import dataclass
from typing import List, Dict, Any, Callable, Optional
import numpy as np

@dataclass
class TransformationRule:
    """A rule that can be applied to transform a grid."""
    id: str
    name: str
    description: str
    confidence: float
    action: Callable
    preconditions: List[Callable]
    metadata: Dict[str, Any]
    
    def applies_to(self, grid, objects):
        """Check if this rule applies to the given grid and objects."""
        for condition in self.preconditions:
            if not condition(grid, objects):
                return False
        return True
    
    def apply(self, grid, objects):
        """Apply the rule to transform the grid."""
        return self.action(grid, objects, self.metadata)


class RuleEngine:
    """
    Manages and applies transformation rules for ARC problems.
    """
    def __init__(self):
        self.rules = []
        self.actions = {}
        self.default_rules = self._create_default_rules()
    
    def add_rule(self, rule_data):
        """Add a new transformation rule."""
        action_func = self.actions.get(rule_data.get('action_type'))
        
        if not action_func:
            raise ValueError(f"Unknown action type: {rule_data.get('action_type')}")
        
        preconditions = []
        for precond in rule_data.get('preconditions', []):
            if callable(precond):
                preconditions.append(precond)
            else:
                # Convert string/dict representation to function
                preconditions.append(self._create_condition_func(precond))
        
        rule = TransformationRule(
            id=rule_data.get('id', f'rule_{len(self.rules)}'),
            name=rule_data.get('name', 'Unnamed Rule'),
            description=rule_data.get('description', ''),
            confidence=rule_data.get('confidence', 0.5),
            action=action_func,
            preconditions=preconditions,
            metadata=rule_data.get('metadata', {})
        )
        
        self.rules.append(rule)
        return rule
    
    def add_action(self, action_data):
        """Add a new action type."""
        action_type = action_data.get('type')
        action_func = action_data.get('function')
        
        if not action_type or not callable(action_func):
            raise ValueError("Action must have a type and a callable function")
        
        self.actions[action_type] = action_func
        return action_type
    
    def get_applicable_rules(self, grid, objects):
        """Get all rules that apply to the given grid and objects."""
        applicable_rules = []
        
        for rule in self.rules + self.default_rules:
            if rule.applies_to(grid, objects):
                applicable_rules.append(rule)
        
        return applicable_rules
    
    def update_rules(self, patterns):
        """Update rules based on identified patterns."""
        for pattern in patterns:
            # Check if we already have a similar rule
            existing_rule = self._find_similar_rule(pattern)
            
            if existing_rule:
                # Update confidence of existing rule
                new_confidence = (existing_rule.confidence + pattern.confidence) / 2
                existing_rule.confidence = new_confidence
            else:
                # Create a new rule from the pattern
                self._create_rule_from_pattern(pattern)
    
    def _find_similar_rule(self, pattern):
        """Find a rule similar to the given pattern."""
        pattern_type = pattern.metadata.get('type')
        
        for rule in self.rules:
            rule_type = rule.metadata.get('type')
            
            if pattern_type == rule_type:
                # For movement patterns, check if directions match
                if pattern_type == 'movement':
                    if (pattern.metadata.get('dx') == rule.metadata.get('dx') and
                        pattern.metadata.get('dy') == rule.metadata.get('dy')):
                        return rule
                
                # For color changes, check if colors match
                elif pattern_type == 'color_change' or pattern_type == 'color_mapping':
                    if (pattern.metadata.get('input_color') == rule.metadata.get('input_color') and
                        pattern.metadata.get('output_color') == rule.metadata.get('output_color')):
                        return rule
        
        return None
    
    def _create_rule_from_pattern(self, pattern):
        """Create a new rule from a pattern."""
        pattern_type = pattern.metadata.get('type')
        
        if pattern_type == 'movement':
            # Create movement rule
            action_func = self.actions.get('move_objects')
            if not action_func:
                return None
            
            rule_data = {
                'id': f"rule_{pattern.id}",
                'name': f"Rule: {pattern.name}",
                'description': pattern.description,
                'confidence': pattern.confidence,
                'action_type': 'move_objects',
                'preconditions': [],
                'metadata': {
                    'dx': pattern.metadata.get('dx'),
                    'dy': pattern.metadata.get('dy'),
                    'type': 'movement'
                }
            }
            
            return self.add_rule(rule_data)
        
        elif pattern_type == 'color_change' or pattern_type == 'color_mapping':
            # Create color change rule
            action_func = self.actions.get('change_color')
            if not action_func:
                return None
            
            rule_data = {
                'id': f"rule_{pattern.id}",
                'name': f"Rule: {pattern.name}",
                'description': pattern.description,
                'confidence': pattern.confidence,
                'action_type': 'change_color',
                'preconditions': [],
                'metadata': {
                    'input_color': pattern.metadata.get('input_color'),
                    'output_color': pattern.metadata.get('output_color'),
                    'type': pattern_type
                }
            }
            
            return self.add_rule(rule_data)
        
        return None
    
    def _create_condition_func(self, condition_spec):
        """Create a condition function from a specification."""
        if isinstance(condition_spec, str):
            # Simple condition types
            if condition_spec == 'has_objects':
                return lambda grid, objects: len(objects) > 0
            # Add more condition types as needed
        
        # Default always-true condition
        return lambda grid, objects: True
    
    def _create_default_rules(self):
        """Create some default fallback rules."""
        default_rules = []
        
        # Identity rule (do nothing)
        identity_rule = TransformationRule(
            id="default_identity",
            name="Identity Transformation",
            description="Keep the grid unchanged",
            confidence=0.1,  # Very low confidence
            action=lambda grid, objects, metadata: grid.copy(),
            preconditions=[lambda grid, objects: True],  # Always applies
            metadata={"type": "identity"}
        )
        
        default_rules.append(identity_rule)
        
        return default_rules