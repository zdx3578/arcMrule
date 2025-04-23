"""
Persistence manager for saving and loading data in the ARC framework.
"""
import json
import os
import pickle
from datetime import datetime

class PersistenceManager:
    """
    Manages persistence of rules, patterns, and other data.
    """
    def __init__(self, data_dir="arc_data"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        # Create subdirectories
        self.rules_dir = os.path.join(data_dir, "rules")
        self.patterns_dir = os.path.join(data_dir, "patterns")
        self.actions_dir = os.path.join(data_dir, "actions")
        self.knowledge_dir = os.path.join(data_dir, "knowledge")
        
        for directory in [self.rules_dir, self.patterns_dir, 
                         self.actions_dir, self.knowledge_dir]:
            os.makedirs(directory, exist_ok=True)
    
    def save_rules(self, rules, version=None):
        """Save rules to a file."""
        if not version:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        filename = os.path.join(self.rules_dir, f"rules_{version}.json")
        
        # Convert rules to serializable format
        serializable_rules = []
        for rule in rules:
            # We can't directly serialize functions, so we store their names
            # and reconstruct them when loading
            serializable_rule = {
                'id': rule.id,
                'name': rule.name,
                'description': rule.description,
                'confidence': rule.confidence,
                'action_type': rule.metadata.get('action_type', 'unknown'),
                'metadata': rule.metadata
            }
            serializable_rules.append(serializable_rule)
        
        with open(filename, 'w') as f:
            json.dump(serializable_rules, f, indent=2)
        
        return filename
    
    def load_rules(self, filename, rule_engine):
        """Load rules from a file into the rule engine."""
        if not os.path.exists(filename):
            return []
        
        with open(filename, 'r') as f:
            serialized_rules = json.load(f)
        
        loaded_rules = []
        for rule_data in serialized_rules:
            # Add the rule to the engine
            try:
                rule = rule_engine.add_rule(rule_data)
                loaded_rules.append(rule)
            except ValueError:
                # Skip rules with unknown action types
                pass
        
        return loaded_rules
    
    def save_patterns(self, patterns, version=None):
        """Save patterns to a file."""
        if not version:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        filename = os.path.join(self.patterns_dir, f"patterns_{version}.json")
        
        # Convert patterns to serializable format
        serializable_patterns = []
        for pattern in patterns:
            serializable_pattern = {
                'id': pattern.id,
                'name': pattern.name,
                'description': pattern.description,
                'confidence': pattern.confidence,
                'metadata': pattern.metadata
            }
            serializable_patterns.append(serializable_pattern)
        
        with open(filename, 'w') as f:
            json.dump(serializable_patterns, f, indent=2)
        
        return filename
    
    def load_patterns(self, filename, pattern_recognizer):
        """Load patterns from a file into the pattern recognizer."""
        if not os.path.exists(filename):
            return []
        
        with open(filename, 'r') as f:
            serialized_patterns = json.load(f)
        
        loaded_patterns = []
        for pattern_data in serialized_patterns:
            pattern = pattern_recognizer.add_pattern(pattern_data)
            loaded_patterns.append(pattern)
        
        return loaded_patterns
    
    def save_actions(self, actions, version=None):
        """Save action types to a file."""
        if not version:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        filename = os.path.join(self.actions_dir, f"action_types_{version}.txt")
        
        # We can only save the action types, not the functions themselves
        action_types = list(actions.keys())
        
        with open(filename, 'w') as f:
            for action_type in action_types:
                f.write(f"{action_type}\n")
        
        return filename
    
    def save_knowledge(self, knowledge_data, name):
        """Save general knowledge to a file."""
        filename = os.path.join(self.knowledge_dir, f"{name}.pkl")
        
        with open(filename, 'wb') as f:
            pickle.dump(knowledge_data, f)
        
        return filename
    
    def load_knowledge(self, filename, target_object=None):
        """Load knowledge from a file."""
        if not os.path.exists(filename):
            return None
        
        with open(filename, 'rb') as f:
            knowledge_data = pickle.load(f)
        
        if target_object and hasattr(target_object, 'load_knowledge'):
            target_object.load_knowledge(knowledge_data)
        
        return knowledge_data