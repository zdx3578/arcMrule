"""
Architecture overview of the ARC solving framework.
"""

class ARCFramework:
    """
    Main framework class coordinating all components for ARC problem solving.
    """
    def __init__(self, prior_knowledge_path=None):
        self.grid_processor = GridProcessor()
        self.object_extractor = ObjectExtractor()
        self.pattern_recognizer = PatternRecognizer()
        self.rule_engine = RuleEngine()
        self.weight_calculator = WeightCalculator()
        self.persistence_manager = PersistenceManager()
        
        # Load prior knowledge if provided
        if prior_knowledge_path:
            self.load_prior_knowledge(prior_knowledge_path)
    
    def load_prior_knowledge(self, path):
        """Load prior knowledge from file."""
        self.persistence_manager.load_knowledge(path, self.rule_engine)
    
    def solve_problem(self, input_grid, sample_input_output_pairs=None):
        """
        Solve a single ARC problem.
        
        Args:
            input_grid: The input grid to transform
            sample_input_output_pairs: List of known input-output pairs for learning
            
        Returns:
            Transformed output grid
        """
        # Extract objects from input grid
        input_objects = self.object_extractor.extract_objects(input_grid)
        
        # If we have sample pairs, learn from them
        if sample_input_output_pairs:
            self.learn_from_samples(sample_input_output_pairs)
        
        # Apply rules to input objects
        transformation_rules = self.rule_engine.get_applicable_rules(input_grid, input_objects)
        output_grid = self.apply_transformations(input_grid, input_objects, transformation_rules)
        
        return output_grid
    
    def learn_from_samples(self, sample_pairs):
        """Learn patterns and rules from sample input-output pairs."""
        for input_grid, output_grid in sample_pairs:
            # Calculate difference grid
            diff_grid = self.grid_processor.calculate_diff(input_grid, output_grid)
            
            # Extract objects from all grids
            input_objects = self.object_extractor.extract_objects(input_grid)
            output_objects = self.object_extractor.extract_objects(output_grid)
            diff_objects = self.object_extractor.extract_objects(diff_grid)
            
            # Calculate object weights
            weighted_input_objects = self.weight_calculator.calculate_weights(input_objects)
            weighted_output_objects = self.weight_calculator.calculate_weights(output_objects)
            
            # Identify patterns and rules
            patterns = self.pattern_recognizer.identify_patterns(
                input_grid, output_grid, diff_grid,
                weighted_input_objects, weighted_output_objects, diff_objects
            )
            
            # Update rule engine with new patterns
            self.rule_engine.update_rules(patterns)
    
    def apply_transformations(self, input_grid, input_objects, transformation_rules):
        """Apply transformation rules to generate output grid."""
        # Sort rules by confidence/priority
        sorted_rules = sorted(transformation_rules, key=lambda r: r.confidence, reverse=True)
        
        # Start with a copy of the input grid
        output_grid = input_grid.copy()
        
        # Apply rules in order
        for rule in sorted_rules:
            output_grid = rule.apply(output_grid, input_objects)
            
        return output_grid
    
    def add_pattern(self, pattern_data):
        """Add a new pattern to the pattern library."""
        self.pattern_recognizer.add_pattern(pattern_data)
        self.persistence_manager.save_patterns(self.pattern_recognizer.patterns)
    
    def add_action(self, action_data):
        """Add a new action type to the rule engine."""
        self.rule_engine.add_action(action_data)
        self.persistence_manager.save_actions(self.rule_engine.actions)