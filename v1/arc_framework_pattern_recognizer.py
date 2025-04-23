"""
Pattern recognition module for identifying transformation patterns in ARC.
"""
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

@dataclass
class Pattern:
    """A recognized pattern in ARC data."""
    id: str
    name: str
    description: str
    confidence: float
    metadata: Dict[str, Any]


class PatternRecognizer:
    """
    Recognizes patterns in grid transformations.
    """
    def __init__(self):
        self.patterns = []
        self.pattern_library = {}
    
    def identify_patterns(self, input_grid, output_grid, diff_grid, 
                         input_objects, output_objects, diff_objects):
        """
        Identify transformation patterns between input and output grids.
        
        Args:
            input_grid: Original input grid
            output_grid: Target output grid
            diff_grid: Difference between input and output
            input_objects: Weighted objects from input
            output_objects: Weighted objects from output
            diff_objects: Objects from diff grid
            
        Returns:
            List of identified patterns with confidence scores
        """
        identified_patterns = []
        
        # Analyze object transformations
        object_patterns = self._analyze_object_transformations(
            input_grid, output_grid, input_objects, output_objects)
        identified_patterns.extend(object_patterns)
        
        # Analyze color changes
        color_patterns = self._analyze_color_patterns(
            input_grid, output_grid, diff_grid)
        identified_patterns.extend(color_patterns)
        
        # Analyze structural changes
        structural_patterns = self._analyze_structural_patterns(
            input_grid, output_grid, diff_grid)
        identified_patterns.extend(structural_patterns)
        
        # Analyze spatial relationships
        spatial_patterns = self._analyze_spatial_relationships(
            input_objects, output_objects)
        identified_patterns.extend(spatial_patterns)
        
        return identified_patterns
    
    def add_pattern(self, pattern_data):
        """Add a new pattern to the library."""
        pattern = Pattern(
            id=pattern_data.get('id', f'pattern_{len(self.patterns)}'),
            name=pattern_data.get('name', 'Unnamed Pattern'),
            description=pattern_data.get('description', ''),
            confidence=pattern_data.get('confidence', 0.5),
            metadata=pattern_data.get('metadata', {})
        )
        
        self.patterns.append(pattern)
        self.pattern_library[pattern.id] = pattern
        
        return pattern
    
    def _analyze_object_transformations(self, input_grid, output_grid, 
                                      input_objects, output_objects):
        """Analyze how objects transform between input and output."""
        patterns = []
        
        # Sort objects by weight/priority
        sorted_input = sorted(input_objects, key=lambda x: x[1], reverse=True)
        sorted_output = sorted(output_objects, key=lambda x: x[1], reverse=True)
        
        # Try to match objects between input and output
        for input_obj, input_weight in sorted_input:
            potential_matches = []
            
            for output_obj, output_weight in sorted_output:
                # Calculate similarity score
                similarity = self._calculate_object_similarity(input_obj, output_obj)
                
                if similarity > 0.2:  # Arbitrary threshold
                    potential_matches.append((output_obj, output_weight, similarity))
            
            # Sort potential matches by similarity
            potential_matches.sort(key=lambda x: x[2], reverse=True)
            
            if potential_matches:
                best_match, _, similarity = potential_matches[0]
                
                # Analyze transformation between input_obj and best_match
                transform_pattern = self._analyze_transformation(input_obj, best_match)
                
                if transform_pattern:
                    transform_pattern.confidence = similarity
                    patterns.append(transform_pattern)
        
        return patterns
    
    def _calculate_object_similarity(self, obj1, obj2):
        """Calculate similarity between two objects."""
        # This would be a complex function comparing various attributes
        # Simplified version just looks at size and color
        
        # Check if same color
        color_match = 1.0 if obj1.color == obj2.color else 0.0
        
        # Check size similarity
        size_ratio = min(obj1.area, obj2.area) / max(obj1.area, obj2.area)
        
        # Combined similarity score
        similarity = 0.5 * color_match + 0.5 * size_ratio
        
        return similarity
    
    def _analyze_transformation(self, input_obj, output_obj):
        """Analyze the transformation between two objects."""
        # Check for basic transformations
        
        # Movement transformation
        input_center = self._calculate_center(input_obj)
        output_center = self._calculate_center(output_obj)
        
        dx = output_center[1] - input_center[1]
        dy = output_center[0] - input_center[0]
        
        if dx != 0 or dy != 0:
            return Pattern(
                id=f"movement_{dx}_{dy}",
                name=f"Movement ({dx},{dy})",
                description=f"Object moved by ({dx},{dy}) units",
                confidence=0.8,
                metadata={
                    "dx": dx,
                    "dy": dy,
                    "type": "movement"
                }
            )
        
        # Color transformation
        if input_obj.color != output_obj.color:
            return Pattern(
                id=f"color_change_{input_obj.color}_to_{output_obj.color}",
                name=f"Color Change {input_obj.color} -> {output_obj.color}",
                description=f"Object changed color from {input_obj.color} to {output_obj.color}",
                confidence=0.9,
                metadata={
                    "input_color": input_obj.color,
                    "output_color": output_obj.color,
                    "type": "color_change"
                }
            )
        
        # Size transformation
        if input_obj.area != output_obj.area:
            scale_factor = output_obj.area / input_obj.area
            return Pattern(
                id=f"resize_{scale_factor:.2f}",
                name=f"Resize by {scale_factor:.2f}",
                description=f"Object resized by factor of {scale_factor:.2f}",
                confidence=0.7,
                metadata={
                    "scale_factor": scale_factor,
                    "type": "resize"
                }
            )
        
        return None
    
    def _calculate_center(self, obj):
        """Calculate the center of an object."""
        pixels = obj.pixels
        if not pixels:
            return (0, 0)
            
        r_coords, c_coords = zip(*pixels)
        center_r = sum(r_coords) / len(r_coords)
        center_c = sum(c_coords) / len(c_coords)
        
        return (center_r, center_c)
    
    def _analyze_color_patterns(self, input_grid, output_grid, diff_grid):
        """Analyze patterns in color changes."""
        patterns = []
        
        # Get color histograms
        input_colors = {}
        output_colors = {}
        
        for value in input_grid.flatten():
            input_colors[value] = input_colors.get(value, 0) + 1
        
        for value in output_grid.flatten():
            output_colors[value] = output_colors.get(value, 0) + 1
        
        # Look for color mappings
        for input_color, input_count in input_colors.items():
            for output_color, output_count in output_colors.items():
                if input_count == output_count and input_color != output_color:
                    # Potential color mapping
                    confidence = min(input_count, output_count) / max(input_count, output_count)
                    
                    patterns.append(Pattern(
                        id=f"color_map_{input_color}_{output_color}",
                        name=f"Color Mapping {input_color} -> {output_color}",
                        description=f"All pixels of color {input_color} changed to {output_color}",
                        confidence=confidence,
                        metadata={
                            "input_color": input_color,
                            "output_color": output_color,
                            "count": input_count,
                            "type": "color_mapping"
                        }
                    ))
        
        return patterns
    
    def _analyze_structural_patterns(self, input_grid, output_grid, diff_grid):
        """Analyze structural changes between grids."""
        patterns = []
        
        # Check for grid size changes
        if input_grid.shape != output_grid.shape:
            patterns.append(Pattern(
                id="grid_resize",
                name="Grid Resize",
                description=f"Grid resized from {input_grid.shape} to {output_grid.shape}",
                confidence=0.9,
                metadata={
                    "input_shape": input_grid.shape,
                    "output_shape": output_grid.shape,
                    "type": "grid_resize"
                }
            ))
        
        # Check for rotations
        # This would require more sophisticated analysis
        
        return patterns
    
    def _analyze_spatial_relationships(self, input_objects, output_objects):
        """Analyze spatial relationships between objects."""
        patterns = []
        
        # Check for alignment patterns
        # This would require analyzing relative positions
        
        return patterns