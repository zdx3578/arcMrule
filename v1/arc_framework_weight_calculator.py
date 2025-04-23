"""
Weight calculator for prioritizing objects in ARC grids.
"""
import numpy as np
from collections import Counter

class WeightCalculator:
    """
    Calculates weights for objects to prioritize analysis.
    """
    def __init__(self):
        # Weight factors for different object properties
        self.weight_factors = {
            'size': 1.0,        # Smaller objects get higher weight
            'color': 0.8,       # Rare colors get higher weight
            'position': 0.6,    # Objects at special positions get higher weight
            'shape': 0.7,       # Rare shapes get higher weight
            'repetition': 0.9,  # Repeated objects get higher weight
        }
    
    def calculate_weights(self, objects):
        """
        Calculate weights for a list of objects.
        
        Args:
            objects: List of GridObject instances
            
        Returns:
            List of (object, weight) tuples sorted by weight descending
        """
        if not objects:
            return []
        
        # Calculate basic statistics for normalization
        total_objects = len(objects)
        max_area = max(obj.area for obj in objects)
        color_counts = Counter(obj.color for obj in objects)
        
        # Calculate weights for each object
        weighted_objects = []
        
        for obj in objects:
            weight = 0.0
            
            # Size weight (smaller objects get higher weight)
            size_weight = 1.0 - (obj.area / max_area)
            weight += size_weight * self.weight_factors['size']
            
            # Color weight (rare colors get higher weight)
            color_frequency = color_counts[obj.color] / total_objects
            color_weight = 1.0 - color_frequency
            weight += color_weight * self.weight_factors['color']
            
            # Position weight (objects in corners or center get higher weight)
            position_weight = self._calculate_position_weight(obj)
            weight += position_weight * self.weight_factors['position']
            
            # Shape weight (based on aspect ratio)
            shape_weight = self._calculate_shape_weight(obj)
            weight += shape_weight * self.weight_factors['shape']
            
            weighted_objects.append((obj, weight))
        
        # Sort by weight descending
        weighted_objects.sort(key=lambda x: x[1], reverse=True)
        
        return weighted_objects
    
    def _calculate_position_weight(self, obj):
        """Calculate weight based on object position."""
        # Simplified version - actual implementation would be more complex
        center_r, center_c = self._calculate_center(obj)
        min_r, min_c, max_r, max_c = obj.bounding_box
        
        # Check if object is in a corner
        is_corner = (min_r == 0 or max_r == 0) and (min_c == 0 or max_c == 0)
        
        # Higher weight for objects in corners
        return 0.8 if is_corner else 0.2
    
    def _calculate_shape_weight(self, obj):
        """Calculate weight based on object shape."""
        min_r, min_c, max_r, max_c = obj.bounding_box
        height = max_r - min_r + 1
        width = max_c - min_c + 1
        
        if height == 0 or width == 0:
            return 0.5
        
        # Calculate aspect ratio
        aspect_ratio = width / height
        
        # Objects with aspect ratio far from 1 get higher weight
        return abs(aspect_ratio - 1.0)
    
    def _calculate_center(self, obj):
        """Calculate the center of an object."""
        pixels = obj.pixels
        if not pixels:
            return (0, 0)
            
        r_coords, c_coords = zip(*pixels)
        center_r = sum(r_coords) / len(r_coords)
        center_c = sum(c_coords) / len(c_coords)
        
        return (center_r, center_c)
    
    def adjust_weight_factors(self, new_factors):
        """Adjust the weight factors."""
        for factor, value in new_factors.items():
            if factor in self.weight_factors:
                self.weight_factors[factor] = value