"""
Object extraction module for identifying objects in ARC grids.
"""
import numpy as np
from skimage.measure import label, regionprops
from dataclasses import dataclass
from typing import List, Tuple, Set

@dataclass
class GridObject:
    """Represents an object extracted from a grid."""
    id: int
    color: int
    pixels: List[Tuple[int, int]]
    bounding_box: Tuple[int, int, int, int]  # (min_row, min_col, max_row, max_col)
    area: int
    sub_objects: List['GridObject'] = None
    
    @property
    def height(self):
        return self.bounding_box[2] - self.bounding_box[0]
    
    @property
    def width(self):
        return self.bounding_box[3] - self.bounding_box[1]
    
    def get_mask(self, grid_shape):
        """Create a binary mask for this object."""
        mask = np.zeros(grid_shape, dtype=bool)
        for r, c in self.pixels:
            mask[r, c] = True
        return mask
    
    def get_subgrid(self, grid):
        """Extract the subgrid containing just this object."""
        min_r, min_c, max_r, max_c = self.bounding_box
        subgrid = np.zeros((max_r - min_r + 1, max_c - min_c + 1), dtype=int)
        
        for r, c in self.pixels:
            subgrid[r - min_r, c - min_c] = grid[r, c]
            
        return subgrid


class ObjectExtractor:
    """
    Identifies and extracts objects from grids with hierarchical decomposition.
    """
    def __init__(self, shape_priors=None):
        self.shape_priors = shape_priors or {}
    
    def extract_objects(self, grid):
        """
        Extract objects from a grid.
        
        Args:
            grid: Input grid as numpy array
            
        Returns:
            List of GridObject instances
        """
        objects = []
        
        # Get all unique colors (excluding background, assumed to be 0)
        colors = np.unique(grid)
        colors = colors[colors != 0]  # Remove background
        
        for color in colors:
            # Create a binary mask for current color
            mask = (grid == color)
            
            # Label connected components
            labeled_mask, num_components = label(mask, return_num=True, connectivity=1)
            
            for component_id in range(1, num_components + 1):
                # Get pixels for this component
                component_mask = (labeled_mask == component_id)
                pixels = np.argwhere(component_mask).tolist()
                
                # Calculate bounding box
                if not pixels:
                    continue
                    
                r_coords, c_coords = zip(*pixels)
                min_r, max_r = min(r_coords), max(r_coords)
                min_c, max_c = min(c_coords), max(c_coords)
                
                # Create object
                obj = GridObject(
                    id=len(objects),
                    color=color,
                    pixels=pixels,
                    bounding_box=(min_r, min_c, max_r, max_c),
                    area=len(pixels),
                    sub_objects=[]
                )
                
                # Generate sub-objects
                self._generate_sub_objects(obj, grid)
                
                objects.append(obj)
        
        return objects
    
    def _generate_sub_objects(self, obj, grid):
        """Generate hierarchical sub-objects for a given object."""
        # Generate N-1 size sub-objects
        if len(obj.pixels) <= 1:
            return
        
        # Generate sub-objects recursively by removing one pixel at a time
        for i in range(len(obj.pixels)):
            sub_pixels = obj.pixels.copy()
            removed_pixel = sub_pixels.pop(i)
            
            # Check if sub-pixels are still connected
            if self._is_connected(sub_pixels):
                r_coords, c_coords = zip(*sub_pixels)
                min_r, max_r = min(r_coords), max(r_coords)
                min_c, max_c = min(c_coords), max(c_coords)
                
                sub_obj = GridObject(
                    id=len(obj.sub_objects),
                    color=obj.color,
                    pixels=sub_pixels,
                    bounding_box=(min_r, min_c, max_r, max_c),
                    area=len(sub_pixels),
                    sub_objects=[]
                )
                
                # Recursively generate sub-objects (limited to avoid combinatorial explosion)
                if len(sub_pixels) > 3:  # Only generate sub-objects for larger objects
                    self._generate_sub_objects(sub_obj, grid)
                
                obj.sub_objects.append(sub_obj)
    
    def _is_connected(self, pixels):
        """Check if a set of pixels is connected."""
        if not pixels:
            return True
        
        # Simple BFS to check connectivity
        visited = set()
        queue = [pixels[0]]
        
        while queue:
            r, c = queue.pop(0)
            if (r, c) in visited:
                continue
                
            visited.add((r, c))
            
            # Check neighbors
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if (nr, nc) in pixels and (nr, nc) not in visited:
                    queue.append((nr, nc))
        
        return len(visited) == len(pixels)
    
    def match_shape_prior(self, obj):
        """Try to match an object to known shape priors."""
        # Implementation would check against known shapes like rectangles, circles, etc.
        matched_priors = []
        
        # Example: Check if the object is a rectangle
        min_r, min_c, max_r, max_c = obj.bounding_box
        height = max_r - min_r + 1
        width = max_c - min_c + 1
        
        if obj.area == height * width:
            matched_priors.append(("rectangle", 1.0))
        
        # Add more shape recognition logic
        
        return matched_priors