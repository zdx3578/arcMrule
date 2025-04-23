"""
Grid processing module for handling ARC grids.
"""
import numpy as np

class GridProcessor:
    """
    Handles grid operations, transformations, and difference calculations.
    """
    def __init__(self):
        pass
    
    def calculate_diff(self, grid1, grid2):
        """
        Calculate the difference grid between two grids.
        
        Args:
            grid1: First grid (usually input)
            grid2: Second grid (usually output)
            
        Returns:
            Difference grid where:
            - Same values are marked as 0
            - Different values show the new value from grid2
        """
        if grid1.shape != grid2.shape:
            # Handle different sized grids
            # This is simplified - real implementation would be more complex
            max_rows = max(grid1.shape[0], grid2.shape[0])
            max_cols = max(grid1.shape[1], grid2.shape[1])
            
            padded_grid1 = np.zeros((max_rows, max_cols), dtype=int)
            padded_grid1[:grid1.shape[0], :grid1.shape[1]] = grid1
            
            padded_grid2 = np.zeros((max_rows, max_cols), dtype=int)
            padded_grid2[:grid2.shape[0], :grid2.shape[1]] = grid2
            
            diff = np.where(padded_grid1 == padded_grid2, 0, padded_grid2)
        else:
            diff = np.where(grid1 == grid2, 0, grid2)
        
        return diff
    
    def rotate(self, grid, degrees):
        """Rotate grid by specified degrees (90, 180, 270)."""
        k = degrees // 90
        return np.rot90(grid, k)
    
    def flip(self, grid, axis):
        """Flip grid horizontally or vertically."""
        if axis == 'horizontal':
            return np.fliplr(grid)
        elif axis == 'vertical':
            return np.flipud(grid)
        return grid
    
    def crop(self, grid, top, left, height, width):
        """Crop grid to specified dimensions."""
        return grid[top:top+height, left:left+width]
    
    def pad(self, grid, padding, value=0):
        """Pad grid with specified value."""
        return np.pad(grid, padding, constant_values=value)