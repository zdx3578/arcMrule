"""
Standard actions that can be used by rules in the ARC framework.
"""
import numpy as np

def move_objects(grid, objects, metadata):
    """
    Move objects in the grid.
    
    Args:
        grid: The input grid
        objects: List of objects in the grid
        metadata: Contains dx and dy for the movement
        
    Returns:
        Transformed grid
    """
    dx = metadata.get('dx', 0)
    dy = metadata.get('dy', 0)
    
    if dx == 0 and dy == 0:
        return grid.copy()
    
    # Create a new empty grid
    new_grid = np.zeros_like(grid)
    
    # Move each object
    for obj in objects:
        for r, c in obj.pixels:
            new_r = r + dy
            new_c = c + dx
            
            # Check if new position is within bounds
            if 0 <= new_r < grid.shape[0] and 0 <= new_c < grid.shape[1]:
                new_grid[new_r, new_c] = grid[r, c]
    
    return new_grid

def change_color(grid, objects, metadata):
    """
    Change colors in the grid.
    
    Args:
        grid: The input grid
        objects: List of objects in the grid
        metadata: Contains input_color and output_color
        
    Returns:
        Transformed grid
    """
    input_color = metadata.get('input_color')
    output_color = metadata.get('output_color')
    
    if input_color is None or output_color is None:
        return grid.copy()
    
    # Create a new grid
    new_grid = grid.copy()
    
    # Change colors
    new_grid[new_grid == input_color] = output_color
    
    return new_grid

def rotate_grid(grid, objects, metadata):
    """
    Rotate the entire grid.
    
    Args:
        grid: The input grid
        objects: List of objects in the grid
        metadata: Contains degrees for rotation
        
    Returns:
        Transformed grid
    """
    degrees = metadata.get('degrees', 90)
    k = degrees // 90  # Number of 90-degree rotations
    
    return np.rot90(grid, k)

def flip_grid(grid, objects, metadata):
    """
    Flip the grid horizontally or vertically.
    
    Args:
        grid: The input grid
        objects: List of objects in the grid
        metadata: Contains flip_axis ('horizontal' or 'vertical')
        
    Returns:
        Transformed grid
    """
    flip_axis = metadata.get('flip_axis', 'horizontal')
    
    if flip_axis == 'horizontal':
        return np.fliplr(grid)
    elif flip_axis == 'vertical':
        return np.flipud(grid)
    
    return grid.copy()

def resize_objects(grid, objects, metadata):
    """
    Resize objects in the grid.
    
    Args:
        grid: The input grid
        objects: List of objects in the grid
        metadata: Contains scale_factor for resizing
        
    Returns:
        Transformed grid
    """
    scale_factor = metadata.get('scale_factor', 1.0)
    target_color = metadata.get('target_color')
    
    if scale_factor == 1.0:
        return grid.copy()
    
    # Create a new grid
    new_grid = np.zeros_like(grid)
    
    # Process each object
    for obj in objects:
        # Skip objects that don't match target color
        if target_color is not None and obj.color != target_color:
            # Keep original pixels for non-matching objects
            for r, c in obj.pixels:
                new_grid[r, c] = grid[r, c]
            continue
        
        # Get object dimensions
        min_r, min_c, max_r, max_c = obj.bounding_box
        height = max_r - min_r + 1
        width = max_c - min_c + 1
        
        # Calculate new dimensions
        new_height = int(height * scale_factor)
        new_width = int(width * scale_factor)
        
        # Skip if new dimensions are invalid
        if new_height <= 0 or new_width <= 0:
            continue
        
        # Calculate center position
        center_r = (min_r + max_r) // 2
        center_c = (min_c + max_c) // 2
        
        # Calculate new bounding box
        new_min_r = center_r - new_height // 2
        new_min_c = center_c - new_width // 2
        
        # Create a mask for the original object
        mask = np.zeros((height, width), dtype=bool)
        for r, c in obj.pixels:
            mask[r - min_r, c - min_c] = True
        
        # Resize the mask
        # In practice, would use proper image processing methods
        # This is a simplified version
        resized_mask = np.zeros((new_height, new_width), dtype=bool)
        
        if scale_factor > 1.0:
            # Upscaling - copy each pixel multiple times
            for r in range(height):
                for c in range(width):
                    if mask[r, c]:
                        r_start = int(r * scale_factor)
                        c_start = int(c * scale_factor)
                        r_end = int((r + 1) * scale_factor)
                        c_end = int((c + 1) * scale_factor)
                        
                        for nr in range(r_start, r_end):
                            for nc in range(c_start, c_end):
                                if 0 <= nr < new_height and 0 <= nc < new_width:
                                    resized_mask[nr, nc] = True
        else:
            # Downscaling - sample pixels
            for r in range(new_height):
                for c in range(new_width):
                    orig_r = int(r / scale_factor)
                    orig_c = int(c / scale_factor)
                    if 0 <= orig_r < height and 0 <= orig_c < width:
                        resized_mask[r, c] = mask[orig_r, orig_c]
        
        # Apply the resized mask to the new grid
        for r in range(new_height):
            for c in range(new_width):
                if resized_mask[r, c]:
                    grid_r = new_min_r + r
                    grid_c = new_min_c + c
                    
                    if 0 <= grid_r < grid.shape[0] and 0 <= grid_c < grid.shape[1]:
                        new_grid[grid_r, grid_c] = obj.color
    
    return new_grid