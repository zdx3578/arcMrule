import numpy as np
from .base import BaseExtractor
class PixelExtractor(BaseExtractor):
    name = "pixel"
    def extract(self, grid: np.ndarray):
        facts = []
        h, w = grid.shape
        for r in range(h):
            for c in range(w):
                col = int(grid[r, c])
                if col:
                    facts.append(self.fact("pixel_in", r, c, col))
        return facts