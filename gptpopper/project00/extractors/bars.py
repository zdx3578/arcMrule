import numpy as np
from scipy.ndimage import label
from .base import BaseExtractor
class BarsExtractor(BaseExtractor):
    name = "bars"
    def extract(self, grid: np.ndarray):
        facts = []
        mask = grid > 0
        labeled, n = label(mask)
        for idx in range(1, n + 1):
            coords = np.argwhere(labeled == idx)
            rows, cols = coords[:, 0], coords[:, 1]
            if len(np.unique(cols)) == 1 and len(rows) >= 3:
                col = cols[0]
                colour = int(grid[rows[0], col])
                facts.append(self.fact("vbar", col, colour))
        # yellow rows
        ys = np.unique(np.argwhere(grid == 4)[:, 0])
        for r in ys:
            facts.append(self.fact("yellow_row", r))
        return facts