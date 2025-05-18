"""
ARC Auto‑BK Pipeline – Python 3.8‑safe
=====================================
* Replaced **`list[str]` type hints → `List[str]`** for compatibility with
  Python ≤ 3.8 (Kaggle / Colab default).
* No other logic changed; the project still scaffolds & calls Popper.

```bash
python arc_autogen_framework.py --init   # generate ./project tree
python arc_autogen_framework.py --run    # build BK/EXS/BIAS & run Popper
```
"""

import argparse, os, pathlib, subprocess, sys, yaml, json, numpy as np
from typing import Dict, List

FILES: Dict[str, str] = {
    # ---- extractors ----------------------------------------------------
    'extractors/__init__.py': '''from .pixel import PixelExtractor
from .bars import BarsExtractor
EXTRACTOR_MAP = {
    "pixel": PixelExtractor,
    "bars":  BarsExtractor,
}
''',

    'extractors/base.py': '''import abc
from typing import List
class BaseExtractor(metaclass=abc.ABCMeta):
    name: str = ''
    @abc.abstractmethod
    def extract(self, grid: 'np.ndarray') -> List[str]: ...
    @staticmethod
    def fact(name: str, *args) -> str:
        argstr = ','.join(map(str, args))
        return f"{name}({argstr})."
''',

    'extractors/pixel.py': '''import numpy as np
from .base import BaseExtractor
class PixelExtractor(BaseExtractor):
    name = "pixel"
    def extract(self, grid: np.ndarray):
        facts: List[str] = []
        h, w = grid.shape
        for r in range(h):
            for c in range(w):
                col = int(grid[r, c])
                if col:
                    facts.append(self.fact('pixel_in', r, c, col))
        return facts
''',

    'extractors/bars.py': '''import numpy as np
from typing import List
from scipy.ndimage import label
from .base import BaseExtractor
class BarsExtractor(BaseExtractor):
    name = "bars"
    def extract(self, grid: np.ndarray):
        facts: List[str] = []
        mask = grid > 0
        labeled, n = label(mask)
        for idx in range(1, n + 1):
            coords = np.argwhere(labeled == idx)
            rows, cols = coords[:,0], coords[:,1]
            if len(np.unique(cols)) == 1 and len(rows) >= 3:
                col = cols[0]
                colour = int(grid[rows[0], col])
                facts.append(self.fact('vbar', col, colour))
        ys = np.unique(np.argwhere(grid == 4)[:,0])
        for r in ys:
            facts.append(self.fact('yellow_row', r))
        return facts
''',

    # ---- relation miner ----------------------------------------------
    'relation_miner.py': '''from typing import List, Tuple
class RelationMiner:
    @staticmethod
    def derive_between(vbars: List[Tuple[int,int]]):
        if len(vbars) != 2:
            return []
        (c1,_), (c2,_) = vbars
        if c1 > c2:
            c1, c2 = c2, c1
        return [
            f"left_bar({c1}).",
            f"right_bar({c2}).",
            f"between_col({c1},{c2}).",
            f"inside(R,C):-between_col({c1},{c2}),C>{c1},C<{c2}.",
            f"adj_left_end(R,{c1 + 1}).",
            f"adj_right_end(R,{c2 - 1}).",
        ]
''',

    # ---- util ---------------------------------------------------------
    'util/grid_io.py': '''import json, numpy as np
COLMAP = {0:'bg',1:'cyan',2:'red',3:'blue',4:'yellow',5:'green'}

def load_pair(path):
    d = json.load(open(path))
    return np.array(d['train'][0]['input']), np.array(d['train'][0]['output'])
''',

    # ---- runner -------------------------------------------------------
    'run_pipeline.py': '''import yaml, pathlib, subprocess, sys
import numpy as np
from typing import List
from extractors import EXTRACTOR_MAP
from util.grid_io import load_pair, COLMAP
from relation_miner import RelationMiner

def main(cfg_path):
    cfg = yaml.safe_load(open(cfg_path))
    grid_in, grid_out = load_pair(cfg['task_json'])

    facts: List[str] = []
    vbars = []
    for name in cfg['extractors']:
        ext = EXTRACTOR_MAP[name]()
        new = ext.extract(grid_in)
        facts.extend(new)
        if name == 'bars':
            vbars = [tuple(map(int, f[5:-2].split(','))) for f in new if f.startswith('vbar')]

    facts.extend(RelationMiner.derive_between(vbars))

    bk  = '\n'.join(facts) + '\n'
    exs = '\n'.join(
        f"pos(out({r},{c},{COLMAP[int(grid_out[r,c])]}))." for r,c in np.argwhere(grid_out) if grid_out[r,c]
    ) + '\n'
    body = ['inside/2','adj_left_end/2','adj_right_end/2','left_bar/1','right_bar/1','yellow_row/1']
    bias = 'head_pred(out,3).\n' + '\n'.join(
        f"body_pred({p.split('/')[0]},{p.split('/')[1]})." for p in body
    ) + '\nmax_body(3).\nmax_clauses(3).\nmax_vars(6).\n'

    outdir = pathlib.Path('generated'); outdir.mkdir(exist_ok=True)
    (outdir/'bk.pl').write_text(bk)
    (outdir/'exs.pl').write_text(exs)
    (outdir/'bias.pl').write_text(bias)
    print('Files written to', outdir)

    if cfg.get('run_popper', True):
        subprocess.run(['popper','--cwa','true','--timeout','60',str(outdir)], check=False)

if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'config.yml')
''',
}

# ---------------------------- scaffolder ------------------------------

def scaffold():
    proj = pathlib.Path('project'); proj.mkdir(exist_ok=True)
    for rel, code in FILES.items():
        p = proj / rel; p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(code)
    (proj/'config.yml').write_text('task_json: sample.json\nextractors: [pixel, bars]\nrun_popper: true\n')
    print('Scaffold written to ./project')

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--init',action='store_true'); ap.add_argument('--run',action='store_true')
    args = ap.parse_args()
    if args.init:
        scaffold(); sys.exit(0)
    if args.run:
        sys.path.insert(0,'project'); os.chdir('project'); import run_pipeline as rp; rp.main('config.yml'); sys.exit(0)
    print('Use --init or --run')
