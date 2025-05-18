import yaml, pathlib, subprocess, sys
import numpy as np
from extractors import EXTRACTOR_MAP
from util.grid_io import load_pair, COLMAP
from relation_miner import RelationMiner

def main(cfg_path: str):
    cfg = yaml.safe_load(open(cfg_path))
    grid_in, grid_out = load_pair(cfg["task_json"])

    # 1. Extract facts
    facts, vbars = [], []
    for name in cfg["extractors"]:
        ext = EXTRACTOR_MAP[name]()
        new = ext.extract(grid_in)
        facts.extend(new)
        if name == "bars":
            vbars = [tuple(map(int, f[5:-2].split(','))) for f in new if f.startswith('vbar')]

    # 2. Relation mining
    facts += RelationMiner.derive_between(vbars)

    # 3. Generate bk, exs, bias
    bk = "
".join(facts)
    exs = "
".join(
        f"pos(out({r},{c},{COLMAP[col]}))."
        for r, c in np.argwhere(grid_out)
        for col in [int(grid_out[r, c])]
        if col
    )
    body_preds = [
        'inside/2','adj_left_end/2','adj_right_end/2',
        'left_bar/1','right_bar/1','yellow_row/1'
    ]
    bias = 'head_pred(out,3).
' + '
'.join(
        f"body_pred({p.split('/')[0]},{p.split('/')[1]})." for p in body_preds
    ) + '
max_body(3).
max_clauses(3).
max_vars(6).'

    work = pathlib.Path('generated'); work.mkdir(exist_ok=True)
    (work / 'bk.pl').write_text(bk)
    (work / 'exs.pl').write_text(exs)
    (work / 'bias.pl').write_text(bias)
    print('Generated files under ./generated')

    if cfg.get('run_popper', True):
        subprocess.run(['popper', '--cwa', 'true', '--timeout', '60', str(work)], check=False)

if __name__ == '__main__':
    cfg = sys.argv[1] if len(sys.argv) > 1 else 'config.yml'
    main(cfg)
