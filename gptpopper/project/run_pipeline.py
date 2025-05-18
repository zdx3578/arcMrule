import pathlib, subprocess, sys, yaml, numpy as np
from typing import List
from extractors import EXTRACTOR_MAP
from util.grid_io import load_pair, COLMAP
from relation_miner import RelationMiner

def main(cfg_path='config.yml'):
    cfg = yaml.safe_load(open(cfg_path))
    grid_in, grid_out = load_pair(cfg['task_json'])

    # 1. facts
    facts: List[str] = []
    vbars = []
    for name in cfg['extractors']:
        ext = EXTRACTOR_MAP[name]()
        new = ext.extract(grid_in)
        facts.extend(new)
        if name == 'bars':
            vbars = [tuple(map(int, f[5:-2].split(',')))
                     for f in new if f.startswith('vbar')]

    # 2. mined relations
    facts.extend(RelationMiner.derive_between(vbars))

    # 3. write bk / exs / bias
    bk  = '\n'.join(facts) + '\n'
    exs = '\n'.join(
        f"pos(out({r},{c},{COLMAP[int(grid_out[r,c])]}))."
        for r,c in np.argwhere(grid_out) if grid_out[r,c]
    ) + '\n'

    body = ['inside/2','adj_left_end/2','adj_right_end/2',
            'left_bar/1','right_bar/1','yellow_row/1']
    bias = (
        'head_pred(out,3).\n' +
        '\n'.join(f"body_pred({p.split('/')[0]},{p.split('/')[1]})."
                  for p in body) +
        '\nmax_body(3).\nmax_clauses(3).\nmax_vars(6).\n'
    )

    outdir = pathlib.Path('generated'); outdir.mkdir(exist_ok=True)
    (outdir/'bk.pl').write_text(bk)
    (outdir/'exs.pl').write_text(exs)
    (outdir/'bias.pl').write_text(bias)
    print('Files written to', outdir)

    if cfg.get('run_popper', True):
        subprocess.run(
            ['popper','--cwa','true','--timeout','60',str(outdir)],
            check=False
        )

if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'config.yml')
