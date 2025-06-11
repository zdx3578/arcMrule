"""Object extractor for ARC grids (skeleton).

Assumptions:
- `grid` is a 2‑D list/NumPy array of ints (0‑9 colors, 0 = background).
- Objects are 4‑connected components of non‑background color.
- A 'hole' is a background pixel fully enclosed by the object.
The implementation below is a **placeholder** – plug in your own logic.
"""

from typing import List, Dict, Any, Tuple

def extract_objects(grid) -> List[Dict[str, Any]]:
    """Return a list of object dicts.
    Each dict at least contains:
    id      : str   – unique object id
    pixels  : List[Tuple[int,int]]
    color   : int   – original color id
    holes   : int   – number of internal holes
    bbox    : Tuple[int,int,int,int]  – (min_row,min_col,max_row,max_col)
    """
    # TODO: replace with your own extraction method (no CNN needed)
    raise NotImplementedError("extract_objects not implemented")

if __name__ == "__main__":
    # quick smoke test with a dummy 3×3 single‑pixel object grid
    dummy = [[1,0,0],[0,0,0],[0,0,2]]
    try:
        objs = extract_objects(dummy)
        print(objs)
    except NotImplementedError:
        print("[INFO] extractor is a stub – implement me first!")