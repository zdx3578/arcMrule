"""CEGIS synthesiser placeholder.

For each (in_grid, out_grid) pair, we want to search for a *concrete*
transformation program λ that maps input → output exactly.

This is just a scaffold – choose your favourite solver (Rosette, z3‑py, 
PyEDA…) and fill in the details.

The λ you return should be serialisable to an S‑expression or Prolog term.
"""

from typing import Any, Dict

def synthesize_one(example: Dict[str, Any]) -> Dict[str, Any]:
    """Given a single ARC training example, return a dict:
        {
          'example_id': example['id'],
          'program_ast': '(paint (recolor obj1 red))',   # S‑expr string
          'cost'      : 3.0
        }
    """
    # TODO: implement real synthesis
    raise NotImplementedError("CEGIS synthesis not implemented")