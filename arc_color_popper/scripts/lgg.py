"""Very small anti‑unification helper.

Works on S‑expressions encoded as nested python lists.
Returns a single LGG AST with '_' wildcards.
"""

import ast
from typing import List, Any

def anti_unify_many(exprs: List[Any]) -> Any:
    """exprs: list of parsed S‑expr (python lists / atoms)."""

    def unify(a, b):
        if a == b:
            return a
        if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
            return [unify(x,y) for x,y in zip(a,b)]
        return '_'   # wildcard

    g = exprs[0]
    for e in exprs[1:]:
        g = unify(g, e)
    return g