from typing import List, Tuple
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
