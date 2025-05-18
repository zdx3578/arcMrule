class RelationMiner:
    @staticmethod
    def derive_between(vbars: list[tuple[int,int]]):
        if len(vbars) != 2:
            return []
        c1, _ = vbars[0]
        c2, _ = vbars[1]
        if c1 > c2:
            c1, c2 = c2, c1
        facts = [
            f"left_bar({c1}).",
            f"right_bar({c2}).",
            f"between_col({c1},{c2}).",
            f"inside(R,C):-between_col({c1},{c2}),C>{c1},C<{c2}.",
            f"adj_left_end(R,{c1+1}).",
            f"adj_right_end(R,{c2-1}).",
        ]
        return facts