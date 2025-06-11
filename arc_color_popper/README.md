# ARC Colour‑by‑Hole‑Count (Popper Skeleton)

**Purpose**  
Demonstrates a *minimal* end‑to‑end pipeline for learning an ARC transformation
where an object's output colour depends on its number of internal holes.

── **Directory Layout**

```
arc_color_popper/
  scripts/
    extractor.py   # object extraction – implement yourself
    cegis.py       # per‑example program synthesis (CEGIS) – implement
    lgg.py         # anti‑unification helper
  popper/
    bias.pl        # mode declarations + metarules
    bk.pl          # background facts (auto‑gen stub)
    examples.pl    # pos/neg examples     (stub)
  run.sh           # convenience wrapper: ./run.sh
```

── **Typical Workflow**

1. **Object Extraction (`extractor.py`)**  
   Parse ARC JSON → produce object facts → append to `popper/bk.pl`.

2. **Per‑example CEGIS (`cegis.py`)**  
   Generate concrete λ programs; serialise as S‑expressions.

3. **Anti‑Unification (`lgg.py`)**  
   Generalise λ list to LGG; translate to `transform/2` clauses; add to `examples.pl`
   (positive ground atoms).

4. **Run Popper**  
   ```bash
   ./run.sh            # assumes popper in PATH
   ```

   Popper reads `bias.pl`, `bk.pl`, `examples.pl` and outputs the
   simplest program covering all positives and excluding negatives.

5. **Apply & Verify**  
   (Not shown) – write an `apply.py` that converts Popper's `program.pl`
   back into grid transformations and checks against ARC test grids.

── **Next Steps / TODO markers**
* Fill in `TODO` blocks across the codebase:
  - Better primitives in `bias.pl`.
  - Actual extractor logic.
  - Real CEGIS search.
  - Example generation.
* Add negative examples if needed to prune over‑general programs.
* Expand metarules / max_body bounds as your task grows.

*Generated on* 2025-06-11 05:06:03.765564