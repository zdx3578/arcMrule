# # Comprehensive Popper-based Program Synthesis Framework for ARC Tasks

# ## Project Architecture Overview

# This framework integrates Popper (Inductive Logic Programming), object extraction, CEGIS (Counterexample-Guided Inductive Synthesis), and anti-unification into a unified system for solving ARC (Abstraction and Reasoning Corpus) tasks through program synthesis.

# ## [](https://)Complete Project Structure

# ```
# arc_synthesis_framework/
# ├── core/
# │   ├── __init__.py
# │   ├── synthesis_engine.py      # Main CEGIS engine
# │   ├── popper_interface.py      # Popper ILP integration
# │   ├── anti_unification.py     # Pattern generalization
# │   └── oracle.py               # Solution verification
# ├── extraction/
# │   ├── __init__.py
# │   ├── object_extractor.py     # Grid object detection
# │   ├── spatial_predicates.py   # Spatial reasoning
# │   └── transformations.py      # Color/shape transformations
# ├── popper_files/
# │   ├── bias/                   # Bias files for different task types
# │   ├── background/             # Background knowledge files
# │   ├── examples/               # Generated example files
# │   └── templates/              # Template files
# ├── cegis/
# │   ├── __init__.py
# │   ├── synthesizer.py          # Candidate generation
# │   ├── verifier.py             # Program verification
# │   └── counterexample.py       # Counterexample generation
# ├── utils/
# │   ├── __init__.py
# │   ├── arc_loader.py           # ARC dataset utilities
# │   ├── logging.py              # Comprehensive logging
# │   └── metrics.py              # Performance metrics
# ├── examples/
# │   ├── simple_tasks/           # Basic ARC task examples
# │   ├── complex_tasks/          # Advanced transformation examples
# │   └── demonstrations/         # End-to-end demos
# ├── tests/
# ├── benchmarks/
# └── docs/
# ```

# ## Core Implementation

# ### 1. Main Synthesis Engine (core/synthesis_engine.py)

```python
"""
Main CEGIS-based synthesis engine integrating all components
"""
import logging
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
import json
import time

from .popper_interface import PopperInterface
from .anti_unification import AntiUnifier
from .oracle import SolutionOracle
from ..extraction.object_extractor import ARCObjectExtractor
from ..cegis.synthesizer import CEGISSynthesizer
from ..cegis.verifier import ProgramVerifier
from ..cegis.counterexample import CounterexampleGenerator
from ..utils.arc_loader import ARCDataLoader
from ..utils.metrics import SynthesisMetrics

@dataclass
class SynthesisTask:
    """Represents an ARC synthesis task"""
    task_id: str
    train_pairs: List[Tuple[List[List[int]], List[List[int]]]]
    test_pairs: List[Tuple[List[List[int]], List[List[int]]]]
    metadata: Dict[str, Any]

@dataclass
class SynthesisResult:
    """Results from program synthesis"""
    success: bool
    program: Optional[str]
    confidence: float
    synthesis_time: float
    iterations: int
    counterexamples_used: int
    generalization_pattern: Optional[str]

class ARCSynthesisEngine:
    """
    Main synthesis engine combining Popper ILP, CEGIS, object extraction,
    and anti-unification for ARC tasks
    """

    def __init__(self, config_path: str = "config/synthesis.yaml"):
        """Initialize synthesis engine with configuration"""
        self.config = self._load_config(config_path)
        self._setup_logging()

        # Initialize core components
        self.popper = PopperInterface(self.config['popper'])
        self.object_extractor = ARCObjectExtractor(self.config['extraction'])
        self.cegis_synthesizer = CEGISSynthesizer(self.config['cegis'])
        self.verifier = ProgramVerifier(self.config['verification'])
        self.counterexample_gen = CounterexampleGenerator(self.config['counterexamples'])
        self.anti_unifier = AntiUnifier(self.config['anti_unification'])
        self.oracle = SolutionOracle(self.config['oracle'])
        self.metrics = SynthesisMetrics()

        # Synthesis state
        self.synthesis_history = []
        self.learned_patterns = {}

        logger.info("ARC Synthesis Engine initialized")

    def synthesize_program(self, task: SynthesisTask) -> SynthesisResult:
        """
        Main synthesis method implementing CEGIS loop with ILP integration

        Args:
            task: ARC task specification

        Returns:
            SynthesisResult with program and metadata
        """
        start_time = time.time()
        logger.info(f"Starting synthesis for task {task.task_id}")

        try:
            # Phase 1: Object extraction and analysis
            extracted_objects = self._extract_objects_from_pairs(task.train_pairs)
            spatial_relations = self._analyze_spatial_relations(extracted_objects)

            # Phase 2: Generate Popper input files
            popper_files = self._generate_popper_files(
                task, extracted_objects, spatial_relations
            )

            # Phase 3: CEGIS synthesis loop
            result = self._cegis_synthesis_loop(task, popper_files)

            # Phase 4: Anti-unification and generalization
            if result.success:
                generalized_pattern = self._generalize_solution(result.program, task)
                result.generalization_pattern = generalized_pattern

            # Phase 5: Final validation
            if result.success:
                validation_result = self._validate_solution(result.program, task)
                result.success = validation_result

            result.synthesis_time = time.time() - start_time
            self._log_synthesis_result(task, result)

            return result

        except Exception as e:
            logger.error(f"Synthesis failed for task {task.task_id}: {str(e)}")
            return SynthesisResult(
                success=False,
                program=None,
                confidence=0.0,
                synthesis_time=time.time() - start_time,
                iterations=0,
                counterexamples_used=0,
                generalization_pattern=None
            )

    def _cegis_synthesis_loop(self, task: SynthesisTask, popper_files: Dict) -> SynthesisResult:
        """
        Implement CEGIS loop integrating Popper for candidate generation
        """
        counterexamples = []
        iteration = 0
        max_iterations = self.config['cegis']['max_iterations']

        while iteration < max_iterations:
            logger.info(f"CEGIS iteration {iteration + 1}")

            # Generate candidate using Popper ILP
            candidate = self._generate_candidate_with_popper(
                popper_files, counterexamples, iteration
            )

            if candidate is None:
                logger.info("No more candidates available - synthesis failed")
                return SynthesisResult(
                    success=False, program=None, confidence=0.0,
                    synthesis_time=0, iterations=iteration,
                    counterexamples_used=len(counterexamples),
                    generalization_pattern=None
                )

            # Verify candidate against all training examples
            verification_result = self.verifier.verify_candidate(candidate, task.train_pairs)

            if verification_result.is_valid:
                logger.info(f"Valid candidate found in iteration {iteration + 1}")
                confidence = self._calculate_confidence(candidate, task)
                return SynthesisResult(
                    success=True, program=candidate, confidence=confidence,
                    synthesis_time=0, iterations=iteration + 1,
                    counterexamples_used=len(counterexamples),
                    generalization_pattern=None
                )
            else:
                # Generate counterexample and continue
                new_counterexample = self.counterexample_gen.generate(
                    candidate, verification_result.failed_example
                )
                counterexamples.append(new_counterexample)
                logger.info(f"Added counterexample {len(counterexamples)}")

            iteration += 1

        return SynthesisResult(
            success=False, program=None, confidence=0.0,
            synthesis_time=0, iterations=max_iterations,
            counterexamples_used=len(counterexamples),
            generalization_pattern=None
        )

    def _extract_objects_from_pairs(self, train_pairs: List[Tuple]) -> Dict:
        """Extract objects from all input-output pairs"""
        all_objects = {}

        for pair_idx, (input_grid, output_grid) in enumerate(train_pairs):
            # Extract objects from input grid
            input_objects = self.object_extractor.process_arc_grid(input_grid)
            output_objects = self.object_extractor.process_arc_grid(output_grid)

            # Analyze transformation between input and output
            transformation = self.object_extractor.extract_transformation_pattern(
                input_grid, output_grid
            )

            all_objects[f"pair_{pair_idx}"] = {
                'input_objects': input_objects,
                'output_objects': output_objects,
                'transformation': transformation
            }

        return all_objects

    def _generate_popper_files(self, task: SynthesisTask, objects: Dict, relations: Dict) -> Dict:
        """Generate Popper input files from extracted information"""

        # Generate examples file (exs.pl)
        examples_content = self._generate_examples_file(task, objects)

        # Generate background knowledge file (bk.pl)
        bk_content = self._generate_background_knowledge(objects, relations)

        # Generate bias file (bias.pl)
        bias_content = self._generate_bias_file(task, objects)

        # Write files to disk
        task_dir = Path(f"popper_files/tasks/{task.task_id}")
        task_dir.mkdir(parents=True, exist_ok=True)

        files = {
            'examples': task_dir / "exs.pl",
            'background': task_dir / "bk.pl",
            'bias': task_dir / "bias.pl"
        }

        files['examples'].write_text(examples_content)
        files['background'].write_text(bk_content)
        files['bias'].write_text(bias_content)

        return files

    def _generate_examples_file(self, task: SynthesisTask, objects: Dict) -> str:
        """Generate Popper examples file from ARC task"""
        examples = []

        for pair_idx, (input_grid, output_grid) in enumerate(task.train_pairs):
            # Convert grids to Popper predicates
            input_pred = self._grid_to_predicate(input_grid, f"input_{pair_idx}")
            output_pred = self._grid_to_predicate(output_grid, f"output_{pair_idx}")

            # Create transformation example
            transform_example = f"pos(transform({input_pred}, {output_pred}))."
            examples.append(transform_example)

        return "\\n".join(examples)

    def _generate_background_knowledge(self, objects: Dict, relations: Dict) -> str:
        """Generate background knowledge from extracted objects and relations"""
        bk_lines = [
            "% Background knowledge generated from object extraction",
            "",
            "% Grid utility predicates",
            "grid_size(Grid, Width, Height) :-",
            "    Grid = grid(Rows),",
            "    length(Rows, Height),",
            "    Rows = [FirstRow|_],",
            "    length(FirstRow, Width).",
            "",
            "in_bounds(X, Y, Grid) :-",
            "    grid_size(Grid, W, H),",
            "    X >= 0, X < W,",
            "    Y >= 0, Y < H.",
            "",
            "% Object detection predicates",
            "detect_objects(Grid, Objects) :-",
            "    findall(obj(Type,Coords,Color,Size),",
            "            detect_object_type(Grid,Type,Coords,Color,Size),",
            "            Objects).",
            "",
            "% Spatial relationship predicates"
        ]

        # Add spatial predicates based on extracted relations
        for relation_type in ['adjacent', 'above', 'below', 'left_of', 'right_of']:
            bk_lines.extend([
                f"{relation_type}(Obj1, Obj2) :-",
                f"    object_centroid(Obj1, X1, Y1),",
                f"    object_centroid(Obj2, X2, Y2),",
                f"    {relation_type}_coords(X1, Y1, X2, Y2).",
                ""
            ])

        # Add transformation predicates
        bk_lines.extend([
            "% Transformation predicates",
            "translate_object(ObjIn, DX, DY, ObjOut) :-",
            "    object_coords(ObjIn, Coords),",
            "    translate_coords(Coords, DX, DY, NewCoords),",
            "    update_object_coords(ObjIn, NewCoords, ObjOut).",
            "",
            "change_color(ObjIn, NewColor, ObjOut) :-",
            "    object_color(ObjIn, _),",
            "    update_object_color(ObjIn, NewColor, ObjOut).",
            ""
        ])

        return "\\n".join(bk_lines)

    def _generate_bias_file(self, task: SynthesisTask, objects: Dict) -> str:
        """Generate Popper bias file for ARC spatial reasoning"""
        bias_lines = [
            "% Bias file for ARC spatial reasoning tasks",
            "",
            "% Head predicates",
            "head_pred(transform,2).",
            "head_pred(pattern,1).",
            "",
            "% Body predicates for spatial reasoning",
            "body_pred(cell,3).           % cell(X,Y,Color)",
            "body_pred(adjacent,4).       % adjacent(X1,Y1,X2,Y2)",
            "body_pred(above,2).          % above(Obj1,Obj2)",
            "body_pred(below,2).          % below(Obj1,Obj2)",
            "body_pred(left_of,2).        % left_of(Obj1,Obj2)",
            "body_pred(right_of,2).       % right_of(Obj1,Obj2)",
            "",
            "% Object manipulation predicates",
            "body_pred(translate_object,4). % translate_object(ObjIn,DX,DY,ObjOut)",
            "body_pred(rotate_object,3).    % rotate_object(ObjIn,Angle,ObjOut)",
            "body_pred(scale_object,3).     % scale_object(ObjIn,Factor,ObjOut)",
            "body_pred(change_color,3).     % change_color(ObjIn,Color,ObjOut)",
            "",
            "% Shape and attribute predicates",
            "body_pred(rectangle,1).      % rectangle(Object)",
            "body_pred(line,1).          % line(Object)",
            "body_pred(color,2).         % color(Object,Color)",
            "body_pred(size,2).          % size(Object,Size)",
            "",
            "% Type annotations",
            "type(transform,(grid,grid)).",
            "type(cell,(int,int,color)).",
            "type(translate_object,(object,int,int,object)).",
            "type(color,(object,color)).",
            "",
            "% Control settings",
            "max_vars(8).",
            "max_body(10).",
            "max_rules(5)."
        ]

        return "\\n".join(bias_lines)

    # Additional helper methods for anti-unification, validation, etc.
    def _generalize_solution(self, program: str, task: SynthesisTask) -> str:
        """Use anti-unification to generalize solution across examples"""
        return self.anti_unifier.generalize_program(program, task.train_pairs)

    def _validate_solution(self, program: str, task: SynthesisTask) -> bool:
        """Final validation of synthesized program"""
        return self.oracle.validate_program(program, task.test_pairs)
```

### 2. Popper Interface (core/popper_interface.py)

```python
"""
Interface for integrating with Popper ILP system
"""
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class PopperInterface:
    """Interface for running Popper ILP on generated problem files"""

    def __init__(self, config: Dict):
        self.config = config
        self.popper_path = Path(config.get('popper_path', './popper'))
        self.timeout = config.get('timeout', 600)
        self.solver = config.get('solver', 'rc2')
        self.max_vars = config.get('max_vars', 8)
        self.max_body = config.get('max_body', 10)

        # Verify Popper installation
        if not self._verify_popper_installation():
            raise RuntimeError("Popper not found or not properly installed")

    def learn_program(self, task_dir: Path, constraints: List[str] = None) -> Optional[str]:
        """
        Run Popper on task directory to learn program

        Args:
            task_dir: Directory containing exs.pl, bk.pl, bias.pl
            constraints: Additional constraints from CEGIS feedback

        Returns:
            Learned program as string, or None if learning failed
        """
        logger.info(f"Running Popper on {task_dir}")

        try:
            # Prepare modified files if constraints provided
            if constraints:
                working_dir = self._create_constrained_task(task_dir, constraints)
            else:
                working_dir = task_dir

            # Build Popper command
            cmd = [
                'python', str(self.popper_path / 'popper.py'),
                str(working_dir),
                '--timeout', str(self.timeout),
                '--solver', self.solver,
                '--stats'
            ]

            if self.config.get('noisy', False):
                cmd.append('--noisy')

            # Run Popper
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout + 30  # Extra buffer for process management
            )

            if result.returncode == 0:
                program = self._parse_popper_output(result.stdout)
                logger.info(f"Popper succeeded: {program}")
                return program
            else:
                logger.warning(f"Popper failed: {result.stderr}")
                return None

        except subprocess.TimeoutExpired:
            logger.warning(f"Popper timed out after {self.timeout} seconds")
            return None
        except Exception as e:
            logger.error(f"Error running Popper: {str(e)}")
            return None
        finally:
            # Clean up temporary files
            if constraints and working_dir != task_dir:
                shutil.rmtree(working_dir)

    def _create_constrained_task(self, original_dir: Path, constraints: List[str]) -> Path:
        """Create temporary task directory with additional constraints"""
        temp_dir = Path(tempfile.mkdtemp(prefix="popper_constrained_"))

        # Copy original files
        for file_name in ['exs.pl', 'bk.pl', 'bias.pl']:
            src = original_dir / file_name
            dst = temp_dir / file_name
            if src.exists():
                shutil.copy2(src, dst)

        # Add constraints to bias file
        bias_file = temp_dir / 'bias.pl'
        if bias_file.exists():
            with open(bias_file, 'a') as f:
                f.write("\\n% CEGIS-generated constraints\\n")
                for constraint in constraints:
                    f.write(f"{constraint}\\n")

        return temp_dir

    def _parse_popper_output(self, output: str) -> Optional[str]:
        """Parse Popper output to extract learned program"""
        lines = output.strip().split('\\n')
        program_lines = []
        in_program = False

        for line in lines:
            line = line.strip()
            if line.startswith('Program:') or in_program:
                if line.startswith('Program:'):
                    in_program = True
                    continue
                if line.startswith('Precision:') or line.startswith('Recall:'):
                    break
                if line and not line.startswith('%'):
                    program_lines.append(line)

        return '\\n'.join(program_lines) if program_lines else None

    def _verify_popper_installation(self) -> bool:
        """Verify that Popper is installed and accessible"""
        try:
            result = subprocess.run(
                ['python', str(self.popper_path / 'popper.py'), '--help'],
                capture_output=True,
                timeout=10
            )
            return result.returncode == 0
        except:
            return False
```

### 3. Object Extractor (extraction/object_extractor.py)

```python
"""
Advanced object extraction from ARC grids using connected components
"""
import numpy as np
from typing import List, Dict, Set, Tuple, Any
from collections import defaultdict, Counter
import math
from dataclasses import dataclass

@dataclass
class ARCObject:
    """Represents an extracted object from ARC grid"""
    id: int
    cells: Set[Tuple[int, int]]
    color: int
    bbox: Tuple[int, int, int, int]  # min_row, min_col, max_row, max_col
    centroid: Tuple[float, float]
    size: int
    shape_type: str
    attributes: Dict[str, Any]

class ARCObjectExtractor:
    """
    Comprehensive object extraction system for ARC grids
    Implements connected components with spatial analysis
    """

    def __init__(self, config: Dict):
        self.config = config
        self.connectivity = config.get('connectivity', 4)
        self.min_object_size = config.get('min_object_size', 1)
        self.background_color = config.get('background_color', 0)

    def process_arc_grid(self, grid: List[List[int]]) -> Dict[str, Any]:
        """
        Complete processing pipeline for ARC grid

        Args:
            grid: 2D list representing ARC grid

        Returns:
            Dictionary with objects, relationships, and analysis
        """
        if not grid or not grid[0]:
            return {'objects': [], 'relationships': [], 'grid_analysis': {}}

        # Convert to numpy for efficient processing
        np_grid = np.array(grid)

        # Extract connected components
        components = self._extract_connected_components(np_grid)

        # Analyze each component
        objects = []
        for comp_id, component_data in components.items():
            if len(component_data['cells']) >= self.min_object_size:
                obj = self._analyze_object(component_data, np_grid, comp_id)
                if obj:
                    objects.append(obj)

        # Analyze spatial relationships
        relationships = self._analyze_relationships(objects)

        # Grid-level analysis
        grid_analysis = self._analyze_grid_structure(np_grid, objects)

        return {
            'objects': objects,
            'relationships': relationships,
            'grid_analysis': grid_analysis,
            'grid': grid
        }

    def _extract_connected_components(self, grid: np.ndarray) -> Dict[int, Dict]:
        """
        Extract connected components using Union-Find algorithm
        Optimized implementation with path compression
        """
        rows, cols = grid.shape
        parent = list(range(rows * cols))
        rank = [0] * (rows * cols)

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])  # Path compression
            return parent[x]

        def union(x, y):
            px, py = find(x), find(y)
            if px == py:
                return
            if rank[px] < rank[py]:
                px, py = py, px
            parent[py] = px
            if rank[px] == rank[py]:
                rank[px] += 1

        # Define connectivity patterns
        if self.connectivity == 4:
            directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        else:  # 8-connected
            directions = [(0, 1), (1, 0), (0, -1), (-1, 0),
                         (1, 1), (1, -1), (-1, 1), (-1, -1)]

        # Build union-find structure
        for r in range(rows):
            for c in range(cols):
                if grid[r, c] == self.background_color:
                    continue

                current_idx = r * cols + c
                for dr, dc in directions:
                    nr, nc = r + dr, c + dc
                    if (0 <= nr < rows and 0 <= nc < cols and
                        grid[nr, nc] == grid[r, c]):
                        neighbor_idx = nr * cols + nc
                        union(current_idx, neighbor_idx)

        # Group cells by component
        components = defaultdict(lambda: {'cells': set(), 'color': 0})
        for r in range(rows):
            for c in range(cols):
                if grid[r, c] != self.background_color:
                    idx = r * cols + c
                    root = find(idx)
                    components[root]['cells'].add((r, c))
                    components[root]['color'] = grid[r, c]

        return dict(components)

    def _analyze_object(self, component_data: Dict, grid: np.ndarray, obj_id: int) -> ARCObject:
        """Comprehensive object attribute analysis"""
        cells = component_data['cells']
        color = component_data['color']

        if not cells:
            return None

        # Basic geometric properties
        rows = [r for r, c in cells]
        cols = [c for r, c in cells]
        bbox = (min(rows), min(cols), max(rows), max(cols))
        centroid = (sum(rows) / len(rows), sum(cols) / len(cols))
        size = len(cells)

        # Shape analysis
        shape_type = self._classify_shape(cells, bbox)

        # Advanced attributes
        attributes = {
            'width': bbox[3] - bbox[1] + 1,
            'height': bbox[2] - bbox[0] + 1,
            'aspect_ratio': (bbox[3] - bbox[1] + 1) / (bbox[2] - bbox[0] + 1),
            'filled_ratio': size / ((bbox[2] - bbox[0] + 1) * (bbox[3] - bbox[1] + 1)),
            'perimeter': self._compute_perimeter(cells),
            'holes': self._count_holes(cells, bbox),
            'symmetry': self._analyze_symmetry(cells),
            'compactness': self._compute_compactness(cells),
            'connectivity_type': self._analyze_connectivity(cells)
        }

        return ARCObject(
            id=obj_id,
            cells=cells,
            color=color,
            bbox=bbox,
            centroid=centroid,
            size=size,
            shape_type=shape_type,
            attributes=attributes
        )

    def _classify_shape(self, cells: Set[Tuple[int, int]], bbox: Tuple[int, int, int, int]) -> str:
        """Classify object shape based on geometric properties"""
        width = bbox[3] - bbox[1] + 1
        height = bbox[2] - bbox[0] + 1
        size = len(cells)
        expected_rectangular_size = width * height

        # Rectangle check
        if size == expected_rectangular_size:
            if width == height:
                return "square"
            else:
                return "rectangle"

        # Line check
        if width == 1 or height == 1:
            return "line"

        # Single cell
        if size == 1:
            return "point"

        # L-shape detection
        if self._is_l_shape(cells, bbox):
            return "l_shape"

        # T-shape detection
        if self._is_t_shape(cells, bbox):
            return "t_shape"

        # Cross detection
        if self._is_cross(cells, bbox):
            return "cross"

        return "irregular"

    def _analyze_relationships(self, objects: List[ARCObject]) -> List[Dict]:
        """Analyze spatial relationships between objects"""
        relationships = []

        for i, obj1 in enumerate(objects):
            for j, obj2 in enumerate(objects[i+1:], i+1):
                rel = {
                    'obj1_id': obj1.id,
                    'obj2_id': obj2.id,
                    'spatial': self._compute_spatial_relations(obj1, obj2),
                    'distance': self._compute_distance(obj1, obj2),
                    'color_relation': self._analyze_color_relation(obj1, obj2),
                    'size_relation': self._analyze_size_relation(obj1, obj2)
                }
                relationships.append(rel)

        return relationships

    def _compute_spatial_relations(self, obj1: ARCObject, obj2: ARCObject) -> Dict[str, bool]:
        """Compute comprehensive spatial relationships"""
        return {
            'adjacent': self._are_adjacent(obj1, obj2),
            'overlapping': bool(obj1.cells & obj2.cells),
            'aligned_horizontal': abs(obj1.centroid[0] - obj2.centroid[0]) < 0.5,
            'aligned_vertical': abs(obj1.centroid[1] - obj2.centroid[1]) < 0.5,
            'obj1_above_obj2': obj1.bbox[2] < obj2.bbox[0],
            'obj1_below_obj2': obj1.bbox[0] > obj2.bbox[2],
            'obj1_left_of_obj2': obj1.bbox[3] < obj2.bbox[1],
            'obj1_right_of_obj2': obj1.bbox[1] > obj2.bbox[3],
            'contains': obj1.cells.issuperset(obj2.cells) or obj2.cells.issuperset(obj1.cells),
            'same_row': not (obj1.bbox[2] < obj2.bbox[0] or obj2.bbox[2] < obj1.bbox[0]),
            'same_column': not (obj1.bbox[3] < obj2.bbox[1] or obj2.bbox[3] < obj1.bbox[1])
        }

    # Additional helper methods...
    def extract_transformation_pattern(self, input_grid: List[List[int]],
                                     output_grid: List[List[int]]) -> Dict[str, Any]:
        """Analyze transformation pattern between input and output grids"""
        input_analysis = self.process_arc_grid(input_grid)
        output_analysis = self.process_arc_grid(output_grid)

        transformation = {
            'color_changes': self._detect_color_changes(input_grid, output_grid),
            'object_changes': self._detect_object_changes(
                input_analysis['objects'], output_analysis['objects']
            ),
            'spatial_changes': self._detect_spatial_changes(
                input_analysis['relationships'], output_analysis['relationships']
            ),
            'pattern_type': self._classify_transformation_type(input_analysis, output_analysis)
        }

        return transformation
```

### 4. CEGIS Implementation (cegis/synthesizer.py)

```python
"""
CEGIS synthesizer implementation for ARC program synthesis
"""
from typing import List, Dict, Optional, Tuple, Any
import logging
from dataclasses import dataclass
from ..core.popper_interface import PopperInterface

logger = logging.getLogger(__name__)

@dataclass
class VerificationResult:
    """Result of program verification"""
    is_valid: bool
    failed_example: Optional[Tuple[Any, Any]] = None
    error_message: Optional[str] = None

class CEGISSynthesizer:
    """
    CEGIS-based synthesizer integrating with Popper ILP
    Implements counterexample-guided program synthesis
    """

    def __init__(self, config: Dict):
        self.config = config
        self.max_iterations = config.get('max_iterations', 50)
        self.synthesis_timeout = config.get('synthesis_timeout', 300)
        self.verification_timeout = config.get('verification_timeout', 60)

    def synthesize_from_examples(self, examples: List[Tuple],
                               background_knowledge: str,
                               bias_specification: str) -> Optional[str]:
        """
        Main CEGIS synthesis loop

        Args:
            examples: List of (input, output) example pairs
            background_knowledge: Domain knowledge as Prolog predicates
            bias_specification: Search space constraints

        Returns:
            Synthesized program or None if synthesis fails
        """
        counterexamples = []

        for iteration in range(self.max_iterations):
            logger.info(f"CEGIS iteration {iteration + 1}/{self.max_iterations}")

            # Generate candidate program
            candidate = self._generate_candidate(
                examples, background_knowledge, bias_specification, counterexamples
            )

            if candidate is None:
                logger.info("No candidate program found - search space exhausted")
                return None

            # Verify candidate against all examples
            verification_result = self._verify_candidate(candidate, examples)

            if verification_result.is_valid:
                logger.info(f"Valid program found in iteration {iteration + 1}")
                return candidate
            else:
                # Generate counterexample for next iteration
                counterexample = self._generate_counterexample(
                    candidate, verification_result.failed_example
                )
                counterexamples.append(counterexample)
                logger.info(f"Added counterexample: {counterexample}")

        logger.warning("CEGIS synthesis failed - maximum iterations reached")
        return None

    def _generate_candidate(self, examples: List[Tuple],
                          background_knowledge: str,
                          bias_specification: str,
                          counterexamples: List[str]) -> Optional[str]:
        """
        Generate candidate program using Popper ILP
        Incorporates counterexamples as additional constraints
        """
        # Create temporary Popper problem
        task_data = {
            'examples': self._format_examples_for_popper(examples),
            'background_knowledge': background_knowledge,
            'bias': self._augment_bias_with_counterexamples(bias_specification, counterexamples)
        }

        # TODO: Interface with Popper to generate candidate
        # This would involve creating temporary files and calling Popper
        # For now, return a placeholder
        return self._call_popper_synthesis(task_data)

    def _verify_candidate(self, candidate: str, examples: List[Tuple]) -> VerificationResult:
        """
        Verify candidate program against all examples
        Uses Prolog execution to check correctness
        """
        # TODO: Implement program verification
        # This would involve executing the Prolog program against examples
        # and checking if outputs match expected results

        for example_input, expected_output in examples:
            try:
                actual_output = self._execute_program(candidate, example_input)
                if not self._outputs_match(actual_output, expected_output):
                    return VerificationResult(
                        is_valid=False,
                        failed_example=(example_input, expected_output),
                        error_message=f"Output mismatch: expected {expected_output}, got {actual_output}"
                    )
            except Exception as e:
                return VerificationResult(
                    is_valid=False,
                    failed_example=(example_input, expected_output),
                    error_message=f"Execution error: {str(e)}"
                )

        return VerificationResult(is_valid=True)

    def _generate_counterexample(self, candidate: str, failed_example: Tuple) -> str:
        """
        Generate counterexample constraint from failed verification
        Returns constraint to add to bias for next iteration
        """
        example_input, expected_output = failed_example

        # Analyze why the program failed on this example
        # Generate constraint to exclude this type of failure

        # For ARC tasks, this might involve:
        # - Spatial relationship constraints
        # - Color transformation constraints
        # - Object manipulation constraints

        constraint = f":- program_produces_output({candidate}, {example_input}, Output), Output \\= {expected_output}."
        return constraint
```

### 5. Anti-Unification Module (core/anti_unification.py)

```python
"""
Anti-unification implementation for pattern generalization
"""
from typing import List, Dict, Tuple, Any, Optional, Set
import logging
from dataclasses import dataclass
from collections import defaultdict

logger = logging.getLogger(__name__)

@dataclass
class Term:
    """Represents a logical term for anti-unification"""
    functor: str
    args: List['Term']
    is_variable: bool = False

    def __hash__(self):
        if self.is_variable:
            return hash(self.functor)
        return hash((self.functor, tuple(self.args)))

    def __eq__(self, other):
        if not isinstance(other, Term):
            return False
        return (self.functor == other.functor and
                self.args == other.args and
                self.is_variable == other.is_variable)

class AntiUnifier:
    """
    Anti-unification implementation for generalizing programs
    Based on Plotkin's least general generalization algorithm
    """

    def __init__(self, config: Dict):
        self.config = config
        self.variable_counter = 0
        self.disagreement_store = {}

    def generalize_programs(self, programs: List[str]) -> str:
        """
        Generalize multiple programs using anti-unification

        Args:
            programs: List of program strings to generalize

        Returns:
            Generalized program pattern
        """
        if not programs:
            return ""

        if len(programs) == 1:
            return programs[0]

        # Parse programs to term representation
        term_programs = [self._parse_program_to_terms(prog) for prog in programs]

        # Perform pairwise anti-unification
        result = term_programs[0]
        for i in range(1, len(term_programs)):
            result = self._anti_unify_programs(result, term_programs[i])

        # Convert back to program string
        return self._terms_to_program_string(result)

    def generalize_program(self, program: str, examples: List[Tuple]) -> str:
        """
        Generalize a single program based on multiple examples
        Extracts patterns from how the program handles different inputs
        """
        # Analyze program behavior on different examples
        behavioral_patterns = []

        for example_input, example_output in examples:
            pattern = self._extract_behavioral_pattern(program, example_input, example_output)
            behavioral_patterns.append(pattern)

        # Generalize behavioral patterns
        if behavioral_patterns:
            generalized_pattern = self._generalize_patterns(behavioral_patterns)
            return self._pattern_to_program(generalized_pattern)

        return program

    def _anti_unify_programs(self, prog1: List[Term], prog2: List[Term]) -> List[Term]:
        """
        Anti-unify two programs represented as lists of terms
        Implements Plotkin's LGG algorithm
        """
        self.disagreement_store.clear()
        self.variable_counter = 0

        if len(prog1) != len(prog2):
            # Programs have different structure - create more general template
            return self._create_general_template(prog1, prog2)

        unified_clauses = []
        for term1, term2 in zip(prog1, prog2):
            unified_clause = self._anti_unify_terms(term1, term2)
            unified_clauses.append(unified_clause)

        return unified_clauses

    def _anti_unify_terms(self, term1: Term, term2: Term) -> Term:
        """
        Anti-unify two terms using Plotkin's algorithm

        Core anti-unification rules:
        1. If terms are identical, return the term
        2. If terms have same functor and arity, recursively anti-unify arguments
        3. Otherwise, introduce fresh variable
        """
        # Rule 1: Identical terms
        if term1 == term2:
            return term1

        # Rule 2: Same functor and arity
        if (not term1.is_variable and not term2.is_variable and
            term1.functor == term2.functor and len(term1.args) == len(term2.args)):

            unified_args = []
            for arg1, arg2 in zip(term1.args, term2.args):
                unified_arg = self._anti_unify_terms(arg1, arg2)
                unified_args.append(unified_arg)

            return Term(term1.functor, unified_args)

        # Rule 3: Introduce fresh variable
        disagreement_key = (term1, term2)
        if disagreement_key in self.disagreement_store:
            return self.disagreement_store[disagreement_key]

        fresh_var = Term(f"X{self.variable_counter}", [], is_variable=True)
        self.variable_counter += 1
        self.disagreement_store[disagreement_key] = fresh_var

        return fresh_var

    def _parse_program_to_terms(self, program: str) -> List[Term]:
        """
        Parse Prolog program string to internal term representation
        Simplified parser for demonstration - production version would be more robust
        """
        terms = []

        # Split program into clauses
        clauses = [clause.strip() for clause in program.split('.') if clause.strip()]

        for clause in clauses:
            # Parse each clause to term structure
            term = self._parse_clause_to_term(clause)
            if term:
                terms.append(term)

        return terms

    def _parse_clause_to_term(self, clause: str) -> Optional[Term]:
        """Parse a single Prolog clause to term representation"""
        clause = clause.strip()
        if not clause:
            return None

        # Handle rule vs fact
        if ':-' in clause:
            head, body = clause.split(':-', 1)
            head_term = self._parse_predicate(head.strip())
            body_terms = self._parse_body(body.strip())
            return Term(':-', [head_term, body_terms])
        else:
            # Fact
            return self._parse_predicate(clause)

    def _parse_predicate(self, pred_str: str) -> Term:
        """Parse predicate string to term"""
        pred_str = pred_str.strip()

        if '(' not in pred_str:
            # Atom with no arguments
            return Term(pred_str, [])

        # Extract functor and arguments
        paren_idx = pred_str.index('(')
        functor = pred_str[:paren_idx]
        args_str = pred_str[paren_idx+1:-1]  # Remove parentheses

        # Parse arguments (simplified - doesn't handle nested structures)
        args = []
        if args_str.strip():
            arg_strs = [arg.strip() for arg in args_str.split(',')]
            for arg_str in arg_strs:
                if arg_str.isalpha() and arg_str[0].isupper():
                    # Variable
                    args.append(Term(arg_str, [], is_variable=True))
                else:
                    # Atom/number
                    args.append(Term(arg_str, []))

        return Term(functor, args)

    def _terms_to_program_string(self, terms: List[Term]) -> str:
        """Convert term representation back to Prolog program string"""
        clauses = []

        for term in terms:
            clause_str = self._term_to_string(term)
            clauses.append(clause_str + '.')

        return '\\n'.join(clauses)

    def _term_to_string(self, term: Term) -> str:
        """Convert single term to string representation"""
        if term.is_variable:
            return term.functor

        if not term.args:
            return term.functor

        if term.functor == ':-':
            # Rule
            head_str = self._term_to_string(term.args[0])
            body_str = self._term_to_string(term.args[1])
            return f"{head_str} :- {body_str}"

        # Regular predicate
        args_str = ', '.join(self._term_to_string(arg) for arg in term.args)
        return f"{term.functor}({args_str})"
```

### 6. Usage Examples and Templates

```python
"""
Complete usage example and integration demonstration
"""
from pathlib import Path

# Example 1: Basic ARC task synthesis
def example_basic_synthesis():
    """Demonstrate basic synthesis on a simple ARC task"""

    # Initialize synthesis engine
    engine = ARCSynthesisEngine("config/default.yaml")

    # Load ARC task
    task = SynthesisTask(
        task_id="example_001",
        train_pairs=[
            # Simple color transformation task
            ([[0, 1, 0], [1, 1, 1], [0, 1, 0]],
             [[0, 2, 0], [2, 2, 2], [0, 2, 0]]),
            ([[0, 1, 1], [1, 0, 1], [1, 1, 0]],
             [[0, 2, 2], [2, 0, 2], [2, 2, 0]])
        ],
        test_pairs=[
            ([[1, 0, 1], [0, 1, 0], [1, 0, 1]],
             [[2, 0, 2], [0, 2, 0], [2, 0, 2]])
        ],
        metadata={"description": "Replace color 1 with color 2"}
    )

    # Run synthesis
    result = engine.synthesize_program(task)

    if result.success:
        print(f"Synthesis successful!")
        print(f"Program: {result.program}")
        print(f"Confidence: {result.confidence:.2f}")
        print(f"Time: {result.synthesis_time:.2f}s")
    else:
        print("Synthesis failed")

    return result

# Example 2: Complex spatial reasoning task
def example_spatial_synthesis():
    """Demonstrate synthesis on spatial reasoning task"""

    engine = ARCSynthesisEngine("config/spatial.yaml")

    # Task involving object movement
    task = SynthesisTask(
        task_id="spatial_001",
        train_pairs=[
            # Object translation pattern
            ([[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
             [[0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0]]),
            ([[0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
             [[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        ],
        test_pairs=[
            ([[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0]],
             [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0]])
        ],
        metadata={"description": "Move objects one position right and down"}
    )

    result = engine.synthesize_program(task)
    return result

# Example 3: Configuration templates
def create_config_templates():
    """Create configuration templates for different task types"""

    # Basic configuration
    basic_config = {
        'popper': {
            'popper_path': './popper',
            'timeout': 300,
            'solver': 'rc2',
            'max_vars': 6,
            'max_body': 8
        },
        'extraction': {
            'connectivity': 4,
            'min_object_size': 1,
            'background_color': 0
        },
        'cegis': {
            'max_iterations': 25,
            'synthesis_timeout': 300,
            'verification_timeout': 60
        },
        'anti_unification': {
            'max_generalization_depth': 5,
            'preserve_structure': True
        },
        'oracle': {
            'validation_method': 'exact_match',
            'tolerance': 0.0
        }
    }

    # Spatial reasoning configuration
    spatial_config = basic_config.copy()
    spatial_config['popper'].update({
        'max_vars': 10,
        'max_body': 12,
        'enable_recursion': False
    })
    spatial_config['extraction'].update({
        'connectivity': 8,
        'analyze_shapes': True,
        'detect_patterns': True
    })

    return basic_config, spatial_config

if __name__ == "__main__":
    # Run examples
    print("Running basic synthesis example...")
    basic_result = example_basic_synthesis()

    print("\\nRunning spatial synthesis example...")
    spatial_result = example_spatial_synthesis()

    # Create configuration files
    basic_config, spatial_config = create_config_templates()

    # Save configurations
    import yaml
    with open("config/basic.yaml", "w") as f:
        yaml.dump(basic_config, f, default_flow_style=False)

    with open("config/spatial.yaml", "w") as f:
        yaml.dump(spatial_config, f, default_flow_style=False)

    print("\\nConfiguration templates created in config/ directory")
```

## Best Practices and Extension Points

### Extension Guidelines

1. **Adding New Transformation Types**: Extend the `transformations.py` module with new spatial operations
2. **Custom Object Detection**: Implement domain-specific object extractors in `extraction/`
3. **Alternative Synthesis Methods**: Add new synthesizers in `cegis/` following the interface pattern
4. **Background Knowledge**: Extend Prolog predicates in `popper_files/background/`
5. **Evaluation Metrics**: Add custom metrics in `utils/metrics.py`

### Performance Optimization

1. **Caching**: Implement result caching for expensive operations
2. **Parallel Processing**: Use multiprocessing for independent synthesis tasks
3. **Incremental Learning**: Store and reuse learned patterns across tasks
4. **Memory Management**: Optimize data structures for large grids

### Integration Points

1. **External Solvers**: Interface with different SMT/SAT solvers
2. **Neural Components**: Add neural network modules for pattern recognition
3. **Visualization**: Implement grid visualization and program explanation
4. **Benchmarking**: Integration with ARC evaluation frameworks

This comprehensive framework provides a robust foundation for Popper-based program synthesis on ARC tasks, combining state-of-the-art techniques in ILP, object extraction, CEGIS, and anti-unification into a unified, extensible system.
