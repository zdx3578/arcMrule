
class PatternAnalysisMixin:

    def _analyze_underlying_pattern_for_addition(self, target_color, rule):
        """分析添加操作背后可能存在的模式

        Args:
            target_color: 被添加的目标颜色
            rule: 添加操作的规则

        Returns:
            如果找到可能的模式，返回模式信息；否则返回None
        """
        if not hasattr(self, 'task') or not self.task:
            return None  # 如果没有训练任务数据，无法分析

        # 收集所有支持该规则的训练示例
        supporting_examples = []
        for pair_idx in rule.get('supporting_pairs', []):
            if pair_idx < len(self.task['train']):
                supporting_examples.append(self.task['train'][pair_idx])

        if not supporting_examples:
            return None

        # 初始化模式分析结果
        pattern_candidates = {
            'four_box_pattern': {'instances': [], 'confidence': 0},
            'symmetry_pattern': {'instances': [], 'confidence': 0},
            'proximity_pattern': {'instances': [], 'confidence': 0},
            'alignment_pattern': {'instances': [], 'confidence': 0}
        }

        # 对每个示例，分析添加位置与可能的模式
        # for example in supporting_examples:
        for idx, example in enumerate(supporting_examples):
            print(f"Index {idx}, ")
            input_grid = example['input']
            output_grid = example['output']

            # 找出所有新添加的目标颜色位置
            added_positions = self._find_added_positions(input_grid, output_grid, target_color)
            if not added_positions:
                continue

            # 检查这些添加位置是否与特定模式相关联
            # 1. 检查4Box模式
            fourbox_instances = self._check_for_4box_patterns(input_grid, added_positions)
            if fourbox_instances:
                pattern_candidates['four_box_pattern']['instances'].extend(fourbox_instances)

            # # 2. 检查对称性模式#! 需要完善，暂时不用
            # symmetry_instances = self._check_for_symmetry_patterns(input_grid, added_positions)
            # if symmetry_instances:
            #     pattern_candidates['symmetry_pattern']['instances'].extend(symmetry_instances)

            # # 3. 检查邻近性模式
            # proximity_instances = self._check_for_proximity_patterns(input_grid, added_positions)
            # if proximity_instances:
            #     pattern_candidates['proximity_pattern']['instances'].extend(proximity_instances)

            # # 4. 检查对齐模式
            # alignment_instances = self._check_for_alignment_patterns(input_grid, added_positions)
            # if alignment_instances:
            #     pattern_candidates['alignment_pattern']['instances'].extend(alignment_instances)

        # 计算每种模式的置信度
        total_added_positions = sum(len(self._find_added_positions(ex['input'], ex['output'], target_color))
                                for ex in supporting_examples)

        for pattern_type, data in pattern_candidates.items():
            if total_added_positions > 0:
                data['confidence'] = len(data['instances']) / total_added_positions

        # 找出置信度最高的模式
        best_pattern = max(pattern_candidates.items(), key=lambda x: x[1]['confidence'])
        pattern_type, pattern_data = best_pattern

        # 仅当置信度足够高时才返回模式
        if pattern_data['confidence'] > 0.5 and pattern_data['instances']:
            return self._create_pattern_description(pattern_type, pattern_data, target_color)

        return None

    def _find_added_positions(self, input_grid, output_grid, color):
        """找出所有新添加的指定颜色的位置"""
        added_positions = []

        # 确保网格维度一致
        input_height = len(input_grid)
        input_width = len(input_grid[0]) if input_height > 0 else 0
        output_height = len(output_grid)
        output_width = len(output_grid[0]) if output_height > 0 else 0

        # 检查现有范围内的变化
        for y in range(min(output_height, input_height)):
            for x in range(min(output_width, input_width)):
                # 检查在输出中是指定颜色，但在输入中不是
                if output_grid[y][x] == color and input_grid[y][x] != color:
                    added_positions.append((x, y))

        # 检查输出比输入更大的区域
        for y in range(output_height):
            for x in range(output_width):
                if y >= input_height or x >= input_width:
                    if output_grid[y][x] == color:
                        added_positions.append((x, y))

        return added_positions

    def _check_for_4box_patterns0(self, grid, positions):
        """检查位置是否与4Box模式相关，只考虑添加对象的边界位置"""
        height = len(grid)
        width = len(grid[0]) if height > 0 else 0

        # 将位置列表转换为集合，便于快速查找
        positions_set = set(positions)

        # 找出所有边界位置
        boundary_positions = []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 上、下、左、右

        for x, y in positions:
            is_boundary = False
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                # 如果相邻位置不是添加位置(不在positions_set中)，
                # 或者在网格边界之外，那么当前位置是边界位置
                if (nx, ny) not in positions_set or nx < 0 or ny < 0 or nx >= width or ny >= height:
                    is_boundary = True
                    break

            if is_boundary:
                boundary_positions.append((x, y))

        if self.debug:
            self.debug_print(f"发现{len(positions)}个添加位置中有{len(boundary_positions)}个边界位置")

        # 只对边界位置进行4Box模式检查
        fourbox_instances = []

        for x, y in boundary_positions:
            surrounding_colors = {}

            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < width and 0 <= ny < height and (nx, ny) not in positions_set:
                    # 只检查非添加位置的周围颜色
                    color = grid[ny][nx]
                    if color not in surrounding_colors:
                        surrounding_colors[color] = 0
                    surrounding_colors[color] += 1

            # 如果有一种颜色出现在多个方向，可能是4Box模式
            for color, count in surrounding_colors.items():
                if count >= 2:  # 要求至少两个方向有相同颜色
                    fourbox_instances.append({
                        'center_position': (x, y),
                        'center_color': grid[y][x] if 0 <= y < height and 0 <= x < width else None,
                        'surrounding_color': color,
                        'surrounding_count': count,
                        'complete': count == 4  # 完全的4Box需要四个方向都有
                    })

        return fourbox_instances

    def _check_for_symmetry_patterns(self, grid, positions):
        """检查位置是否与对称性模式相关"""
        # 简化实现，检查位置是否形成对称图案
        height = len(grid)
        width = len(grid[0]) if height > 0 else 0

        # 检查中心对称
        center_x = width // 2
        center_y = height // 2

        symmetric_instances = []
        for x, y in positions:
            # 计算相对于中心的对称点
            sym_x = 2 * center_x - x
            sym_y = 2 * center_y - y

            # 检查对称点是否也在添加位置列表中
            if (sym_x, sym_y) in positions:
                symmetric_instances.append({
                    'position': (x, y),
                    'symmetric_position': (sym_x, sym_y),
                    'symmetry_type': 'central',
                    'center': (center_x, center_y)
                })

        return symmetric_instances

    def _check_for_proximity_patterns(self, grid, positions):
        """检查位置是否与邻近性模式相关"""
        height = len(grid)
        width = len(grid[0]) if height > 0 else 0

        proximity_instances = []

        for x, y in positions:
            # 检查周围是否有特定颜色的对象
            nearby_objects = {}
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < width and 0 <= ny < height and (dx != 0 or dy != 0):
                        color = grid[ny][nx]
                        if color != 0:  # 假设0是背景
                            if color not in nearby_objects:
                                nearby_objects[color] = []
                            nearby_objects[color].append((nx, ny, dx, dy))

            # 如果周围有对象，记录邻近性关系
            if nearby_objects:
                proximity_instances.append({
                    'position': (x, y),
                    'nearby_objects': nearby_objects
                })

        return proximity_instances

    def _check_for_alignment_patterns(self, grid, positions):
        """检查位置是否与对齐模式相关"""
        # 检查添加位置是否与现有对象在同一行或同一列
        height = len(grid)
        width = len(grid[0]) if height > 0 else 0

        alignment_instances = []

        # 对每个位置找出在同行或同列的非零值
        for x, y in positions:
            row_alignments = []
            col_alignments = []

            # 检查同行对象
            for nx in range(width):
                if nx != x and grid[y][nx] != 0:
                    row_alignments.append((nx, y, grid[y][nx]))

            # 检查同列对象
            for ny in range(height):
                if ny != y and grid[ny][x] != 0:
                    col_alignments.append((x, ny, grid[ny][x]))

            if row_alignments or col_alignments:
                alignment_instances.append({
                    'position': (x, y),
                    'row_alignments': row_alignments,
                    'col_alignments': col_alignments
                })

        return alignment_instances

    def _create_pattern_description(self, pattern_type, pattern_data, target_color):
        """根据模式类型和数据创建模式描述"""
        instances = pattern_data['instances']
        confidence = pattern_data['confidence']

        # 根据模式类型创建不同的描述
        if pattern_type == 'four_box_pattern':
            # 分析4Box模式的主要特征
            center_colors = {}
            surr_colors = {}

            for instance in instances:
                center_color = instance.get('center_color')
                if center_color is not None:
                    if center_color not in center_colors:
                        center_colors[center_color] = 0
                    center_colors[center_color] += 1

                surr_color = instance.get('surrounding_color')
                if surr_color is not None:
                    if surr_color not in surr_colors:
                        surr_colors[surr_color] = 0
                    surr_colors[surr_color] += 1

            # 找出最常见的中心颜色和环绕颜色
            main_center_color = max(center_colors.items(), key=lambda x: x[1])[0] if center_colors else None
            main_surr_color = max(surr_colors.items(), key=lambda x: x[1])[0] if surr_colors else None

            description = f"中心颜色{main_center_color}被颜色{main_surr_color}包围的4Box模式触发添加颜色{target_color}的对象"

            return {
                'pattern_type': '4Box模式',
                'center_color': main_center_color,
                'surrounding_color': main_surr_color,
                'confidence': confidence,
                'instances_count': len(instances),
                'description': description
            }

        elif pattern_type == 'symmetry_pattern':
            # 分析对称模式
            symm_types = {}
            for instance in instances:
                stype = instance.get('symmetry_type', 'unknown')
                if stype not in symm_types:
                    symm_types[stype] = 0
                symm_types[stype] += 1

            main_symm_type = max(symm_types.items(), key=lambda x: x[1])[0] if symm_types else 'unknown'

            return {
                'pattern_type': '对称模式',
                'symmetry_type': main_symm_type,
                'confidence': confidence,
                'instances_count': len(instances),
                'description': f"基于{main_symm_type}对称添加颜色{target_color}的对象"
            }

        elif pattern_type == 'proximity_pattern':
            # 分析邻近性模式
            nearby_colors = {}
            for instance in instances:
                for color, positions in instance.get('nearby_objects', {}).items():
                    if color not in nearby_colors:
                        nearby_colors[color] = 0
                    nearby_colors[color] += len(positions)

            main_nearby_color = max(nearby_colors.items(), key=lambda x: x[1])[0] if nearby_colors else None

            return {
                'pattern_type': '邻近模式',
                'nearby_color': main_nearby_color,
                'confidence': confidence,
                'instances_count': len(instances),
                'description': f"在颜色{main_nearby_color}对象附近添加颜色{target_color}的对象"
            }

        elif pattern_type == 'alignment_pattern':
            # 分析对齐模式
            row_alignment = sum(1 for i in instances if i.get('row_alignments', []))
            col_alignment = sum(1 for i in instances if i.get('col_alignments', []))

            alignment_type = "行对齐" if row_alignment > col_alignment else "列对齐"

            return {
                'pattern_type': '对齐模式',
                'alignment_type': alignment_type,
                'confidence': confidence,
                'instances_count': len(instances),
                'description': f"基于{alignment_type}添加颜色{target_color}的对象"
            }

        # 默认情况
        return {
            'pattern_type': pattern_type,
            'confidence': confidence,
            'instances_count': len(instances),
            'description': f"基于{pattern_type}添加颜色{target_color}的对象"
        }







    def _check_for_4box_patterns(self, grid, positions):
        """
        检查整体对象是否与4Box模式相关，将相连的添加像素视为一个整体对象
        """
        height = len(grid)
        width = len(grid[0]) if height > 0 else 0

        # 将位置列表转换为集合，便于快速查找
        positions_set = set(positions)

        # 标识连通区域（对象）
        objects = self._find_connected_objects(positions)

        if self.debug:
            self.debug_print(f"发现{len(positions)}个添加位置，组成了{len(objects)}个连通对象")

        fourbox_instances = []

        # 对每个对象分析其边界周围的颜色分布
        for obj_idx, obj_positions in enumerate(objects):
            obj_boundary = self._find_object_boundary(obj_positions, width, height)

            # 分析对象边界周围的颜色
            surrounding_colors_count = self._analyze_surrounding_colors(obj_boundary, positions_set, grid)

            # 检查是否存在主要的围绕颜色
            for color, direction_counts in surrounding_colors_count.items():
                # 计算这个颜色在不同方向出现的数量
                total_directions = sum(1 for count in direction_counts.values() if count > 0)

                # 如果一个颜色出现在至少2个方向，可能是4Box模式
                if total_directions >= 2:
                    # 计算对象的中心点（用于记录）
                    center_x = sum(x for x, y in obj_positions) / len(obj_positions)
                    center_y = sum(y for x, y in obj_positions) / len(obj_positions)
                    center_pos = (int(center_x), int(center_y))

                    # 查找对象内部的颜色（如果对象覆盖了原始网格上的区域）
                    original_colors = {}
                    for x, y in obj_positions:
                        if 0 <= x < width and 0 <= y < height:
                            orig_color = grid[y][x]
                            if orig_color not in original_colors:
                                original_colors[orig_color] = 0
                            original_colors[orig_color] += 1

                    # 确定对象的主要颜色
                    main_color = max(original_colors.items(), key=lambda x: x[1])[0] if original_colors else None

                    # 添加发现的实例
                    fourbox_instances.append({
                        'object_index': obj_idx,
                        'object_positions': list(obj_positions),
                        'center_position': center_pos,
                        'center_color': main_color,
                        'surrounding_color': color,
                        'direction_counts': direction_counts,
                        'total_directions': total_directions,
                        'complete': total_directions == 4,  # 完全的4Box需要四个方向都有
                        'surrounding_ratio': sum(direction_counts.values()) / len(obj_boundary)  # 包围程度
                    })

        return fourbox_instances

    def _find_connected_objects(self, positions):
        """
        将相连的像素分组为对象

        Args:
            positions: 添加位置的列表

        Returns:
            列表，每个元素是一组相连的位置坐标
        """
        if not positions:
            return []

        # 将位置列表转换为集合，便于快速查找
        remaining = set(positions)
        objects = []

        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # 上下左右四个方向

        while remaining:
            # 从剩余位置中取出一个作为起点
            start = next(iter(remaining))

            # 使用BFS查找所有相连的位置
            current_object = set()
            queue = [start]
            current_object.add(start)
            remaining.remove(start)

            while queue:
                x, y = queue.pop(0)

                # 检查四个方向的邻居
                for dx, dy in directions:
                    nx, ny = x + dx, y + dy
                    neighbor = (nx, ny)

                    # 如果邻居在剩余位置中，添加到当前对象
                    if neighbor in remaining:
                        queue.append(neighbor)
                        current_object.add(neighbor)
                        remaining.remove(neighbor)

            # 将当前对象添加到对象列表
            objects.append(current_object)

        return objects

    def _find_object_boundary(self, obj_positions, width, height):
        """
        找出对象的边界像素

        Args:
            obj_positions: 对象的位置集合
            width, height: 网格的宽度和高度

        Returns:
            对象边界位置的列表
        """
        obj_positions_set = set(obj_positions)
        boundary_positions = []

        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # 上下左右四个方向

        for x, y in obj_positions:
            # 检查是否是边界像素
            for dx, dy in directions:
                nx, ny = x + dx, y + dy

                # 如果邻居不在对象中，当前像素是边界
                if (nx, ny) not in obj_positions_set:
                    boundary_positions.append((x, y, dx, dy))  # 记录方向

        return boundary_positions

    def _analyze_surrounding_colors(self, boundary_positions, all_added_positions, grid):
        """
        分析对象边界周围的颜色分布

        Args:
            boundary_positions: 对象边界位置列表，每个元素是(x, y, dx, dy)
            all_added_positions: 所有添加位置的集合
            grid: 输入网格

        Returns:
            字典，键为颜色，值为该颜色在不同方向出现的次数
        """
        height = len(grid)
        width = len(grid[0]) if height > 0 else 0

        # 按颜色记录在不同方向出现的次数
        surrounding_colors_count = {}  # 颜色 -> {方向 -> 计数}

        # 方向映射
        direction_map = {
            (0, 1): 'top',
            (1, 0): 'right',
            (0, -1): 'bottom',
            (-1, 0): 'left'
        }

        for x, y, dx, dy in boundary_positions:
            nx, ny = x + dx, y + dy

            # 检查邻居是否在网格内且不是添加位置
            if 0 <= nx < width and 0 <= ny < height and (nx, ny) not in all_added_positions:
                color = grid[ny][nx]
                direction = direction_map.get((dx, dy), 'unknown')

                if color not in surrounding_colors_count:
                    surrounding_colors_count[color] = {
                        'top': 0,
                        'right': 0,
                        'bottom': 0,
                        'left': 0
                    }

                surrounding_colors_count[color][direction] += 1

        return surrounding_colors_count