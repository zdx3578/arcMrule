
def analyze_added_object_positions(input_grid, output_grid, color=4):
    """分析被添加的特定颜色对象的位置模式"""
    # 找出所有在output中但不在input中的颜色为4的位置
    added_positions = []
    for y in range(len(output_grid)):
        for x in range(len(output_grid[0])):
            if output_grid[y][x] == color:
                # 检查此位置在input中是否为其他颜色或超出边界
                if (y >= len(input_grid) or x >= len(input_grid[0]) or
                    input_grid[y][x] != color):
                    added_positions.append((x, y))

    # 分析位置模式
    position_patterns = {
        "corners": check_if_corners(added_positions, output_grid),
        "edges": check_if_edges(added_positions, output_grid),
        "center": check_if_center(added_positions, output_grid),
        "relative_to_other_colors": find_relative_positions(added_positions, output_grid)
    }

    return position_patterns

def analyze_added_object_shapes(input_grid, output_grid, color=4):
    """分析被添加的特定颜色对象的形状特征"""
    # 提取输出中的新增形状
    added_shapes = extract_shapes_of_color(output_grid, color)
    input_shapes = extract_shapes_of_color(input_grid, color)

    # 找出仅在输出中存在的形状
    new_shapes = [s for s in added_shapes if not any(shape_similarity(s, is_) > 0.8 for is_ in input_shapes)]

    # 分析形状特征
    shape_analysis = {
        "fixed_shape": all_shapes_similar(new_shapes),
        "derived_from_input": check_shape_derivation(new_shapes, input_grid),
        "symmetry": check_shape_symmetry(new_shapes),
        "common_pattern": extract_shape_pattern(new_shapes)
    }

    return shape_analysis

def analyze_addition_conditions(input_grid, output_grid, color=4):
    """分析对象添加的条件规则"""
    # 提取输入特征
    input_features = extract_grid_features(input_grid)

    # 检查条件关联性
    condition_analysis = {
        "related_to_colors": check_color_dependencies(input_grid, output_grid, color),
        "related_to_counts": check_count_dependencies(input_grid, output_grid, color),
        "related_to_layout": check_layout_dependencies(input_grid, output_grid, color),
        "related_to_symmetry": check_symmetry_dependencies(input_grid, output_grid, color)
    }

    return condition_analysis

def generate_detailed_addition_rule(position_patterns, shape_analysis, condition_analysis):
    """生成详细的添加规则描述"""
    rule_details = {}

    # 确定最可能的位置模式
    most_likely_position = max(position_patterns.items(), key=lambda x: x[1]['confidence'])[0]
    rule_details['position_pattern'] = most_likely_position

    # 确定最可能的形状来源
    if shape_analysis['derived_from_input']['confidence'] > 0.7:
        rule_details['shape_origin'] = 'derived_from_input'
        rule_details['shape_transformation'] = shape_analysis['derived_from_input']['transformation']
    else:
        rule_details['shape_origin'] = 'fixed_pattern'
        rule_details['shape_description'] = shape_analysis['common_pattern']

    # 确定最可能的触发条件
    for cond_type, cond_data in condition_analysis.items():
        if cond_data['confidence'] > 0.8:
            rule_details['condition_type'] = cond_type
            rule_details['condition_details'] = cond_data['details']
            break

    return rule_details
