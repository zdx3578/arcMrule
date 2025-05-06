def analyze_common_patterns_with_weights(self):
    """
    分析多个训练数据对之间的共同模式，考虑权重因素，
    能够检测跨实例的复杂模式，例如基于被移除对象的形状来决定保留对象的颜色变化
    
    Returns:
        带权重的共同模式字典
    """
    if not self.mapping_rules or len(self.mapping_rules) < 2:
        if self.debug:
            self._debug_print("需要至少两个训练数据对来分析共同模式")
        return {"message": "需要至少两个训练数据对来分析共同模式", "patterns": []}

    if self.debug:
        self._debug_print(f"开始分析 {len(self.mapping_rules)} 个训练数据对的共同模式...")

    # 创建集合存储提取的各类模式
    all_patterns = {
        "basic_transformations": self.pattern_analyzer.analyze_common_patterns(self.mapping_rules),
        "conditional_patterns": self._extract_conditional_patterns(),
        "shape_based_patterns": self._extract_shape_based_patterns(),
        "cross_instance_patterns": self._extract_cross_instance_patterns(),
        "attribute_dependency_patterns": self._extract_attribute_dependency_patterns()
    }
    
    # 应用权重和置信度分析
    weighted_patterns = self._apply_weights_to_patterns(all_patterns)
    
    if self.debug:
        self._debug_save_json(weighted_patterns, "weighted_common_patterns")
        self._debug_print(f"找到 {len(weighted_patterns['prioritized_patterns'])} 个优先级排序的加权模式")
    
    return weighted_patterns

def _extract_conditional_patterns(self):
    """
    提取条件模式: 如果X条件发生，则Y结果发生
    例如: 如果颜色为红色的对象被移除，那么蓝色对象会变成绿色
    
    Returns:
        条件模式列表
    """
    conditional_patterns = []
    
    # 收集跨训练对的所有对象变化
    all_removals = []  # [(pair_id, object_info), ...]
    all_color_changes = []  # [(pair_id, from_obj, to_obj, from_color, to_color), ...]
    
    # 首先收集所有训练对的移除对象和颜色变化信息
    for rule in self.mapping_rules:
        pair_id = rule.get("pair_id")
        
        # 收集移除的对象
        if "input_to_output_transformation" in rule:
            transform_rule = rule["input_to_output_transformation"]
            
            # 分析移除的对象
            for removed in transform_rule.get("removed_objects", []):
                obj_info = removed.get("object", {})
                obj_id = removed.get("input_obj_id")
                weight = removed.get("weight", 1.0)
                
                # 获取移除对象的关键属性
                obj_shape_hash = None
                obj_color = None
                
                if obj_info and "obj_000" in obj_info:
                    obj_shape_hash = hash(tuple(map(tuple, obj_info.get("obj_000", []))))
                if obj_info and "main_color" in obj_info:
                    obj_color = obj_info["main_color"]
                
                if obj_shape_hash is not None:
                    all_removals.append({
                        "pair_id": pair_id,
                        "obj_id": obj_id,
                        "shape_hash": obj_shape_hash,
                        "color": obj_color,
                        "weight": weight
                    })
            
            # 分析颜色变化
            for modified in transform_rule.get("modified_objects", []):
                if "transformation" in modified and "color_transform" in modified["transformation"]:
                    color_transform = modified["transformation"]["color_transform"]
                    if "color_mapping" in color_transform:
                        for from_color, to_color in color_transform["color_mapping"].items():
                            all_color_changes.append({
                                "pair_id": pair_id,
                                "input_obj_id": modified.get("input_obj_id"),
                                "output_obj_id": modified.get("output_obj_id"),
                                "from_color": from_color,
                                "to_color": to_color,
                                "weight": modified.get("weight_product", 1.0)
                            })
    
    # 分析条件模式: 如果移除了特定形状/颜色的对象，其他对象的颜色会如何变化
    if all_removals and all_color_changes:
        # 按形状分组的移除对象
        removals_by_shape = {}
        for removal in all_removals:
            shape_hash = removal["shape_hash"]
            if shape_hash not in removals_by_shape:
                removals_by_shape[shape_hash] = []
            removals_by_shape[shape_hash].append(removal)
        
        # 按颜色变化分组
        changes_by_color = {}
        for change in all_color_changes:
            key = (change["from_color"], change["to_color"])
            if key not in changes_by_color:
                changes_by_color[key] = []
            changes_by_color[key].append(change)
        
        # 检查每种移除形状是否与特定颜色变化相关
        for shape_hash, removals in removals_by_shape.items():
            removal_pair_ids = {r["pair_id"] for r in removals}
            
            for color_key, changes in changes_by_color.items():
                from_color, to_color = color_key
                change_pair_ids = {c["pair_id"] for c in changes}
                
                # 计算移除与颜色变化的重叠率
                overlap = removal_pair_ids.intersection(change_pair_ids)
                if len(overlap) >= 2:  # 至少在两个训练对中同时出现
                    overlap_ratio = len(overlap) / min(len(removal_pair_ids), len(change_pair_ids))
                    
                    if overlap_ratio >= 0.7:  # 设置阈值，这里是70%的重叠率
                        # 找到可能的条件模式
                        pattern = {
                            "type": "conditional_pattern",
                            "subtype": "removal_color_change",
                            "condition": {
                                "type": "object_removal",
                                "shape_hash": shape_hash,
                                "shape_library_key": self._get_library_key_for_shape(shape_hash)
                            },
                            "result": {
                                "type": "color_change",
                                "from_color": from_color,
                                "to_color": to_color
                            },
                            "supporting_pairs": list(overlap),
                            "confidence": overlap_ratio,
                            "weight": self._calculate_avg_weight(
                                [r for r in removals if r["pair_id"] in overlap],
                                [c for c in changes if c["pair_id"] in overlap]
                            )
                        }
                        conditional_patterns.append(pattern)
    
    # 排序条件模式
    conditional_patterns.sort(key=lambda x: (x["confidence"], x["weight"]), reverse=True)
    
    return conditional_patterns

def _extract_shape_based_patterns(self):
    """
    提取基于形状的模式: 特定形状对象的变化规律
    例如: 所有三角形都变成圆形，所有红色对象都变成蓝色
    
    Returns:
        基于形状的模式列表
    """
    shape_patterns = []
    
    # 从形状库中获取所有唯一形状
    shapes = list(self.shape_library.keys())
    
    # 收集每个形状的对象如何变化
    shape_transformations = {}  # shape_hash -> [(pair_id, transformation_info), ...]
    
    for rule in self.mapping_rules:
        pair_id = rule.get("pair_id")
        
        if "input_to_output_transformation" not in rule:
            continue
            
        transform_rule = rule["input_to_output_transformation"]
        
        # 分析修改的对象
        for modified in transform_rule.get("modified_objects", []):
            if "transformation" not in modified:
                continue
                
            # 获取输入对象信息
            input_obj_id = modified.get("input_obj_id")
            input_obj = None
            
            # 寻找输入对象详情
            for obj in self.all_objects["input"]:
                if obj[0] == pair_id:  # 找到对应pair_id的输入对象列表
                    for obj_info in obj[1]:
                        if obj_info.obj_id == input_obj_id:
                            input_obj = obj_info
                            break
                    break
            
            if not input_obj:
                continue
                
            # 获取输入对象的形状哈希
            try:
                shape_hash = hash(tuple(map(tuple, input_obj.obj_000)))
            except (TypeError, AttributeError):
                continue
                
            # 记录变换信息
            transformation = modified["transformation"]
            transformation_info = {
                "input_obj_id": input_obj_id,
                "output_obj_id": modified.get("output_obj_id"),
                "transform_type": transformation.get("type"),
                "position_change": transformation.get("position_change"),
                "color_transform": transformation.get("color_transform"),
                "confidence": modified.get("confidence", 0.5),
                "weight": modified.get("weight_product", 1.0)
            }
            
            if shape_hash not in shape_transformations:
                shape_transformations[shape_hash] = []
            
            shape_transformations[shape_hash].append((pair_id, transformation_info))
    
    # 分析每个形状的一致性变化模式
    for shape_hash, transformations in shape_transformations.items():
        # 按变换类型分组
        by_transform_type = {}
        
        for pair_id, transform_info in transformations:
            transform_type = transform_info.get("transform_type", "unknown")
            
            if transform_type not in by_transform_type:
                by_transform_type[transform_type] = []
            
            by_transform_type[transform_type].append((pair_id, transform_info))
        
        # 检查每种变换类型的一致性
        for transform_type, type_transforms in by_transform_type.items():
            if len(type_transforms) >= 2:  # 至少在两个训练对中出现
                # 检查颜色变化的一致性
                color_changes = {}
                
                for _, transform_info in type_transforms:
                    if "color_transform" in transform_info and "color_mapping" in transform_info["color_transform"]:
                        for from_color, to_color in transform_info["color_transform"]["color_mapping"].items():
                            key = (from_color, to_color)
                            if key not in color_changes:
                                color_changes[key] = 0
                            color_changes[key] += 1
                
                # 找出最常见的颜色变化
                most_common_color_change = None
                max_count = 0
                
                for color_change, count in color_changes.items():
                    if count > max_count:
                        max_count = count
                        most_common_color_change = color_change
                
                # 创建形状变换模式
                pattern = {
                    "type": "shape_based_pattern",
                    "shape_hash": shape_hash,
                    "shape_library_key": self._get_library_key_for_shape(shape_hash),
                    "transform_type": transform_type,
                    "consistency": len(type_transforms) / len(transformations),
                    "supporting_pairs": list(set(pair_id for pair_id, _ in type_transforms)),
                    "count": len(type_transforms),
                    "total_count": len(transformations),
                    "confidence": len(type_transforms) / len(self.mapping_rules)
                }
                
                # 添加颜色变化信息（如果存在）
                if most_common_color_change and max_count >= 2:
                    from_color, to_color = most_common_color_change
                    pattern["color_change"] = {
                        "from_color": from_color,
                        "to_color": to_color,
                        "consistency": max_count / len(type_transforms)
                    }
                    
                    # 调整信心分数
                    pattern["confidence"] *= pattern["color_change"]["consistency"]
                
                # 计算平均权重
                total_weight = sum(t[1].get("weight", 1.0) for t in type_transforms)
                pattern["weight"] = total_weight / len(type_transforms)
                
                shape_patterns.append(pattern)
    
    # 排序形状模式
    shape_patterns.sort(key=lambda x: (x["confidence"], x["weight"]), reverse=True)
    
    return shape_patterns

def _extract_cross_instance_patterns(self):
    """
    提取跨实例模式: 分析不同训练对之间的关系模式
    例如: 如果训练对A中出现了形状X，那么训练对B中的颜色会发生特定变化
    
    Returns:
        跨实例模式列表
    """
    cross_patterns = []
    
    # 从所有训练对中构建特征矩阵
    # 特征包括：移除的对象形状、保留对象的形状、颜色变化等
    
    # 每个训练对的特征字典
    pair_features = {}
    
    for rule in self.mapping_rules:
        pair_id = rule.get("pair_id")
        
        if "input_to_output_transformation" not in rule:
            continue
        
        transform_rule = rule["input_to_output_transformation"]
        
        # 初始化特征
        features = {
            "removed_shapes": [],  # [(shape_hash, color, weight), ...]
            "added_shapes": [],    # [(shape_hash, color, weight), ...]
            "color_changes": [],   # [(from_color, to_color, weight), ...]
            "preserved_shapes": [] # [(shape_hash, color, weight), ...]
        }
        
        # 提取被移除的对象形状
        for removed in transform_rule.get("removed_objects", []):
            obj_info = removed.get("object", {})
            weight = removed.get("weight", 1.0)
            
            obj_shape_hash = None
            obj_color = None
            
            if obj_info:
                if "obj_000" in obj_info:
                    try:
                        obj_shape_hash = hash(tuple(map(tuple, obj_info.get("obj_000", []))))
                    except (TypeError, AttributeError):
                        continue
                        
                obj_color = obj_info.get("main_color")
            
            if obj_shape_hash is not None:
                features["removed_shapes"].append((obj_shape_hash, obj_color, weight))
        
        # 提取增加的对象形状
        for added in transform_rule.get("added_objects", []):
            obj_info = added.get("object", {})
            weight = added.get("weight", 1.0)
            
            obj_shape_hash = None
            obj_color = None
            
            if obj_info:
                if "obj_000" in obj_info:
                    try:
                        obj_shape_hash = hash(tuple(map(tuple, obj_info.get("obj_000", []))))
                    except (TypeError, AttributeError):
                        continue
                        
                obj_color = obj_info.get("main_color")
            
            if obj_shape_hash is not None:
                features["added_shapes"].append((obj_shape_hash, obj_color, weight))
        
        # 提取颜色变化
        for modified in transform_rule.get("modified_objects", []):
            if "transformation" in modified and "color_transform" in modified["transformation"]:
                color_transform = modified["transformation"]["color_transform"]
                weight = modified.get("weight_product", 1.0)
                
                if "color_mapping" in color_transform:
                    for from_color, to_color in color_transform["color_mapping"].items():
                        features["color_changes"].append((from_color, to_color, weight))
        
        # 提取保留的对象形状
        for preserved in transform_rule.get("preserved_objects", []):
            obj_info = preserved.get("object", {})
            weight = preserved.get("weight", 1.0)
            
            obj_shape_hash = None
            obj_color = None
            
            if obj_info:
                if "obj_000" in obj_info:
                    try:
                        obj_shape_hash = hash(tuple(map(tuple, obj_info.get("obj_000", []))))
                    except (TypeError, AttributeError):
                        continue
                        
                obj_color = obj_info.get("main_color")
            
            if obj_shape_hash is not None:
                features["preserved_shapes"].append((obj_shape_hash, obj_color, weight))
        
        pair_features[pair_id] = features
    
    # 分析训练对之间的关系模式
    if len(pair_features) >= 2:
        # 1. 分析移除对象形状与颜色变化的关系
        # 特别是当训练对A中移除了形状X，训练对B中的颜色变化为Y的情况
        
        # 按移除形状分组
        removed_by_shape = {}
        
        for pair_id, features in pair_features.items():
            for shape_hash, color, weight in features["removed_shapes"]:
                if shape_hash not in removed_by_shape:
                    removed_by_shape[shape_hash] = []
                removed_by_shape[shape_hash].append((pair_id, color, weight))
        
        # 按颜色变化分组
        changes_by_color = {}
        
        for pair_id, features in pair_features.items():
            for from_color, to_color, weight in features["color_changes"]:
                key = (from_color, to_color)
                if key not in changes_by_color:
                    changes_by_color[key] = []
                changes_by_color[key].append((pair_id, weight))
        
        # 检查移除形状与颜色变化之间的关系
        for shape_hash, removals in removed_by_shape.items():
            removal_pair_ids = {r[0] for r in removals}
            
            for color_key, changes in changes_by_color.items():
                from_color, to_color = color_key
                change_pair_ids = {c[0] for c in changes}
                
                # 计算形状移除与颜色变化的关联度
                if removal_pair_ids and change_pair_ids:
                    # 直接关联：相同训练对中同时发生
                    direct_overlap = removal_pair_ids.intersection(change_pair_ids)
                    
                    if len(direct_overlap) >= 2:
                        # 创建直接关联模式
                        pattern = {
                            "type": "cross_instance_pattern",
                            "subtype": "direct_removal_color_association",
                            "shape_hash": shape_hash,
                            "shape_library_key": self._get_library_key_for_shape(shape_hash),
                            "color_change": {
                                "from_color": from_color, 
                                "to_color": to_color
                            },
                            "supporting_pairs": list(direct_overlap),
                            "confidence": len(direct_overlap) / min(len(removal_pair_ids), len(change_pair_ids)),
                            "description": f"当形状{shape_hash}被移除时，颜色从{from_color}变为{to_color}"
                        }
                        
                        # 计算平均权重
                        shape_weights = [w for p, _, w in removals if p in direct_overlap]
                        color_weights = [w for p, w in changes if p in direct_overlap]
                        
                        if shape_weights and color_weights:
                            avg_shape_weight = sum(shape_weights) / len(shape_weights)
                            avg_color_weight = sum(color_weights) / len(color_weights)
                            pattern["weight"] = (avg_shape_weight + avg_color_weight) / 2
                        
                        cross_patterns.append(pattern)
                    
                    # 2. 分析更复杂的关系：如果形状X被移除，那么在其他实例中会有特定的颜色变化
                    non_overlap_removal = removal_pair_ids - change_pair_ids
                    non_overlap_change = change_pair_ids - removal_pair_ids
                    
                    if non_overlap_removal and non_overlap_change and len(non_overlap_removal) + len(non_overlap_change) >= 2:
                        # 尝试寻找一个模式，其中某些实例中的形状移除可以预测其他实例中的颜色变化
                        pattern = {
                            "type": "cross_instance_pattern",
                            "subtype": "complex_shape_color_association",
                            "shape_hash": shape_hash,
                            "shape_library_key": self._get_library_key_for_shape(shape_hash),
                            "color_change": {
                                "from_color": from_color,
                                "to_color": to_color
                            },
                            "removal_pairs": list(non_overlap_removal),
                            "change_pairs": list(non_overlap_change),
                            "confidence": 0.5,  # 初始置信度较低，因为这是复杂关系
                            "description": f"当一些实例中形状{shape_hash}被移除时，其他实例中颜色从{from_color}变为{to_color}"
                        }
                        
                        cross_patterns.append(pattern)
    
    # 排序跨实例模式
    cross_patterns.sort(key=lambda x: x.get("confidence", 0), reverse=True)
    
    return cross_patterns

def _extract_attribute_dependency_patterns(self):
    """
    提取属性依赖模式: 分析对象属性之间的依赖关系
    例如: 对象的颜色如何依赖于其形状，或对象的位置如何依赖于其大小
    
    Returns:
        属性依赖模式列表
    """
    dependency_patterns = []
    
    # 收集所有对象的属性关系
    object_attributes = []  # [(pair_id, obj_id, {attributes}), ...]
    
    # 从所有训练对中收集对象属性
    for rule in self.mapping_rules:
        pair_id = rule.get("pair_id")
        
        # 收集输入对象属性
        for pair_data in self.all_objects["input"]:
            if pair_data[0] == pair_id:
                for obj_info in pair_data[1]:
                    attrs = {
                        "shape_hash": hash(tuple(map(tuple, obj_info.obj_000))) if hasattr(obj_info, "obj_000") else None,
                        "color": obj_info.main_color if hasattr(obj_info, "main_color") else None,
                        "size": obj_info.size if hasattr(obj_info, "size") else None,
                        "position": (
                            obj_info.top if hasattr(obj_info, "top") else None,
                            obj_info.left if hasattr(obj_info, "left") else None
                        ),
                        "weight": obj_info.obj_weight if hasattr(obj_info, "obj_weight") else 1.0
                    }
                    object_attributes.append((pair_id, obj_info.obj_id, "input", attrs))
        
        # 收集输出对象属性
        for pair_data in self.all_objects["output"]:
            if pair_data[0] == pair_id:
                for obj_info in pair_data[1]:
                    attrs = {
                        "shape_hash": hash(tuple(map(tuple, obj_info.obj_000))) if hasattr(obj_info, "obj_000") else None,
                        "color": obj_info.main_color if hasattr(obj_info, "main_color") else None,
                        "size": obj_info.size if hasattr(obj_info, "size") else None,
                        "position": (
                            obj_info.top if hasattr(obj_info, "top") else None,
                            obj_info.left if hasattr(obj_info, "left") else None
                        ),
                        "weight": obj_info.obj_weight if hasattr(obj_info, "obj_weight") else 1.0
                    }
                    object_attributes.append((pair_id, obj_info.obj_id, "output", attrs))
    
    # 分析属性间的依赖关系 
    if object_attributes:
        # 1. 首先分析同一训练对中，输入对象的形状如何影响输出对象的颜色
        
        # 按训练对分组
        by_pair_id = {}
        for pair_id, obj_id, io_type, attrs in object_attributes:
            if pair_id not in by_pair_id:
                by_pair_id[pair_id] = {"input": [], "output": []}
            by_pair_id[pair_id][io_type].append((obj_id, attrs))
        
        # 分析每个训练对
        shape_to_color_patterns = {}  # (shape_hash, from_color, to_color) -> [(pair_id, weight), ...]
        
        for pair_id, io_data in by_pair_id.items():
            input_objs = io_data["input"]
            output_objs = io_data["output"]
            
            # 分析每个输入和输出对象的关系
            for in_obj_id, in_attrs in input_objs:
                in_shape = in_attrs["shape_hash"]
                in_color = in_attrs["color"]
                
                for out_obj_id, out_attrs in output_objs:
                    out_shape = out_attrs["shape_hash"]
                    out_color = out_attrs["color"]
                    
                    # 找出输入和输出相似形状但颜色改变的对象
                    if in_shape == out_shape and in_color != out_color:
                        key = (in_shape, in_color, out_color)
                        
                        if key not in shape_to_color_patterns:
                            shape_to_color_patterns[key] = []
                        
                        # 使用输入对象的权重
                        shape_to_color_patterns[key].append((pair_id, in_attrs["weight"]))
        
        # 过滤出重复出现的模式
        for key, occurrences in shape_to_color_patterns.items():
            if len(occurrences) >= 2:  # 至少在两个训练对中出现
                shape_hash, from_color, to_color = key
                
                # 创建形状-颜色依赖模式
                pattern = {
                    "type": "attribute_dependency",
                    "subtype": "shape_determines_color_change",
                    "shape_hash": shape_hash,
                    "shape_library_key": self._get_library_key_for_shape(shape_hash),
                    "from_color": from_color,
                    "to_color": to_color,
                    "supporting_pairs": [p for p, _ in occurrences],
                    "confidence": len(occurrences) / len(by_pair_id),
                    "description": f"形状{shape_hash}的对象从颜色{from_color}变为{to_color}"
                }
                
                # 计算平均权重
                weights = [w for _, w in occurrences]
                if weights:
                    pattern["weight"] = sum(weights) / len(weights)
                
                dependency_patterns.append(pattern)
        
        # 2. 分析移除对象的形状如何影响剩余对象的颜色变化
        
        # 收集每个训练对中移除的形状
        removed_shapes_by_pair = {}  # pair_id -> [(shape_hash, color, weight), ...]
        
        for rule in self.mapping_rules:
            pair_id = rule.get("pair_id")
            
            if "input_to_output_transformation" not in rule:
                continue
                
            transform_rule = rule["input_to_output_transformation"]
            removed_shapes = []
            
            for removed in transform_rule.get("removed_objects", []):
                obj_info = removed.get("object", {})
                weight = removed.get("weight", 1.0)
                
                obj_shape_hash = None
                obj_color = None
                
                if obj_info:
                    if "obj_000" in obj_info:
                        try:
                            obj_shape_hash = hash(tuple(map(tuple, obj_info.get("obj_000", []))))
                        except (TypeError, AttributeError):
                            continue
                            
                    obj_color = obj_info.get("main_color")
                
                if obj_shape_hash is not None:
                    removed_shapes.append((obj_shape_hash, obj_color, weight))
            
            removed_shapes_by_pair[pair_id] = removed_shapes
        
        # 收集每个训练对中的颜色变化
        color_changes_by_pair = {}  # pair_id -> [(from_color, to_color, weight), ...]
        
        for rule in self.mapping_rules:
            pair_id = rule.get("pair_id")
            
            if "input_to_output_transformation" not in rule:
                continue
                
            transform_rule = rule["input_to_output_transformation"]
            color_changes = []
            
            for modified in transform_rule.get("modified_objects", []):
                if "transformation" in modified and "color_transform" in modified["transformation"]:
                    color_transform = modified["transformation"]["color_transform"]
                    weight = modified.get("weight_product", 1.0)
                    
                    if "color_mapping" in color_transform:
                        for from_color, to_color in color_transform["color_mapping"].items():
                            color_changes.append((from_color, to_color, weight))
            
            color_changes_by_pair[pair_id] = color_changes
        
        # 分析移除形状与颜色变化之间的关系
        removal_color_patterns = {}  # (removed_shape, from_color, to_color) -> [(pair_id, weight), ...]
        
        for pair_id, removed_shapes in removed_shapes_by_pair.items():
            if pair_id not in color_changes_by_pair or not removed_shapes or not color_changes_by_pair[pair_id]:
                continue
                
            # 对于该训练对中的每个形状移除，检查与颜色变化的关系
            for shape_hash, _, shape_weight in removed_shapes:
                for from_color, to_color, color_weight in color_changes_by_pair[pair_id]:
                    key = (shape_hash, from_color, to_color)
                    
                    if key not in removal_color_patterns:
                        removal_color_patterns[key] = []
                    
                    # 权重使用形状权重和颜色变化权重的平均值
                    avg_weight = (shape_weight + color_weight) / 2
                    removal_color_patterns[key].append((pair_id, avg_weight))
        
        # 过滤出重复出现的模式
        for key, occurrences in removal_color_patterns.items():
            if len(occurrences) >= 2:  # 至少在两个训练对中出现
                shape_hash, from_color, to_color = key
                
                # 创建移除形状影响颜色变化的模式
                pattern = {
                    "type": "attribute_dependency",
                    "subtype": "removal_shape_influences_color_change",
                    "removed_shape_hash": shape_hash,
                    "shape_library_key": self._get_library_key_for_shape(shape_hash),
                    "color_change": {
                        "from_color": from_color,
                        "to_color": to_color
                    },
                    "supporting_pairs": [p for p, _ in occurrences],
                    "confidence": len(occurrences) / len(by_pair_id),
                    "description": f"当形状{shape_hash}的对象被移除时，某些对象的颜色从{from_color}变为{to_color}"
                }
                
                # 计算平均权重
                weights = [w for _, w in occurrences]
                if weights:
                    pattern["weight"] = sum(weights) / len(weights)
                
                dependency_patterns.append(pattern)
    
    # 排序属性依赖模式
    dependency_patterns.sort(key=lambda x: (x.get("confidence", 0), x.get("weight", 0)), reverse=True)
    
    return dependency_patterns

def _apply_weights_to_patterns(self, all_patterns):
    """
    将权重应用到所有检测到的模式，并按优先级排序

    Args:
        all_patterns: 所有类型的模式字典

    Returns:
        加权并排序的模式字典
    """
    # 合并所有模式
    combined_patterns = []
    
    # 添加基本变换模式
    basic_transformations = all_patterns.get("basic_transformations", {})
    
    # 形状变换
    for pattern in basic_transformations.get("shape_transformations", []):
        combined_patterns.append({
            "source": "basic_transform",
            "type": "shape_transformation",
            "data": pattern,
            "confidence": pattern.get("confidence", 0.5),
            "weight": pattern.get("weight", 1.0)
        })
    
    # 颜色映射
    for from_color, mapping in basic_transformations.get("color_mappings", {}).get("mappings", {}).items():
        combined_patterns.append({
            "source": "basic_transform",
            "type": "color_mapping",
            "data": {
                "from_color": from_color,
                "to_color": mapping.get("to_color"),
                "confidence": mapping.get("confidence", 0.5)
            },
            "confidence": mapping.get("confidence", 0.5),
            "weight": mapping.get("weight", 1.0)
        })
    
    # 位置变化
    for pattern in basic_transformations.get("position_changes", []):
        combined_patterns.append({
            "source": "basic_transform",
            "type": "position_change",
            "data": pattern,
            "confidence": pattern.get("confidence", 0.5),
            "weight": pattern.get("weight", 1.0)
        })
    
    # 添加条件模式
    for pattern in all_patterns.get("conditional_patterns", []):
        combined_patterns.append({
            "source": "conditional",
            "type": pattern.get("subtype", "conditional_pattern"),
            "data": pattern,
            "confidence": pattern.get("confidence", 0.5),
            "weight": pattern.get("weight", 1.0)
        })
    
    # 添加形状基础模式
    for pattern in all_patterns.get("shape_based_patterns", []):
        combined_patterns.append({
            "source": "shape_based",
            "type": "shape_transformation_pattern",
            "data": pattern,
            "confidence": pattern.get("confidence", 0.5),
            "weight": pattern.get("weight", 1.0)
        })
    
    # 添加跨实例模式
    for pattern in all_patterns.get("cross_instance_patterns", []):
        combined_patterns.append({
            "source": "cross_instance",
            "type": pattern.get("subtype", "cross_instance_pattern"),
            "data": pattern,
            "confidence": pattern.get("confidence", 0.5),
            "weight": pattern.get("weight", 1.0) if "weight" in pattern else 1.0
        })
    
    # 添加属性依赖模式
    for pattern in all_patterns.get("attribute_dependency_patterns", []):
        combined_patterns.append({
            "source": "attribute_dependency",
            "type": pattern.get("subtype", "attribute_dependency"),
            "data": pattern,
            "confidence": pattern.get("confidence", 0.5),
            "weight": pattern.get("weight", 1.0) if "weight" in pattern else 1.0
        })
    
    # 计算最终优先级得分 (confidence * 0.7 + weight * 0.3)
    for pattern in combined_patterns:
        pattern["priority_score"] = (
            pattern.get("confidence", 0.5) * 0.7 + 
            pattern.get("weight", 1.0) * 0.3
        )
    
    # 根据优先级得分排序
    combined_patterns.sort(key=lambda x: x["priority_score"], reverse=True)
    
    # 构建最终结果
    result = {
        "original_patterns": all_patterns,
        "prioritized_patterns": combined_patterns,
        "top_patterns": combined_patterns[:min(10, len(combined_patterns))]
    }
    
    return result

def _calculate_avg_weight(self, removals, changes):
    """计算平均权重"""
    removal_weights = [r.get("weight", 1.0) for r in removals]
    change_weights = [c.get("weight", 1.0) for c in changes]
    
    if not removal_weights and not change_weights:
        return 1.0
    
    total_weights = removal_weights + change_weights
    return sum(total_weights) / len(total_weights)

def _get_library_key_for_shape(self, shape_hash):
    """从形状库中获取形状的键"""
    for key, shape_info in self.shape_library.items():
        try:
            lib_shape_hash = hash(tuple(map(tuple, shape_info.get("normalized_shape", []))))
            if lib_shape_hash == shape_hash:
                return key
        except (TypeError, AttributeError):
            continue
    
    return None