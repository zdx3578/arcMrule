"""
将多维度关系库系统集成到ARC分析流程中的示例
"""

def analyze_common_patterns_with_weights(self):
    """
    改进的版本：分析多对训练数据的共同模式，考虑权重因素，
    通过多维度关系库发现复杂的跨实例模式
    
    Returns:
        带权重的共同模式字典
    """
    if not self.mapping_rules:
        return {}

    # 1. 初始化多维度关系库系统
    from arc_relationship_libraries import ARCRelationshipLibraries
    relationship_libs = ARCRelationshipLibraries(debug=self.debug, debug_print=self._debug_print)
    
    # 2. 构建关系库
    relationship_libs.build_libraries_from_data(self.mapping_rules, self.all_objects)
    
    # 3. 查找跨数据对的模式
    cross_pair_patterns = relationship_libs.find_patterns_across_pairs()
    
    # 4. 调用原始的模式分析器获取基本模式
    basic_patterns = self.pattern_analyzer.analyze_common_patterns(self.mapping_rules)
    
    # 5. 结合基本模式和跨数据对模式
    combined_patterns = {
        "basic": basic_patterns,
        "cross_instance": cross_pair_patterns
    }
    
    # 6. 对所有模式进行权重计算和排序
    weighted_patterns = self._compute_pattern_weights(combined_patterns)
    
    # 7. 保存关系库状态到文件（可选）
    if self.debug:
        relationship_libs.export_libraries_to_json(f"{self.debug_dir}/relationship_libraries.json")
        self._debug_save_json(weighted_patterns, "weighted_patterns_from_libs")
    
    # 8. 返回带权重的模式
    return weighted_patterns

def _compute_pattern_weights(self, combined_patterns):
    """
    计算所有模式的权重并排序
    
    Args:
        combined_patterns: 组合的模式字典
        
    Returns:
        带权重的排序模式
    """
    # 提取所有模式到一个列表
    all_patterns = []
    
    # 处理基本模式
    basic = combined_patterns.get("basic", {})
    
    # 形状变换模式
    for pattern in basic.get("shape_transformations", []):
        all_patterns.append({
            "type": "shape_transformation",
            "source": "basic",
            "data": pattern,
            "confidence": pattern.get("confidence", 0.5),
            "raw_weight": 1.0
        })
    
    # 颜色映射模式
    for from_color, mapping in basic.get("color_mappings", {}).get("mappings", {}).items():
        all_patterns.append({
            "type": "color_mapping",
            "source": "basic",
            "data": {"from": from_color, "to": mapping.get("to_color")},
            "confidence": mapping.get("confidence", 0.5),
            "raw_weight": 1.0
        })
    
    # 位置变化模式
    for pattern in basic.get("position_changes", []):
        all_patterns.append({
            "type": "position_change",
            "source": "basic",
            "data": pattern,
            "confidence": pattern.get("confidence", 0.5),
            "raw_weight": 1.0
        })
    
    # 处理跨实例模式
    for pattern in combined_patterns.get("cross_instance", []):
        pattern_type = pattern.get("type", "unknown")
        subtype = pattern.get("subtype", "")
        
        all_patterns.append({
            "type": f"{pattern_type}_{subtype}" if subtype else pattern_type,
            "source": "cross_instance",
            "data": pattern,
            "confidence": pattern.get("confidence", 0.5),
            "raw_weight": pattern.get("weight", 1.0)
        })
    
    # 计算最终权重分数 (0.7 * confidence + 0.3 * raw_weight)
    for pattern in all_patterns:
        pattern["weight"] = 0.7 * pattern["confidence"] + 0.3 * pattern["raw_weight"]
    
    # 按权重排序
    all_patterns.sort(key=lambda x: x["weight"], reverse=True)
    
    # 构建最终结果
    result = {
        "patterns": all_patterns,
        "top_patterns": all_patterns[:min(10, len(all_patterns))],
        "total_patterns": len(all_patterns),
        "original": combined_patterns
    }
    
    return result