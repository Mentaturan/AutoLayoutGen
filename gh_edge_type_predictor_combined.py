import numpy as np
import json
import os

# 使用相对路径，确保在Grasshopper环境中能正确找到文件
FIXED_DIR = "."

# 文件路径定义
NODE_ATTR_FILE = os.path.join(FIXED_DIR, "node_attr_edge_conn_4GEN.json")   # 节点属性和边缘连接文件
LLM_RESULT_FILE = os.path.join(FIXED_DIR, "llm_result.txt")                # LLM预测结果文件
LLM_PRED_FILE = os.path.join(FIXED_DIR, "llm_predictions.txt")             # 格式化的LLM预测结果文件
LLM_NPZ_FILE = os.path.join(FIXED_DIR, "llm_predictions.npz")              # 二进制格式的LLM预测结果文件


def calculate_edge_properties(source_x, source_y, target_x, target_y, layout_matrix):
    """
    计算边缘的方位属性和方向属性
    
    参数:
        source_x, source_y: 源节点坐标
        target_x, target_y: 目标节点坐标
        layout_matrix: 布局矩阵
    
    返回:
        edge_orientation: 边缘方位（上/下/左/右/内部）
        edge_direction: 边缘方向（水平/竖直）
    """
    # 计算边缘的方向（水平或竖直）
    if abs(source_x - target_x) > abs(source_y - target_y):
        edge_direction = "水平"
    else:
        edge_direction = "竖直"
    
    # 计算建筑的边界（找出所有非零值的行列范围）
    building_rows = set()
    building_cols = set()
    for row in range(layout_matrix.shape[0]):
        for col in range(layout_matrix.shape[1]):
            if layout_matrix[row, col] != 0:  # 非零值表示建筑内部
                building_rows.add(row)
                building_cols.add(col)
    
    # 确定建筑的边界坐标
    min_row = min(building_rows)
    max_row = max(building_rows)
    min_col = min(building_cols)
    max_col = max(building_cols)
    
    # 计算边缘的中点坐标
    mid_x = (source_x + target_x) / 2
    mid_y = (source_y + target_y) / 2
    
    # 将中点坐标转换为布局矩阵中的索引（每个单元格大小为20x20）
    mid_row = min(int(mid_y / 20), layout_matrix.shape[0] - 1)
    mid_col = min(int(mid_x / 20), layout_matrix.shape[1] - 1)
    
    # 确定边缘的方位（上/下/左/右/内部）
    edge_orientation = "内部"
    
    # 检查是否为建筑的上下左右边界
    if mid_row == min_row and edge_direction == "水平":
        edge_orientation = "上"
    elif mid_row == max_row and edge_direction == "水平":
        edge_orientation = "下"
    elif mid_col == min_col and edge_direction == "竖直":
        edge_orientation = "左"
    elif mid_col == max_col and edge_direction == "竖直":
        edge_orientation = "右"
    
    return edge_orientation, edge_direction


def classify_edges(node_attrs, edge_conn, layout_matrix=None):
    """
    分类边缘：识别需要预测的功能分区交界处边缘
    
    规则：
    1. 完全外部边缘：两个节点都在外部空地（layout_matrix中值为0）的边缘 - 自动标记为空白边缘
    2. 功能分区内部边缘：同一功能区块内的边缘 - 自动标记为空白边缘
    3. 建筑外墙边缘：连接内部区域和外部空地的边缘 - 需要LLM预测
    4. 功能分区交界处边缘：不同功能区块之间的边缘 - 需要LLM预测
    5. 正门边缘：连接外部空间和入口区域的边缘 - 需要LLM预测
    """
    need_pred_edges = []
    auto_label_edges = []
    
    for i in range(edge_conn.shape[1]):
        source = edge_conn[0, i]
        target = edge_conn[1, i]
        
        # 获取源节点和目标节点的坐标
        source_x, source_y = node_attrs[source][0], node_attrs[source][1]
        target_x, target_y = node_attrs[target][0], node_attrs[target][1]
        
        # 将坐标转换为layout_matrix中的索引
        source_row = min(int(source_y / 20), layout_matrix.shape[0] - 1)
        source_col = min(int(source_x / 20), layout_matrix.shape[1] - 1)
        target_row = min(int(target_y / 20), layout_matrix.shape[0] - 1)
        target_col = min(int(target_x / 20), layout_matrix.shape[1] - 1)
        
        # 获取源节点和目标节点的空间类型
        source_zone = layout_matrix[source_row, source_col]
        target_zone = layout_matrix[target_row, target_col]
        
        # 边缘类型分类
        is_both_external = (source_zone == 0 and target_zone == 0)  # 完全外部边缘
        is_same_internal = (source_zone != 0 and target_zone != 0 and source_zone == target_zone)  # 功能分区内部边缘
        is_need_prediction = not (is_both_external or is_same_internal)  # 需要预测的边缘
        
        if is_both_external or is_same_internal:
            # 自动标记的边缘
            auto_label_edges.append(i)
        else:
            # 需要预测的边缘：包括建筑外墙边缘、功能分区交界处边缘和正门边缘
            need_pred_edges.append(i)
    
    return need_pred_edges, auto_label_edges


def parse_llm_output(llm_output_file):
    """
    解析LLM输出，提取边缘类型预测
    
    参数:
        llm_output_file: LLM输出文件路径
    
    返回:
        predictions: 边缘预测结果列表，每个元素为(edge_id, edge_type)元组
    """
    try:
        with open(llm_output_file, 'r', encoding='utf-8') as f:
            llm_output = f.read().strip()
        
        predictions = []
        lines = llm_output.split('\n')
        
        # 过滤掉可能的额外内容，只保留边缘ID和预测类型
        for line in lines:
            line = line.strip()
            if line and ',' in line:
                try:
                    edge_id, edge_type = line.split(',')
                    edge_id = int(edge_id.strip())
                    edge_type = int(edge_type.strip())
                    predictions.append((edge_id, edge_type))
                except ValueError:
                    # 跳过格式不正确的行
                    continue
        
        # 按边缘ID排序
        predictions.sort(key=lambda x: x[0])
        
        print(f"成功解析 {len(predictions)} 个边缘的预测结果")
        return predictions
        
    except Exception as e:
        print(f"解析LLM输出时出错: {str(e)}")
        return None


def save_results(pred_labels, probs):
    """
    保存结果到文件，便于后续使用
    
    参数:
        pred_labels: 预测的边缘类型标签
        probs: 预测的边缘类型概率
    """
    # 保存为npz文件，便于加载
    np.savez(LLM_NPZ_FILE, pred_labels=pred_labels, probs=probs)
    print(f"预测结果已保存到: {LLM_NPZ_FILE}")
    
    # 保存为与原程序相同格式的文本文件
    with open(LLM_PRED_FILE, 'w') as f:
        f.write("# 边缘类型预测结果\n")
        f.write("# 格式: 边缘ID,预测类型\n")
        for i, pred in enumerate(pred_labels):
            f.write(f"{i},{pred}\n")
    print(f"预测结果已保存到: {LLM_PRED_FILE}")


def generate_edge_type_list():
    """
    生成edge_type_list，整合了LLM结果解析和边缘类型预测功能
    
    返回:
        edge_type_list: 边缘类型列表，每个元素为0-5之间的整数
    """
    print("=== 生成边缘类型列表 ===")
    
    # 1. 加载原始数据
    print("Loading data...")
    try:
        with open(NODE_ATTR_FILE, 'r') as f:
            data = json.load(f)
        
        node_attrs = np.array(data['node_attrs'])
        edge_conn = np.array(data['edge_conn'])
        print(f"加载到 {edge_conn.shape[1]} 个边缘连接")
        print(f"加载到 {len(node_attrs)} 个节点属性")
        
        # 加载布局矩阵从FPGGen_RoomLayoutMatrix.json
        layout_matrix_path = os.path.join(FIXED_DIR, "FPGGen_RoomLayoutMatrix.json")
        layout_matrix = None
        try:
            with open(layout_matrix_path, 'r') as f:
                layout_matrix = np.array(json.load(f))
            print(f"Successfully loaded layout matrix from: {layout_matrix_path}")
            print(f"Layout matrix shape: {layout_matrix.shape}")
        except Exception as e:
            print(f"Error occurred while reading layout matrix file: {str(e)}")
        
    except Exception as e:
        print(f"读取原始数据失败: {e}")
        return None
    
    # 2. 解析LLM输出
    print("Parsing LLM result...")
    predictions = parse_llm_output(LLM_RESULT_FILE)
    if predictions is None:
        print("解析LLM结果失败")
        return None
    
    # 3. 分类边缘
    need_pred_edges, auto_label_edges = classify_edges(node_attrs, edge_conn, layout_matrix)
    print(f"\n边缘分类结果:")
    print(f"自动标记的边缘数: {len(auto_label_edges)} (空白边缘)")
    print(f"需要预测的边缘数: {len(need_pred_edges)}")
    
    # 4. 直接使用LLM生成的结果，不进行额外分类
    num_edges = edge_conn.shape[1]
    
    # 生成pred_labels：与原始程序格式一致，形状为(num_edges,)
    # 初始化所有边缘为空白边缘
    pred_labels = np.zeros(num_edges, dtype=np.int64)
    
    # 生成probs：与原始程序格式一致，形状为(num_edges, 6)
    probs = np.zeros((num_edges, 6), dtype=np.float32)
    
    # 初始化为空白边缘(0)，这是所有边缘的默认值
    probs[:, 0] = 1.0
    
    # 5. 填充预测结果，应用LLM预测到所有需要预测的边缘
    # 包括建筑外围边缘、功能分区内部边缘和功能分区交界处边缘
    llm_pred_count = 0
    applied_edges = set()
    
    # 应用LLM预测中的所有类型
    applied_edge_types = {1, 2, 3, 4, 5}  # 应用所有类型
    
    # 应用LLM预测结果到所有需要预测的边缘
    for edge_id, edge_type in predictions:
        if 0 <= edge_id < num_edges:
            # 检查该边缘是否在需要预测的边缘列表中
            if edge_id in need_pred_edges:
                # 获取边缘的方位属性和方向属性
                source = edge_conn[0, edge_id]
                target = edge_conn[1, edge_id]
                source_x, source_y = node_attrs[source][0], node_attrs[source][1]
                target_x, target_y = node_attrs[target][0], node_attrs[target][1]
                edge_orientation, edge_direction = calculate_edge_properties(source_x, source_y, target_x, target_y, layout_matrix)
                
                # 确保建筑外围边缘正确标记为外墙
                # 上下方位的水平边缘和左右方位的竖直边缘应标记为外墙(3)
                if edge_orientation in ["上", "下"] and edge_direction == "水平":
                    # 上下方位的水平边缘应标记为外墙
                    final_edge_type = 3
                elif edge_orientation in ["左", "右"] and edge_direction == "竖直":
                    # 左右方位的竖直边缘应标记为外墙
                    final_edge_type = 3
                else:
                    # 其他边缘使用LLM预测结果
                    final_edge_type = edge_type
                
                # 应用最终确定的边缘类型
                pred_labels[edge_id] = final_edge_type
                # 重置概率分布，只将预测类型的概率设为1.0
                probs[edge_id] = np.zeros(6)
                probs[edge_id, final_edge_type] = 1.0
                llm_pred_count += 1
                applied_edges.add(edge_id)
        else:
            # 只警告超出范围的边缘ID
            print(f"警告: 边缘ID {edge_id} 超出范围 (0-{num_edges-1})")
    
    print(f"\n详细统计:")
    print(f"LLM预测应用到的边缘数: {llm_pred_count}")
    print(f"应用的边缘类型: {applied_edge_types}")
    
    # 6. 完全外部边缘处理
    # 确保完全外部边缘被正确标记为空白边缘(0)
    if layout_matrix is not None:
        for i in range(num_edges):
            source = edge_conn[0, i]
            target = edge_conn[1, i]
            
            # 获取源节点和目标节点的坐标
            source_x, source_y = node_attrs[source][0], node_attrs[source][1]
            target_x, target_y = node_attrs[target][0], node_attrs[target][1]
            
            # 将坐标转换为layout_matrix中的索引
            source_row = min(int(source_y / 20), layout_matrix.shape[0] - 1)
            source_col = min(int(source_x / 20), layout_matrix.shape[1] - 1)
            target_row = min(int(target_y / 20), layout_matrix.shape[0] - 1)
            target_col = min(int(target_x / 20), layout_matrix.shape[1] - 1)
            
            # 获取源节点和目标节点的空间类型
            source_zone = layout_matrix[source_row, source_col]
            target_zone = layout_matrix[target_row, target_col]
            
            # 完全外部边缘（两个节点都在0区域）：标记为空白边缘(0)
            if source_zone == 0 and target_zone == 0:
                pred_labels[i] = 0
                probs[i] = np.zeros(6)
                probs[i, 0] = 1.0
        
        print(f"\n完全外部边缘处理:")
        print(f"已将完全外部边缘标记为空白边缘(0)")
    
    # 7. 验证预测结果完整性
    print(f"\nLLM预测结果统计:")
    print(f"成功应用 {llm_pred_count} 个LLM预测结果")
    print(f"总预测结果: {len(predictions)}")
    
    # 8. 统计空白边缘数量，显示当前结果
    blank_edge_count = np.sum(pred_labels == 0)
    print(f"\n空白边缘统计:")
    print(f"总空白边缘数量: {blank_edge_count} ({blank_edge_count/num_edges*100:.1f}%)")
    
    # 9. 保存结果
    save_results(pred_labels, probs)
    
    # 8. 输出与原始程序相同的格式
    print(f"\n=== 最终结果 ===")
    print(f"Predictions shape: {pred_labels.shape}")
    print(f"Probabilities shape: {probs.shape}")
    
    # 9. 生成edge_type_list，与原始程序第267行一致
    edge_type_list = pred_labels.tolist()
    
    print(f"\nEdge type list (first 10 items): {edge_type_list[:10]}")
    print(f"Total edges: {len(edge_type_list)}")
    
    # 统计各边缘类型的数量
    edge_type_counts = {}
    for edge_type in edge_type_list:
        if edge_type not in edge_type_counts:
            edge_type_counts[edge_type] = 0
        edge_type_counts[edge_type] += 1
    
    print(f"\nEdge type distribution:")
    for edge_type in sorted(edge_type_counts.keys()):
        count = edge_type_counts[edge_type]
        print(f"  Edge type {edge_type}: {count} ({count/len(edge_type_list)*100:.1f}%)")
    
    return edge_type_list

# 主函数，直接生成edge_type_list变量供Grasshopper使用
print("=== Grasshopper边缘类型预测器（整合版） ===")
print(f"使用固定路径: {FIXED_DIR}")

# 生成edge_type_list，供Grasshopper直接使用
edge_type_list = generate_edge_type_list()

if edge_type_list is not None:
    print("\n=== 生成成功 ===")
    print("edge_type_list变量已生成，可直接在Grasshopper中使用")
else:
    print("\n=== 生成失败 ===")
    print("请检查日志信息，修复错误后重新运行")
print(edge_type_list)
