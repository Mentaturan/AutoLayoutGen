import json
import numpy as np
import os

# 全局输入文件路径，使用相对路径
FIXED_DIR = "."
INPUT_FILE_PATH = os.path.join(FIXED_DIR, "node_attr_edge_conn_4GEN.json")

# 布局矩阵编码体系：将数字编码映射为中文房间类型名称
SPACE_TYPE_MAP = {
    0: "外部空地",      # 建筑外部区域
    1: "客厅",           # 主要起居空间
    2: "主卧",           # 主卧室
    3: "厨房",           # 烹饪空间
    4: "卫生间",         # 卫浴空间
    5: "餐厅",           # 用餐空间
    6: "儿童房",          # 儿童卧室
    7: "书房",           # 工作学习空间
    8: "次卧",           # 次卧室
    9: "客房",           # 客人卧室
    10: "阳台/庭院",       # 户外延伸空间
    11: "入口区域",         # 玄关/门厅
    12: "储藏室",          # 储物收纳空间
    13: "正门"            # 入户门位置
}

# 边缘类型定义：详细解释每种边缘类型的含义和用途
EDGE_TYPE_MAP = {
    0: "空白边缘 - 无任何建筑构件的边缘，通常为建筑外的空地、未封闭的开口",
    1: "正门边缘 - 入户正门所在的边缘，需与外部空间连通",
    2: "室内门边缘 - 房间与房间/走廊之间的室内门所在边缘",
    3: "实体墙边缘 - 实心墙体的边缘，无门窗开口",
    4: "幕墙边缘 - 玻璃幕墙的边缘，通常为大面积采光的非承重墙体",
    5: "矮墙边缘 - 高度低于常规墙体的边缘，用于阳台护栏、空间半隔断等"
}

def load_node_attributes_and_edge_connections(file_path):
    """
    从JSON文件加载节点属性和边缘连接数据，并从FPGGen_RoomLayoutMatrix.json加载布局矩阵
    
    参数:
        file_path: 包含节点属性和边缘连接的JSON文件路径
        
    返回:
        node_attrs: 节点属性数组
        edge_conn: 边缘连接数组，形状为(2, num_edges)
        layout_matrix: 布局矩阵数组，或None（加载失败时）
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        print(f"Successfully loaded file: {file_path}")
        print(f"Keys in file: {list(data.keys())}")

        node_attrs = np.array(data['node_attrs'])
        print(f"Node attributes shape: {node_attrs.shape}")
        
        source_nodes = data['edge_conn'][0]
        target_nodes = data['edge_conn'][1]
        edge_conn = np.array([source_nodes, target_nodes])
        print(f"Edge connections shape: {edge_conn.shape}")
        
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
            
        return node_attrs, edge_conn, layout_matrix
        
    except Exception as e:
        print(f"Error occurred while reading file: {str(e)}")
        return None, None, None

def classify_edges(node_attrs, edge_conn, layout_matrix=None):
    """
    分类边缘：识别需要预测的功能分区交界处边缘
    
    参数:
        node_attrs: 节点属性数组
        edge_conn: 边缘连接数组
        layout_matrix: 布局矩阵数组（可选）
    
    返回:
        need_pred_edges: 需要LLM预测的边缘索引列表
        auto_label_edges: 可自动标记的边缘索引列表
    
    分类规则：
    1. 建筑外部边缘：至少一个节点在外部空地（layout_matrix中值为0）
    2. 房间内部边缘：两个节点在同一布局矩阵非0区域
    3. 建筑外边缘：连接内部区域和外部区域的边缘
    4. 功能分区交界处：连接不同内部区域的边缘，需要LLM预测
    
    自动标记规则：
    - 建筑外部和房间内部的边缘 → empty(0)
    - 建筑外边缘（除出口外）→ 实墙(3)
    - 功能分区交界处 → 需要LLM预测
    """
    need_pred_edges = []
    auto_label_edges = []
    edge_types = []  # 存储每个边缘的自动标记类型
    
    # 如果没有布局矩阵，使用默认逻辑
    if layout_matrix is None:
        # 建筑边界坐标
        boundary_coords = {0.0, 200.0}
        
        for i in range(edge_conn.shape[1]):
            source = edge_conn[0, i]
            target = edge_conn[1, i]
            
            # 获取源节点和目标节点的坐标
            source_x, source_y = node_attrs[source][0], node_attrs[source][1]
            target_x, target_y = node_attrs[target][0], node_attrs[target][1]
            
            # 检查是否为外部边缘（连接到建筑外部）
            is_external = (source_x in boundary_coords or source_y in boundary_coords or 
                         target_x in boundary_coords or target_y in boundary_coords)
            
            # 直接从node_attrs中获取空间类型（第3个值，索引2）
            source_zone = node_attrs[source][2]
            target_zone = node_attrs[target][2]
            is_same_zone = (source_zone == target_zone)
            
            # 检查是否为同一网格内的边缘
            is_same_grid = (abs(source_x - target_x) < 20.0 and abs(source_y - target_y) < 20.0)
            
            if is_external or is_same_zone or is_same_grid:
                # 外部边缘、同一功能区块内的边缘或同一网格内的边缘，直接标记为空白边缘
                auto_label_edges.append(i)
                edge_types.append(0)  # 空白边缘
            else:
                # 不同功能区块之间的边缘，需要LLM预测
                need_pred_edges.append(i)
        
        return need_pred_edges, auto_label_edges
    
    # 使用layout_matrix来分类边缘
    # 建筑边界由layout_matrix中值为0的区域（外部空地）界定
    # 功能分区由layout_matrix中的非零值界定
    
    # 找到正门位置（layout_matrix中值为13的位置）
    main_door_pos = None
    for row in range(layout_matrix.shape[0]):
        for col in range(layout_matrix.shape[1]):
            if layout_matrix[row, col] == 13:
                main_door_pos = (row, col)
                break
        if main_door_pos:
            break
    
    for i in range(edge_conn.shape[1]):
        source = edge_conn[0, i]
        target = edge_conn[1, i]
        
        # 获取源节点和目标节点的坐标
        source_x, source_y = node_attrs[source][0], node_attrs[source][1]
        target_x, target_y = node_attrs[target][0], node_attrs[target][1]
        
        # 将坐标转换为layout_matrix中的索引
        # 假设坐标范围是0-200，对应layout_matrix的0-9索引
        # 每个单元格的大小是20x20
        source_row = min(int(source_y / 20), layout_matrix.shape[0] - 1)
        source_col = min(int(source_x / 20), layout_matrix.shape[1] - 1)
        target_row = min(int(target_y / 20), layout_matrix.shape[0] - 1)
        target_col = min(int(target_x / 20), layout_matrix.shape[1] - 1)
        
        # 获取源节点和目标节点的空间类型
        source_zone = layout_matrix[source_row, source_col]
        target_zone = layout_matrix[target_row, target_col]
        
        # 边缘类型分类
        is_both_external = (source_zone == 0 and target_zone == 0)  # 建筑外部边缘
        is_both_internal = (source_zone != 0 and target_zone != 0)  # 完全内部边缘
        is_same_internal = (is_both_internal and source_zone == target_zone)  # 房间内部边缘
        is_crossing_external = (source_zone == 0 or target_zone == 0)  # 建筑外边缘（连接内外）
        is_crossing_internal = (is_both_internal and source_zone != target_zone)  # 功能分区交界处
        
        # 检查是否靠近正门位置
        is_near_main_door = False
        if main_door_pos:
            # 检查边缘是否连接到正门所在的单元格或相邻单元格
            is_source_near_door = (abs(source_row - main_door_pos[0]) <= 1 and abs(source_col - main_door_pos[1]) <= 1)
            is_target_near_door = (abs(target_row - main_door_pos[0]) <= 1 and abs(target_col - main_door_pos[1]) <= 1)
            is_near_main_door = (is_source_near_door or is_target_near_door)
        
        # 只将完全外部边缘（两个节点都在0区域）自动标记为空白边缘(0)
        # 其余所有边缘，包括建筑外墙边缘和功能分区交界处边缘，都需要LLM预测
        if is_both_external:
            # 完全外部边缘：两个节点都在0区域（空地）→ 自动标记为空白边缘(0)
            # 0代表空地，那里的墙应该是空的
            auto_label_edges.append(i)
        else:  # is_same_internal or is_crossing_external or is_crossing_internal
            # 其余所有边缘，包括建筑外墙边缘、功能分区内部边缘和功能分区交界处边缘
            # 这些边缘都需要LLM预测
            need_pred_edges.append(i)
    
    return need_pred_edges, auto_label_edges

def calculate_edge_properties(source_x, source_y, target_x, target_y, layout_matrix):
    """
    计算边缘的方位属性和方向属性
    
    参数:
    - source_x, source_y: 源节点坐标
    - target_x, target_y: 目标节点坐标
    - layout_matrix: 布局矩阵
    
    返回:
    - edge_orientation: 边缘方位（上/下/左/右/内部）
    - edge_direction: 边缘方向（水平/竖直）
    """
    # 计算边缘的方向
    if abs(source_x - target_x) > abs(source_y - target_y):
        edge_direction = "水平"
    else:
        edge_direction = "竖直"
    
    # 计算建筑的边界
    building_rows = set()
    building_cols = set()
    for row in range(layout_matrix.shape[0]):
        for col in range(layout_matrix.shape[1]):
            if layout_matrix[row, col] != 0:
                building_rows.add(row)
                building_cols.add(col)
    
    min_row = min(building_rows)
    max_row = max(building_rows)
    min_col = min(building_cols)
    max_col = max(building_cols)
    
    # 计算边缘的中点坐标
    mid_x = (source_x + target_x) / 2
    mid_y = (source_y + target_y) / 2
    
    # 将中点坐标转换为布局矩阵中的索引
    mid_row = min(int(mid_y / 20), layout_matrix.shape[0] - 1)
    mid_col = min(int(mid_x / 20), layout_matrix.shape[1] - 1)
    
    # 确定边缘的方位
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

def graph_to_text(node_attrs, edge_conn, layout_matrix=None):
    """
    将图数据转换为LLM可处理的文本描述，只包含需要预测的边缘
    """
    text = []
    
    # 1. 分类边缘
    need_pred_edges, auto_label_edges = classify_edges(node_attrs, edge_conn, layout_matrix)
    
    # 2. 整体布局描述
    text.append("# 建筑户型布局描述")
    text.append(f"总节点数: {len(node_attrs)}, 总边缘数: {edge_conn.shape[1]}")
    text.append(f"需要预测的边缘数: {len(need_pred_edges)}, 自动标记的边缘数: {len(auto_label_edges)}")
    
    # 3. 布局矩阵与坐标映射
    text.append("\n## 布局矩阵与坐标映射")
    if layout_matrix is not None:
        text.append("布局矩阵编码:")
        text.append("0: 外部空地, 1: 客厅, 2: 主卧, 3: 厨房, 4: 卫生间, 5: 餐厅")
        text.append("6: 儿童房, 7: 书房, 8: 次卧, 9: 客房, 10: 阳台/庭院")
        text.append("11: 入口区域, 12: 储藏室, 13: 正门")
        
        # 完整显示布局矩阵
        text.append("\n布局矩阵:")
        for row_idx, row in enumerate(layout_matrix):
            row_str = ' '.join([str(val).rjust(2) for val in row])
            text.append(f"{row_idx:2d}: {row_str}")
        
        text.append("\n坐标映射规则:")
        text.append("- 坐标范围: 0-200，每个单元格大小为20x20")
        text.append("- 坐标(x,y) → 布局矩阵索引(row, col) = (int(y/20), int(x/20))")
    
    # 4. 入口位置
    text.append("\n## 入口位置")
    text.append("- 布局矩阵中值为13的位置为正门")
    text.append("- 入口区域由布局矩阵中值为11的区域界定")
    
    # 5. 收集需要预测边缘的相关节点信息
    relevant_nodes = set()
    for edge_idx in need_pred_edges:
        source = edge_conn[0, edge_idx]
        target = edge_conn[1, edge_idx]
        relevant_nodes.add(source)
        relevant_nodes.add(target)
    
    # 6. 只收集需要预测边缘的相关信息
    # 收集节点的空间类型
    space_types = {}
    for i in relevant_nodes:
        attr = node_attrs[i]
        x, y = attr[0], attr[1]
        
        # 优先从layout_matrix获取空间类型
        space_type_id = 0
        if layout_matrix is not None:
            row = min(int(y / 20), layout_matrix.shape[0] - 1)
            col = min(int(x / 20), layout_matrix.shape[1] - 1)
            space_type_id = layout_matrix[row, col]
        else:
            space_type_id = int(attr[2])
        
        space_type = SPACE_TYPE_MAP.get(space_type_id, f"未知({space_type_id})")
        space_types[i] = space_type
    
    # 7. 需要预测的边缘连接
    text.append("\n## 需要预测的边缘")
    text.append("以下边缘为功能分区交界处或建筑外边缘，需要预测其类型:")
    text.append("每个边缘的描述格式: 边缘ID: 连接 源空间类型 和 目标空间类型 [方位: 上/下/左/右/内部, 方向: 水平/竖直]")
    
    for i in need_pred_edges:
        source = edge_conn[0, i]
        target = edge_conn[1, i]
        
        # 获取源节点和目标节点的坐标
        source_x, source_y = node_attrs[source][0], node_attrs[source][1]
        target_x, target_y = node_attrs[target][0], node_attrs[target][1]
        
        # 获取源节点和目标节点的空间类型
        source_space_type = space_types[source]
        target_space_type = space_types[target]
        
        # 计算边缘的方位和方向属性
        edge_orientation = "内部"
        edge_direction = "水平"
        if layout_matrix is not None:
            edge_orientation, edge_direction = calculate_edge_properties(source_x, source_y, target_x, target_y, layout_matrix)
        
        edge_desc = f"边缘 {i}: 连接 {source_space_type} 和 {target_space_type} [方位: {edge_orientation}, 方向: {edge_direction}]"
        text.append(edge_desc)
    
    return '\n'.join(text)

def generate_llm_prompt(node_attrs, edge_conn, layout_matrix=None):
    """
    生成完整的LLM提示词，只包含需要预测的边缘信息
    """
    # 1. 分类边缘，获取需要预测的边缘列表
    need_pred_edges, auto_label_edges = classify_edges(node_attrs, edge_conn, layout_matrix)
    
    # 2. 确保待预测边缘列表中的ID是唯一的
    need_pred_edges = list(set(need_pred_edges))
    need_pred_edges.sort()
    
    # 3. 将图数据转换为文本描述
    layout_description = graph_to_text(node_attrs, edge_conn, layout_matrix)
    
    # 4. 构建待预测边缘列表
    edge_ids_str = ' '.join(map(str, need_pred_edges))
    
    prompt = f"""你是一位专业的建筑户型设计师，请根据提供的建筑户型布局描述，预测指定边缘的类型。

## 边缘类型定义
{chr(10).join([f"{k}: {v}" for k, v in EDGE_TYPE_MAP.items()])}

## 自动标记规则
1. 完全外部边缘（两个节点都在外部空地，布局矩阵值为0）→ 自动标记为空白边缘(0)（注：0代表空地，那里的墙应该是空的）
2. 房间内部边缘（两个节点在同一内部区域）→ 自动标记为空白边缘(0)

## 需要预测的边缘
- 建筑外墙边缘：连接内部区域和外部空地的边缘，需要形成封闭的建筑轮廓
- 功能分区交界处边缘：连接不同内部区域的边缘
- 正门边缘：连接外部空间和入口区域的边缘

## 建筑设计核心规则
1. **建筑外墙必须形成封闭轮廓**：所有建筑外墙边缘必须连续连接，形成一个没有缺口的封闭环，围绕整个建筑的布局矩阵中非零数字区域
2. **建筑外墙方向一致性**：建筑外墙边缘的方向应保持一致（顺时针或逆时针），这样在生成墙体时才会形成向内封闭的墙体，而不是向外张开
3. **单一正门原则**：每个户型只能有一个正门边缘，且必须直接连接外部空间（布局矩阵值为0）和入口区域（布局矩阵值为11）或正门位置（布局矩阵值为13）
4. **功能分区合理**：不同功能区域之间应使用适当的边缘类型，客厅、餐厅、厨房等公共区域应相互连通
5. **实体墙主要用于分隔**：实体墙边缘(3)主要用于建筑外墙和需要完全分隔的空间
6. **室内门用于连通**：室内门边缘(2)用于不同内部空间之间的通行

## 空间关系描述
1. **建筑整体布局**：建筑位于布局矩阵的中间区域，周围被外部空地（值为0）包围
2. **入口位置**：入口区域（值为11）和正门位置（值为13）位于建筑的左侧中部
3. **主要功能区域**：包括客厅(1)、主卧(2)、厨房(3)、卫生间(4)、餐厅(5)、次卧(8)等
4. **空间连通性**：公共区域（客厅、餐厅、厨房）应相互连通，私密区域（卧室、卫生间）应相对独立

## 墙体生成注意事项
- **墙体方向**：在建筑设计软件中，边缘的方向（起点到终点）决定了墙体生成的方向。同一封闭区域的边缘应保持一致的方向（全部顺时针或全部逆时针），否则墙体会向外张开，无法形成封闭空间
- **外墙围合**：建筑外墙应围绕布局矩阵中非零数字区域的最外层边缘，形成一个完整的封闭环
- **门的位置**：门应位于墙体的合理位置，便于通行

## 户型布局描述
{layout_description}

## 预测要求
请只为以下边缘ID预测其类型（其余边缘已自动标记）：
{edge_ids_str}

### 输出格式要求
请严格按照以下格式输出，不要添加任何额外内容：
边缘ID,预测类型
例如：
0,3
1,2
2,4

请确保：
1. 只预测指定的边缘ID
2. 每个边缘ID只出现一次
3. 预测类型为0-5之间的整数
4. **建筑外墙形成完整封闭环，没有任何缺口**
5. **建筑外墙边缘方向一致，确保墙体向内封闭**
6. 只有一个正门边缘，且位于合理的入口位置
7. 边缘类型分布合理，符合建筑设计规则
8. 公共区域相互连通，私密区域相对独立

请开始预测："""
    
    return prompt

def main():
    """
    主函数
    """
    # 1. 加载数据
    node_attrs, edge_conn, layout_matrix = load_node_attributes_and_edge_connections(INPUT_FILE_PATH)
    
    if node_attrs is None or edge_conn is None:
        print("Failed to load data, exiting...")
        return
    
    # 2. 生成提示词
    prompt = generate_llm_prompt(node_attrs, edge_conn, layout_matrix)
    
    # 3. 保存提示词到文件，使用相对路径
    output_file = os.path.join(FIXED_DIR, "llm_prompt.txt")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(prompt)
    
    print(f"\n提示词已生成并保存到: {output_file}")
    print(f"提示词长度: {len(prompt)} 字符")
    print("\n请将该提示词复制到在线AI（如GPT-4）中获取预测结果。")
    
    # 4. 显示提示词前1000字符预览
    print("\n提示词预览（前1000字符）:")
    print("=" * 50)
    print(prompt[:1000] + ("..." if len(prompt) > 1000 else ""))
    print("=" * 50)

if __name__ == "__main__":
    main()
