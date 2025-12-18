# 导入NumPy库，用于数值计算和矩阵操作
import numpy as np
# 从shapely库导入几何对象，用于处理空间几何计算
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.geometry import Point, Polygon, LineString
# 从shapely库导入合并多边形的函数
from shapely.ops import unary_union
# 导入JSON库，用于数据序列化和反序列化
import json
# 导入OS库，用于文件和目录操作
import os

json_file_path = "FPGGen_RoomLayoutMatrix.json"

# Set input & output directories
input_dir = '.'
output_dir = '.'

with open(json_file_path, 'r') as f:
    room_layout_matrix = np.array(json.load(f))
print(f"Loaded room layout from {json_file_path}")

# Define room types
room_types = {
    0: "House exterior",
    1: "Living room", 
    2: "Master room", 
    3: "Kitchen", 
    4: "Bathroom", 
    5: "Dining room", 
    6: "Child room", 
    7: "Study room", 
    8: "Second room", 
    9: "Guest room", 
    10: "Balcony/Yard", 
    11: "Entrance", 
    12: "Storage",
    13: "Front door"
}

# Room type color mapping
room_colors = {
    0: 'white',         # House exterior
    1: 'lightyellow',   # Living room
    2: 'lightpink',     # Master room
    3: 'lightgreen',    # Kitchen
    4: 'lightblue',     # Bathroom
    5: 'lightsalmon',   # Dining room
    6: 'lavender',      # Child room
    7: 'wheat',         # Study room
    8: 'lightcoral',    # Second room
    9: 'paleturquoise', # Guest room
    10: 'lightgreen',   # Balcony/Yard
    11: 'khaki',        # Entrance
    12: 'silver',       # Storage
    13: 'red'          # Front door
}


def transform_x_coords(x_coords, x_distance):  
    """
    转换x坐标，根据列间距生成实际坐标值
    
    参数:
        x_coords: 初始x坐标索引列表
        x_distance: 每列之间的间距列表
    
    返回:
        transformed_x_coords: 转换后的x坐标列表
    """
    # 初始化转换后的x坐标列表，复制第一行作为初始值  
    transformed_x_coords = [x_coords[0].copy()]  
    
    # 根据列间距计算累积x位置  
    current_x = 0  
    for i in range(1, len(x_coords)):  
        current_x += x_distance[i - 1]  # 累加前一列的间距  
        transformed_x_coords.append([current_x] * len(x_coords[i]))  # 更新当前行的x位置  

    return transformed_x_coords  

def transform_y_coords(y_coords, y_distance):  
    """
    转换y坐标，根据行间距生成实际坐标值
    
    参数:
        y_coords: 初始y坐标索引列表
        y_distance: 每行之间的间距列表
    
    返回:
        transformed_y_coords: 转换后的y坐标列表
    """
    # 转置y坐标，以便使用transform_x_coords函数处理  
    y_coords_transposed = np.array(y_coords).T.tolist()  
    
    # 使用transform_x_coords函数转换转置后的y坐标  
    transformed_y_coords = transform_x_coords(y_coords_transposed, y_distance)  
    
    # 再次转置，恢复原始形状  
    transformed_y_coords = np.array(transformed_y_coords).T.tolist()  

    return transformed_y_coords  

def generate_grid_and_positions(col_distances, row_distances):  
    """
    生成网格单元和节点位置
    
    参数:
        col_distances: 列间距列表
        row_distances: 行间距列表
    
    返回:
        cells: 网格单元列表，每个单元由四个节点索引组成
        node_positions: 节点位置数组，每行包含一个节点的x和y坐标
    """
    # 计算节点行列数  
    num_node_rows = len(row_distances) + 1  # 节点行数 = 单元格行数 + 1  
    num_node_cols = len(col_distances) + 1  # 节点列数 = 单元格列数 + 1  
    cells = []  # 初始化网格单元列表  

    # 按行优先顺序生成网格单元
    for j in range(num_node_rows - 1):  # 遍历行（单元格行）
        for i in range(num_node_cols - 1):  # 遍历列（单元格列）
            # 计算当前单元格四个顶点的节点索引
            lower_left = i * num_node_rows + j  
            upper_left = lower_left + 1  
            lower_right = lower_left + num_node_rows  
            upper_right = lower_right + 1  
            # 添加单元格，顺序为：左下、右下、右上、左上
            cells.append([lower_left, lower_right, upper_right, upper_left])  

    # 生成节点索引矩阵  
    node_indices = np.arange(num_node_rows * num_node_cols).reshape(num_node_cols, num_node_rows)  
    # 提取x和y坐标索引
    x_coords = (node_indices // num_node_rows).tolist()
    y_coords = (node_indices % num_node_rows).tolist()
    
    # 转换坐标索引为实际坐标值
    transformed_x_coords = transform_x_coords(x_coords, col_distances)
    transformed_y_coords = transform_y_coords(y_coords, row_distances)
    
    # 将坐标转换为一维数组
    x_coords = np.array(transformed_x_coords).flatten()
    y_coords = np.array(transformed_y_coords).flatten()

    # 合并x和y坐标，生成节点位置数组
    node_positions = np.column_stack((x_coords, y_coords))  

    return cells, node_positions  

def create_cell_polygons(cells, node_positions):
    """
    为每个网格单元创建多边形对象
    
    参数:
        cells: 网格单元列表
        node_positions: 节点位置数组
    
    返回:
        cell_polygons: 网格单元多边形列表
    """
    cell_polygons = []
    for cell in cells:
        # 获取当前单元格的四个顶点坐标
        cell_positions = [node_positions[node_idx] for node_idx in cell]
        # 创建多边形对象
        polygon = ShapelyPolygon(cell_positions)
        # 添加到多边形列表
        cell_polygons.append(polygon)
    
    return cell_polygons

def merge_rooms_by_type(room_layout, cell_polygons):
    """
    根据房间类型合并相邻的网格单元
    
    参数:
        room_layout: 房间布局矩阵
        cell_polygons: 网格单元多边形列表
    
    返回:
        room_polygons: 按房间类型合并后的多边形字典
    """
    # 初始化房间多边形字典
    room_polygons = {}
    
    # 获取所有唯一的房间类型
    unique_room_types = np.unique(room_layout)

    for room_type in unique_room_types:
        # 收集当前房间类型的所有网格单元
        room_cells = []
        for i in range(room_layout.shape[0]):
            for j in range(room_layout.shape[1]):
                if room_layout[i, j] == room_type:
                    # 计算当前单元格的索引
                    cell_idx = i * room_layout.shape[1] + j
                    # 添加到当前房间类型的单元格列表
                    room_cells.append(cell_polygons[cell_idx])
        
        # 使用shapely的unary_union合并当前房间类型的所有单元格
        if room_cells:
            merged_polygon = unary_union(room_cells)
            room_polygons[room_type] = merged_polygon
    
    return room_polygons

def create_house_boundary_polygon(room_polygons):
    """
    创建建筑边界多边形
    
    参数:
        room_polygons: 按房间类型合并后的多边形字典
    
    返回:
        house_boundary: 建筑边界多边形，或None（无法创建时）
    """
    all_polygons = []
    
    # 遍历所有房间多边形
    for room_type, polygon in room_polygons.items():
        # 跳过正门和建筑外部
        if room_type == 13 or room_type == 0:
            continue
            
        # 跳过空多边形
        if polygon.is_empty:
            continue
            
        # 根据多边形类型添加到列表
        if polygon.geom_type == 'Polygon':
            all_polygons.append(polygon)
        elif polygon.geom_type == 'MultiPolygon':
            all_polygons.extend(list(polygon.geoms))

    # 合并所有多边形，创建建筑边界
    if all_polygons:
        house_boundary = unary_union(all_polygons)
        return house_boundary
    else:
        return None


def save_grid_nodes_and_edges_to_json(node_positions, cells, output_file='grid_nodes_edges.json', hm_bnd=None, front_door_pt=None, room_list=None):
    """
    将网格节点和边缘数据保存到JSON文件
    
    参数:
        node_positions: 节点位置数组
        cells: 网格单元列表
        output_file: 输出文件路径
        hm_bnd: 建筑边界多边形（可选）
        front_door_pt: 正门中心点（可选）
        room_list: 房间列表（可选）
    """
    # 将节点位置转换为列表格式
    node_coords = [[float(pos[0]), float(pos[1])] for pos in node_positions]

    edges_set = set()  # 使用集合避免重复边缘
    
    # 提取所有边缘
    all_edges = []
    for cell in cells:
        all_edges.append((cell[0], cell[1]))  # 底边
        all_edges.append((cell[1], cell[2]))  # 右边
        all_edges.append((cell[2], cell[3]))  # 顶边
        all_edges.append((cell[3], cell[0]))  # 左边

    # 确定边缘方向，确保每个边缘只保留一个方向
    edge_directions = {}
    for edge in all_edges:
        source, target = edge
        # 创建排序后的键，用于检测重复边缘
        key = tuple(sorted([source, target]))
        
        # 保留较小的边缘表示
        if key not in edge_directions or edge < edge_directions[key]:
            edge_directions[key] = edge
    
    # 将边缘添加到集合
    for edge in edge_directions.values():
        edges_set.add(edge)
    
    # 转换为[2, num_edges]格式
    edge_conn = [[], []]
    for source, target in edges_set:
        edge_conn[0].append(source)  # 边缘源节点列表
        edge_conn[1].append(target)  # 边缘目标节点列表
    
    # 创建网格数据字典
    grid_data = {
        'node_coords': node_coords,  # 节点坐标
        'edge_conn': edge_conn       # 边缘连接关系
    }
    
    # 添加建筑边界数据
    if hm_bnd and not hm_bnd.is_empty:
        if hm_bnd.geom_type == 'Polygon':
            coords = list(hm_bnd.exterior.coords)
            grid_data['hm_bnd'] = coords  
        elif hm_bnd.geom_type == 'MultiPolygon':
            polygons = []
            for poly in hm_bnd.geoms:
                coords = list(poly.exterior.coords)
                polygons.append(coords)
            grid_data['hm_bnd'] = polygons 
    
    # 添加正门中心点数据
    if front_door_pt:
        grid_data['front_door_pt'] = [float(front_door_pt[0]), float(front_door_pt[1])]  # 存储为简单数组
    
    # 添加房间列表数据
    if room_list:
        room_types_list = []
        room_polygons_list = []
        
        # 过滤掉建筑外部和正门
        filtered_rooms = [room for room in room_list if room['room_type'] not in [0, 13]]
        # 按房间类型排序
        filtered_rooms.sort(key=lambda x: x['room_type'])

        for room in filtered_rooms:
            room_types_list.append(room['room_type'])

            # 提取房间多边形坐标
            if room['polygon'].geom_type == 'Polygon':
                coords = list(room['polygon'].exterior.coords)
                room_polygons_list.append(coords)
            elif room['polygon'].geom_type == 'MultiPolygon':
                coords = list(room['polygon'].geoms[0].exterior.coords)
                room_polygons_list.append(coords)
        
        # 添加到网格数据
        grid_data['room_type'] = room_types_list
        grid_data['room_polygon'] = room_polygons_list
    
    # 保存到JSON文件
    with open(output_file, 'w') as f:
        json.dump(grid_data, f, indent=2)
    
    # 打印保存信息
    print(f"Grid nodes and edges saved to {output_file}")
    print(f"  - Number of nodes: {len(node_coords)}")
    print(f"  - Number of edges: {len(edge_conn[0])}")
    print(f"  - All edges are unidirectional as required")


def create_room_list(room_polygons):
    """
    创建房间信息列表
    
    参数:
        room_polygons: 按房间类型合并后的多边形字典
    
    返回:
        room_list: 房间信息列表
    """
    room_list = []
    
    # 为每种房间类型创建房间信息
    for room_type, polygon in room_polygons.items():
        # 跳过正门和建筑外部
        if room_type == 13 or room_type == 0:
            continue
            
        # 跳过空多边形
        if polygon.is_empty:
            continue
            
        # 获取房间名称
        room_name = room_types.get(room_type, f"Room {room_type}")
        
        # 创建房间信息字典
        room_info = {
            'room_type': int(room_type),  # 房间类型编码
            'room_name': room_name,        # 房间名称
            'polygon': polygon             # 房间多边形
        }
        # 添加到房间列表
        room_list.append(room_info)
    
    return room_list

def get_front_door_with_center(room_layout, cell_polygons, node_positions):
    """
    获取正门多边形及其中心点
    
    参数:
        room_layout: 房间布局矩阵
        cell_polygons: 网格单元多边形列表
        node_positions: 节点位置数组
    
    返回:
        front_door_polygon: 正门多边形
        front_door_center: 正门中心点坐标
    """
    front_door_cells = []
    front_door_indices = []
    
    # 查找所有正门单元格
    for i in range(room_layout.shape[0]):
        for j in range(room_layout.shape[1]):
            if room_layout[i, j] == 13:  # 正门类型编码为13
                # 计算当前单元格索引
                cell_idx = i * room_layout.shape[1] + j
                # 添加到正门单元格列表
                front_door_cells.append(cell_polygons[cell_idx])
                # 添加到正门索引列表
                front_door_indices.append(cell_idx)
    
    if front_door_cells:
        # 合并正门单元格，创建正门多边形
        front_door_polygon = unary_union(front_door_cells)
        
        if front_door_indices:
            # 获取第一个正门单元格的索引
            first_front_door_idx = front_door_indices[0]
            num_cols = room_layout.shape[1]
            # 计算行和列索引
            row = first_front_door_idx // num_cols
            col = first_front_door_idx % num_cols

            # 计算正门单元格的四个顶点节点索引
            num_node_rows = room_layout.shape[0] + 1
            lower_left = col * num_node_rows + row
            lower_right = lower_left + num_node_rows
            upper_right = lower_right + 1
            upper_left = lower_left + 1
            
            # 获取四个顶点的位置
            node_positions_list = [
                node_positions[lower_left],
                node_positions[lower_right],
                node_positions[upper_right],
                node_positions[upper_left]
            ]
            
            # 计算正门中心点
            center_x = sum(pos[0] for pos in node_positions_list) / 4
            center_y = sum(pos[1] for pos in node_positions_list) / 4
            front_door_center = (center_x, center_y)
        else:
            front_door_center = None
            
        return front_door_polygon, front_door_center
    else:
        # 没有找到正门时返回None
        return None, None


M, N = 10, 10  # 10x10 cells
STEP = 20      # Grid step size

# Create grid distances
x_distances = [STEP] * N
y_distances = [STEP] * M

# Generate grid and node positions
cells, node_positions = generate_grid_and_positions(x_distances, y_distances)

# Create polygons for each cell
cell_polygons = create_cell_polygons(cells, node_positions)

# Merge cells by room type
room_polygons = merge_rooms_by_type(room_layout_matrix, cell_polygons)

# Create house boundary polygon
house_boundary = create_house_boundary_polygon(room_polygons)

# Create room list (excluding house exterior and front door)
room_list = create_room_list(room_polygons)

# Get front door polygon and its center point
front_door_polygon, front_door_center = get_front_door_with_center(room_layout_matrix, cell_polygons, node_positions)


os.makedirs(output_dir, exist_ok=True)

# Save grid nodes and edge connections to JSON file
save_grid_nodes_and_edges_to_json(node_positions, cells, f'{output_dir}/grid_nodes_edges.json', house_boundary, front_door_center, room_list)


def load_grid_data(json_file):
    """
    加载网格数据
    
    参数:
        json_file: 网格数据JSON文件路径
    
    返回:
        data: 加载的网格数据字典
    """
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    return data

def extract_room_type_for_nodes(node_coords, room_types, room_polygons):
    """
    提取每个节点所属的房间类型
    
    参数:
        node_coords: 节点坐标列表
        room_types: 房间类型列表
        room_polygons: 房间多边形坐标列表
    
    返回:
        node_room_types: 每个节点所属的房间类型列表
    """
    # 初始化节点房间类型列表
    node_room_types = [[] for _ in range(len(node_coords))]
    
    # 遍历所有房间
    for room_idx, (room_type, polygon) in enumerate(zip(room_types, room_polygons)):
        # 将多边形坐标转换为LineString对象
        polygon_line = LineString(polygon)
        
        # 检查每个节点是否在当前房间的多边形上
        for node_idx, (x, y) in enumerate(node_coords):
            point = Point(x, y)
            
            # 使用小容差检查节点是否在多边形上
            if polygon_line.distance(point) < 0.1:  # 使用0.1的容差值
                node_room_types[node_idx].append(room_type)
    
    return node_room_types


def room_types_to_one_hot(node_room_types, num_room_types=12):
    """
    将节点房间类型转换为独热编码
    
    参数:
        node_room_types: 每个节点所属的房间类型列表
        num_room_types: 房间类型数量（默认12）
    
    返回:
        one_hot_encoded: 独热编码后的节点房间类型
    """
    one_hot_encoded = []
    
    # 遍历每个节点的房间类型
    for room_types in node_room_types:
        # 初始化独热编码向量
        one_hot = [0] * num_room_types
        
        # 为每个房间类型设置对应位置为1
        for room_type in room_types:
            # 确保房间类型在有效范围内
            if 1 <= room_type <= num_room_types:
                one_hot[room_type - 1] = 1
        
        # 添加到独热编码列表
        one_hot_encoded.append(one_hot)
    
    return one_hot_encoded


def extract_node_attributes(node_coords, house_boundary, front_door_pt=None, room_types=None, room_polygons=None):
    """
    提取节点属性
    
    参数:
        node_coords: 节点坐标列表
        house_boundary: 建筑边界坐标列表
        front_door_pt: 正门中心点坐标
        room_types: 房间类型列表
        room_polygons: 房间多边形坐标列表
    
    返回:
        node_attrs: 节点属性字典
    """
    # 创建建筑边界多边形对象
    house_polygon = Polygon(house_boundary)
    # 创建建筑边界线对象
    house_boundary_line = LineString(house_boundary)
    
    # 获取节点数量
    num_nodes = len(node_coords)
    # 初始化节点属性数组
    is_on_hm_bnd = np.zeros(num_nodes, dtype=int)      # 是否在建筑边界上
    is_inside_hm_bnd = np.zeros(num_nodes, dtype=int)  # 是否在建筑边界内
    is_outside_hm_bnd = np.zeros(num_nodes, dtype=int)  # 是否在建筑边界外
    is_front_door = np.zeros(num_nodes, dtype=int)     # 是否在正门位置
    
    # 计算每个节点的属性
    for i, (x, y) in enumerate(node_coords):
        point = Point(x, y)
        
        # 检查节点是否在建筑边界上
        if house_boundary_line.distance(point) < 1e-6:  
            is_on_hm_bnd[i] = 1
        # 检查节点是否在建筑边界内
        elif house_polygon.contains(point):
            is_inside_hm_bnd[i] = 1
        # 否则节点在建筑边界外
        else:
            is_outside_hm_bnd[i] = 1
    
    # 处理建筑边界坐标，移除重复的最后一个点
    house_boundary_unique = house_boundary[:-1] if house_boundary[0] == house_boundary[-1] else house_boundary
    
    # 计算正门附近的节点
    front_door_point = Point(front_door_pt)
    distances = []
    
    # 计算建筑边界上每个点到正门的距离
    for i, boundary_node in enumerate(house_boundary_unique):
        boundary_point = Point(boundary_node)
        distance = front_door_point.distance(boundary_point)
        distances.append((i, distance, boundary_node))
    
    # 按距离排序
    distances.sort(key=lambda x: x[1])
    # 获取最近的两个边界点
    nearest_two_indices = [distances[0][0], distances[1][0]]
    nearest_two_nodes = [distances[0][2], distances[1][2]]
    
    # 标记正门附近的节点
    for i, node_coord in enumerate(node_coords):
        for nearest_node in nearest_two_nodes:
            # 检查节点是否与最近的边界点重合
            if (abs(node_coord[0] - nearest_node[0]) < 1e-6 and 
                abs(node_coord[1] - nearest_node[1]) < 1e-6):
                is_front_door[i] = 1
                break
    
    # 创建节点属性字典
    node_attrs = {
        'is_on_hm_bnd': is_on_hm_bnd.tolist(),        # 是否在建筑边界上
        'is_inside_hm_bnd': is_inside_hm_bnd.tolist(),  # 是否在建筑边界内
        'is_outside_hm_bnd': is_outside_hm_bnd.tolist(),  # 是否在建筑边界外
        'is_front_door': is_front_door.tolist()       # 是否在正门位置
    }
    
    # 如果提供了房间信息，添加房间类型属性
    if room_types is not None and room_polygons is not None:
        # 提取节点房间类型
        node_room_types = extract_room_type_for_nodes(node_coords, room_types, room_polygons)
        # 转换为独热编码
        node_room_types_one_hot = room_types_to_one_hot(node_room_types)
        # 添加到节点属性字典
        node_attrs['room_types'] = node_room_types
        node_attrs['room_types_one_hot'] = node_room_types_one_hot
    
    return node_attrs


def save_node_attributes_to_json(node_attrs, output_path):
    """
    保存节点属性到JSON文件
    
    参数:
        node_attrs: 节点属性字典
        output_path: 输出文件路径
    """
    # 创建属性数据字典
    attrs_data = {
        'is_on_hm_bnd': node_attrs['is_on_hm_bnd'],
        'is_inside_hm_bnd': node_attrs['is_inside_hm_bnd'],
        'is_outside_hm_bnd': node_attrs['is_outside_hm_bnd'],
        'is_front_door': node_attrs['is_front_door']
    }
    
    # 如果包含房间类型信息，添加到属性数据
    if 'room_types' in node_attrs:
        attrs_data['room_types'] = node_attrs['room_types']
        attrs_data['room_types_one_hot'] = node_attrs['room_types_one_hot']

    # 保存到JSON文件
    with open(output_path, 'w') as f:
        json.dump(attrs_data, f, indent=4)
    
    print(f"Node attributes saved to: {output_path}")


def save_combined_data_to_json(node_attrs_full, edge_conn, output_path):
    """
    保存合并的节点属性和边缘连接数据到JSON文件
    
    参数:
        node_attrs_full: 完整的节点属性数组
        edge_conn: 边缘连接关系
        output_path: 输出文件路径
    """
    # 创建合并数据字典
    combined_data = {
        'node_attrs': node_attrs_full,  # 节点属性
        'edge_conn': edge_conn          # 边缘连接关系
    }
    
    # 保存到JSON文件
    with open(output_path, 'w') as f:
        json.dump(combined_data, f, indent=2)
    
    print(f"Combined data saved to: {output_path}")


os.makedirs(output_dir, exist_ok=True)

input_file = f'{input_dir}/grid_nodes_edges.json'

grid_data = load_grid_data(input_file)

# Extract data
node_coords = grid_data['node_coords']
house_boundary = grid_data['hm_bnd']
front_door_pt = grid_data['front_door_pt']
room_types = grid_data['room_type']
room_polygons = grid_data['room_polygon']

# Extract node attributes
print("\nExtracting node attributes...")
node_attrs = extract_node_attributes(node_coords, house_boundary, front_door_pt, room_types, room_polygons)

# Statistics of attributes
on_boundary_count = sum(node_attrs['is_on_hm_bnd'])
inside_count = sum(node_attrs['is_inside_hm_bnd'])
outside_count = sum(node_attrs['is_outside_hm_bnd'])
front_door_count = sum(node_attrs['is_front_door'])

# Find indices and coordinates of front door nodes
front_door_indices = [i for i, val in enumerate(node_attrs['is_front_door']) if val == 1]
if front_door_indices:
    print(f"\nFront door node details:")
    for idx in front_door_indices:
        print(f"- Node {idx}: Coordinates {node_coords[idx]}")

# Statistics of room types
if 'room_types' in node_attrs:
    room_type_counts = {}
    nodes_with_rooms = 0
    for room_type_list in node_attrs['room_types']:
        if room_type_list:  # If node has room types
            nodes_with_rooms += 1
            for room_type in room_type_list:
                room_type_counts[room_type] = room_type_counts.get(room_type, 0) + 1
    
    print(f"\nRoom type statistics:")
    print(f"- Number of nodes with room types: {nodes_with_rooms}")
    for room_type, count in sorted(room_type_counts.items()):
        room_type_names = {
            1: 'Living room',
            2: 'Master room',
            3: 'Kitchen',
            4: 'Bathroom',
            5: 'Dining room',
            6: 'Child room',
            7: 'Study room',
            8: 'Second room',
            9: 'Guest room',
            10: 'Balcony',
            11: 'Entrance',
            12: 'Storage'
        }
        room_name = room_type_names.get(room_type, f'Room{room_type}')
        print(f"- {room_name} (Type{room_type}): {count} nodes")

# Create complete node attribute array (18 dimensions)

node_attrs_full = []
for i, (x, y) in enumerate(node_coords):
    # NOTE: 18-dimensional attribute vector
    attr_vector = [x, y]  # First two dimensions are coordinates
    attr_vector.extend([0] * 16)  # Initialize the remaining 16 dimensions to 0
    
    # Set positional attributes
    attr_vector[2] = node_attrs['is_on_hm_bnd'][i]  # attr_03
    attr_vector[3] = node_attrs['is_inside_hm_bnd'][i]  # attr_04
    attr_vector[4] = node_attrs['is_outside_hm_bnd'][i]  # attr_05
    attr_vector[5] = node_attrs['is_front_door'][i]  # attr_06
    
    # Set room type attributes (if available)
    if 'room_types_one_hot' in node_attrs:
        # Room types are attr_07 to attr_18 (12 dimensions)
        for j, val in enumerate(node_attrs['room_types_one_hot'][i]):
            attr_vector[6 + j] = val  # attr_07 to attr_18
    
    node_attrs_full.append(attr_vector)

# Load edge_conn data
with open(input_file, 'r') as f:
    grid_data = json.load(f)
edge_conn = grid_data['edge_conn']

# Save combined data to JSON file
combined_output_file = f'{output_dir}/node_attr_edge_conn_4GEN.json'
save_combined_data_to_json(node_attrs_full, edge_conn, combined_output_file)

a = 1

room_center = []

for rm in room_polygons:
    poly = Polygon(rm)
    cnt = poly.centroid
    room_center.append([int(cnt.x), int(cnt.y)])

b = room_center

print(b)

room_type_list = {
    0: "House exterior",
    1: "Living room", 
    2: "Master room", 
    3: "Kitchen", 
    4: "Bathroom", 
    5: "Dining room", 
    6: "Child room", 
    7: "Study room", 
    8: "Second room", 
    9: "Guest room", 
    10: "Balcony/Yard", 
    11: "Entrance", 
    12: "Storage",
    13: "Front door"
}

rm_lables =  [str(room_type_list[key]) for key in room_types]
print(rm_lables)
c = rm_lables
