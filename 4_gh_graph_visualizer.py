# 导入JSON库，用于数据序列化和反序列化
import json
# 导入操作系统库，用于文件和目录操作
import os
# 导入Rhino几何库，用于创建3D几何对象
import Rhino.Geometry as rg


# 全局输入文件路径：包含节点属性和边缘连接的数据文件
INPUT_FILE_PATH = "node_attr_edge_conn_4GEN.json"

# 读取JSON文件
with open(INPUT_FILE_PATH, 'r') as f:
    data = json.load(f)

# 提取节点属性和边缘连接数据
node_attrs = data['node_attrs']
edge_conn = data['edge_conn']

# 从节点属性中提取坐标信息（只使用前两个属性：x和y坐标）
coords = [(attr[0], attr[1]) for attr in node_attrs]

# 创建Rhino点对象列表
rhino_points = []
for x, y in coords:
    # 创建Rhino.Point3d对象，z坐标设为0.0（二维平面）
    pt = rg.Point3d(float(x), float(y), 0.0)
    # 添加到点列表
    rhino_points.append(pt)

# 创建Rhino线对象列表（表示边缘）
rhino_lines = []
# 检查边缘连接数据结构是否正确
if len(edge_conn) == 2 and len(edge_conn[0]) == len(edge_conn[1]):
    # 提取源节点索引和目标节点索引
    source_indices = edge_conn[0]
    target_indices = edge_conn[1]

    # 遍历所有边缘
    for i in range(len(source_indices)):
        src_idx = source_indices[i]  # 源节点索引
        tgt_idx = target_indices[i]  # 目标节点索引

        # 验证节点索引是否有效
        if 0 <= src_idx < len(rhino_points) and 0 <= tgt_idx < len(rhino_points):
            # 获取源节点和目标节点的Rhino点对象
            start_point = rhino_points[src_idx]
            end_point = rhino_points[tgt_idx]

            # 创建Rhino.Line对象，连接源节点和目标节点
            line = rg.Line(start_point, end_point)
            # 添加到线列表
            rhino_lines.append(line)
        else:
            # 打印警告信息，提示无效节点索引
            print("Warning: Invalid node index found in edge connections.")
else:
    # 打印错误信息，提示边缘连接数据结构不正确
    print("Error: edge_conn structure is not [2, num_edges] or lists are unequal length.")


# 4. 输出结果到Grasshopper端口
# 将Rhino点对象列表输出到Grasshopper的端口a
node_list = rhino_points

# 将Rhino线对象列表输出到Grasshopper的端口b
edge_list = rhino_lines

# 打印调试信息，表示程序运行正常
print("work well")

# 从Grasshopper输入的房间中心点数据（扁平化列表）
flat_list = u

# 将扁平化列表转换为二维坐标列表，每两个元素一组表示一个点
room_cnts = [[flat_list[i], flat_list[i + 1]] for i in range(0, len(flat_list), 2)]

# 打印房间中心点坐标列表，用于调试
print(room_cnts)

# 创建房间中心点的Rhino点对象列表
room_cnts_rh_pts = []
for x, y in room_cnts:
    # 创建Rhino.Point3d对象，z坐标设为0.0
    pt = rg.Point3d(float(x), float(y), 0.0)
    # 添加到房间中心点列表
    room_cnts_rh_pts.append(pt)

# 将房间中心点列表输出到Grasshopper的端口rm_center
try:
    rm_center = room_cnts_rh_pts
except Exception as e:
    print(f"Error setting rm_center: {e}")

# 从Grasshopper输入的房间标签
rm_label = v

# 打印房间标签，用于调试
print(v)