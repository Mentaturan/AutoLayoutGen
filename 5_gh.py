# 导入Rhino几何库，用于处理3D几何对象
import Rhino.Geometry as rg

# 从Grasshopper输入端口x获取边缘列表（Rhino线对象列表）
edge_list = x
# 从Grasshopper输入端口y获取边缘类型列表（整数列表，值为0-5）
edge_type_list = y

# 打印前3个边缘对象，用于调试
print(edge_list[:3])

# 打印前3个边缘类型，用于调试
print(edge_type_list[:3])

# 初始化一个字典，用于按边缘类型分组边缘
# 键为边缘类型（0-5），值为对应类型的边缘列表
# 边缘类型定义：
# 0: 空白边缘 - 无任何建筑构件的边缘
# 1: 正门边缘 - 入户正门所在的边缘
# 2: 室内门边缘 - 房间与房间/走廊之间的门所在边缘
# 3: 实体墙边缘 - 实心墙体的边缘
# 4: 幕墙边缘 - 玻璃幕墙的边缘
# 5: 矮墙边缘 - 高度低于常规墙体的边缘

grouped_edges = {
    0: [], # 空白边缘
    1: [], # 正门边缘
    2: [], # 室内门边缘
    3: [], # 实体墙边缘
    4: [], # 幕墙边缘
    5: []  # 矮墙边缘
}

# 确保边缘列表和边缘类型列表长度一致，避免zip时出错
if len(edge_list) != len(edge_type_list):
    print("Error: The number of edges does not match the number of edge types.")
else:
    # 使用zip函数同时遍历边缘列表和边缘类型列表
    for edge, edge_type in zip(edge_list, edge_type_list):
        # 检查边缘类型是否为有效键（0-5）
        if edge_type in grouped_edges:
            # 将边缘添加到对应类型的列表中
            grouped_edges[edge_type].append(edge)
        else:
            # 打印警告信息，提示遇到未知边缘类型
            print("Warning: Encountered an unknown edge type: {}".format(edge_type))

# --- 将结果分配到Grasshopper输出端口 ---

# 将空白边缘输出到Grasshopper的端口a
a = grouped_edges[0] 
# 将正门边缘输出到Grasshopper的端口b
b = grouped_edges[1] 
# 将室内门边缘输出到Grasshopper的端口c
c = grouped_edges[2] 
# 将实体墙边缘输出到Grasshopper的端口d
d = grouped_edges[3] 
# 将幕墙边缘输出到Grasshopper的端口e
e = grouped_edges[4] 
# 将矮墙边缘输出到Grasshopper的端口f
f = grouped_edges[5] 

# 可选：打印各类边缘的数量，用于验证和调试
print("\nEdge Counts by Type:")
print("Type 0 (Empty): {}".format(len(a)))  # 打印空白边缘数量
print("Type 3 (Solid Wall): {}".format(len(d)))  # 打印实体墙边缘数量