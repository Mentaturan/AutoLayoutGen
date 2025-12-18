# 导入Rhino几何库，用于处理3D几何对象
import Rhino.Geometry as rg
# 导入json库，用于数据序列化和反序列化
import json
# 导入os库，用于文件和目录操作
import os

# 网格尺寸设置，默认为10x10网格
M = N = 10
# 网格步长，每个网格单元的大小为20.0单位
# 注意：需与后续generate_grid_and_positions函数中的STEP保持一致
STEP = 20.0  

def compute_boundary_mask(bnd_curve, M=10, N=10, step=20.0):
    """
    根据用户绘制的闭合边界曲线，计算规则网格中每个单元格是否在边界内
    
    参数：
    - bnd_curve: Rhino闭合曲线对象，用户在GH中绘制的建筑边界
    - M: 网格行数，默认10
    - N: 网格列数，默认10
    - step: 网格步长，每个网格单元的大小，默认20.0
    
    返回：
    - mask: 二维列表，表示每个网格单元是否在边界内
            1表示在边界内，0表示在边界外
    """
    # 初始化边界掩码矩阵
    mask = []
    # 容差设置，用于点包含检测
    tol = 1e-6
    
    # 遍历所有网格行（从上到下）
    for i in range(M):      
        row = []
        # 遍历当前行的所有网格列（从左到右）
        for j in range(N):  
            # 计算当前网格单元的中心点坐标
            # 采用row-major逻辑，与矩阵索引一致
            x = (j + 0.5) * step   # 单元格中心点x坐标
            y = (i + 0.5) * step   # 单元格中心点y坐标
            # 创建Rhino点对象
            pt = rg.Point3d(x, y, 0)

            # 检测点是否在闭合曲线内
            # 使用Rhino的Contains方法，考虑容差
            containment = bnd_curve.Contains(pt, rg.Plane.WorldXY, tol)
            
            # 如果点在曲线内或在曲线上，标记为1，否则标记为0
            if containment == rg.PointContainment.Inside or containment == rg.PointContainment.Coincident:
                row.append(1)
            else:
                row.append(0)
        # 将当前行添加到掩码矩阵
        mask.append(row)
    
    # 返回完整的边界掩码矩阵
    return mask

# bnd_crv是从Grasshopper输入的闭合曲线
# 计算边界掩码
boundary_mask = compute_boundary_mask(bnd_crv, M, N, STEP)
# 将边界掩码转换为JSON字符串
boundary_mask_json = json.dumps(boundary_mask)

# 检查输入曲线是否有效
if bnd_crv is not None:
    # 重新计算边界掩码，确保结果正确
    boundary_mask = compute_boundary_mask(bnd_crv, M, N, STEP)
    # 将结果输出到Grasshopper的输出端口a
    a = boundary_mask          
    # 在终端窗口打印边界掩码，便于调试和查看结果
    print(boundary_mask)       
else:
    # 如果输入曲线无效，返回空列表
    a = []