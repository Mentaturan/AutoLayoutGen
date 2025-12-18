# 自动化建筑布局生成工具

**英文名称：AutoLayoutGen**

**⚠️ 项目状态：开发中 ⚠️**

> 注意：当前项目处于开发阶段，文件可能不完整，功能可能存在bug或尚未完全实现。
> 特别注意：项目依赖于Rhino软件及其Grasshopper插件，但Rhino中的工作流尚未完全搭建完成。

一个基于网格的自动化建筑布局生成系统，结合可视化边界绘制和AI驱动的房间布局生成功能，旨在帮助建筑师和设计师快速生成建筑布局方案。

## 项目概述

该工具旨在帮助建筑师和设计师快速生成建筑布局方案。系统通过可视化界面绘制建筑边界，然后利用算法自动生成合理的房间布局和边缘类型预测，最终输出可用于进一步设计的建筑布局方案。

## 功能特点

### 🎨 可视化边界绘制
- 直观的网格界面，支持10x10和20x20网格大小
- 左键绘制边界，右键擦除边界
- 实时预览内部区域
- 支持JSON格式的边界矩阵导入/导出

### 🏗️ 自动化布局生成
- 基于边界矩阵自动生成房间布局
- 智能的房间尺寸和位置分配
- 支持不同建筑类型的布局生成

### 🤖 AI驱动的边缘类型预测
- 自动预测建筑边缘类型
- 支持多种边缘类型分类
- 结合GNN和LLM技术提高预测准确性

### 📊 可视化输出
- 生成直观的建筑布局图形
- 支持多种输出格式
- 便于进一步设计和修改

## 系统架构

该系统采用模块化设计，主要包含以下核心模块：

1. **边界绘制模块** (`matrix_generator.html`)：提供可视化界面，用于手动绘制建筑边界
2. **边界处理模块** (`0_gh_boundary_mask_generator.py`)：处理边界矩阵，生成建筑外轮廓
3. **布局生成模块** (`1_gh_room_layout_generator.py`)：基于边界自动生成房间布局
4. **网格处理模块** (`2_gh_room_layout_grid_processor.py`)：处理生成的房间布局网格
5. **边缘预测模块** (`3_gh_edge_type_predictor.py`)：预测建筑边缘类型
6. **图形可视化模块** (`4_gh_graph_visualizer.py`)：生成可视化的建筑布局图形
7. **主控制模块** (`5_gh.py`)：协调各个模块的工作流程

## 技术栈

### 前端部分
- HTML5：页面结构
- CSS3：样式设计
- JavaScript (ES6+)：交互逻辑

### 后端部分
- Python 3.10+：核心算法实现
- 相关Python库：NumPy, NetworkX, Matplotlib, PyTorch, PyTorch Geometric等

### 3D设计软件依赖
- Rhino 7+：3D建模环境
- Grasshopper：Rhino插件，用于可视化编程和几何处理

### AI技术
- 图神经网络 (GNN)：用于边缘类型预测
- 大语言模型 (LLM)：辅助边缘类型分类

## 项目结构

```
自动化建筑布局生成工具/
├── README.md                          # 项目说明文档
├── matrix_generator.html              # 可视化边界绘制界面
├── 0_gh_boundary_mask_generator.py    # 边界处理模块
├── 1_gh_room_layout_generator.py      # 布局生成模块
├── 2_gh_room_layout_grid_processor.py # 网格处理模块
├── 3_gh_edge_type_predictor.py        # 边缘预测模块
├── 4_gh_graph_visualizer.py           # 图形可视化模块
├── 5_gh.py                            # 主控制模块
├── generate_prompt.py                 # 提示词生成模块
└── gh_edge_type_predictor_combined.py # 组合边缘类型预测模块
```

## 使用说明

### 1. 绘制建筑边界

1. 使用浏览器打开 `matrix_generator.html` 文件
2. 在网格上左键点击/拖拽绘制建筑边界
3. 右键点击/拖拽擦除边界
4. 点击"预览内部区域"查看围合的内部区域
5. 边界闭合后，点击"导出 JSON"按钮生成边界矩阵文件

### 2. 生成建筑布局

#### 2.1 Python环境运行

1. 确保已安装Python 3.10+环境
2. 确保已安装所需的Python库：NumPy, NetworkX, Matplotlib, PyTorch, PyTorch Geometric等
3. 运行主控制脚本：
   ```bash
   python 5_gh.py
   ```
4. 按照提示输入边界矩阵文件路径
5. 系统将自动生成房间布局和边缘类型预测
6. 生成的布局图形将保存到当前目录

#### 2.2 Rhino/Grasshopper环境运行（**注意：工作流尚未完全搭建**）

1. 确保已安装Rhino 7+和Grasshopper插件
2. 打开Rhino软件，新建或打开一个Rhino文件
3. 进入Grasshopper环境
4. 导入相关的Python脚本到Grasshopper画布
5. 连接各模块，构建工作流
6. 按照提示输入边界矩阵文件路径
7. 系统将在Rhino视图中生成可视化的建筑布局

**重要提示**：Rhino中的工作流目前尚未完全搭建完成，部分功能可能无法正常使用。请关注项目更新，等待完整的Rhino工作流实现。

### 3. 单独运行各模块

您也可以根据需要单独运行各个模块：

```bash
# 处理边界矩阵
python 0_gh_boundary_mask_generator.py

# 生成房间布局
python 1_gh_room_layout_generator.py

# 处理房间布局网格
python 2_gh_room_layout_grid_processor.py

# 预测边缘类型
python 3_gh_edge_type_predictor.py

# 可视化建筑布局
python 4_gh_graph_visualizer.py
```

### 4. Rhino/Grasshopper环境使用注意事项

1. **依赖要求**：必须安装Rhino 7+和Grasshopper插件
2. **文件路径**：所有文件必须放在同一目录下，使用相对路径
3. **工作流状态**：当前Rhino工作流尚未完全搭建，部分功能可能无法正常运行
4. **调试建议**：在使用前，建议先在Python环境中测试各模块功能
5. **版本兼容**：确保Python版本与Rhino内置Python版本兼容

**警告**：由于Rhino工作流尚未完全搭建，强烈建议在Python环境中进行初步测试和使用，待Rhino工作流完善后再尝试在Rhino/Grasshopper环境中使用。

## 如何运行

### 本地运行

1. 克隆或下载项目到本地
2. 前端部分：直接用浏览器打开 `matrix_generator.html`
3. 后端部分：
   ```bash
   # 确保已安装Python 3.10+
   python --version
   
   # 运行主控制脚本
   python 5_gh.py
   ```

### 使用HTTP服务器（推荐）

对于前端部分，使用HTTP服务器可以获得更好的体验：

```bash
# 使用Python启动HTTP服务器
python -m http.server 8000

# 或使用Node.js的http-server
npx http-server -p 8000
```

然后在浏览器中访问 `http://localhost:8000/matrix_generator.html`

## 输入输出格式

### 输入：边界矩阵

边界矩阵是一个二维数组，其中：
- `1` 表示建筑边界
- `0` 表示空白区域

示例：
```json
[
  [0, 0, 0, 0, 0],
  [0, 1, 1, 1, 0],
  [0, 1, 0, 1, 0],
  [0, 1, 1, 1, 0],
  [0, 0, 0, 0, 0]
]
```

### 输出：建筑布局

系统将生成多种输出格式，包括：
- 可视化图形文件
- 房间布局矩阵
- 边缘类型预测结果
- 图结构数据

## 算法说明

### 边界闭合检测
使用洪水填充算法从矩阵边缘开始填充，检测是否存在未被填充的内部区域，从而判断边界是否闭合。

### 房间布局生成
基于生成式算法，结合建筑设计原则，自动生成合理的房间布局。算法考虑了房间大小、功能分区和空间关系等因素。

### 边缘类型预测
采用图神经网络 (GNN) 结合大语言模型 (LLM) 的方法，自动预测建筑边缘类型，提高布局方案的准确性和合理性。

## 应用场景

- 建筑方案初始设计阶段
- 快速生成多种布局方案进行比较
- 教学和研究用途
- 辅助建筑师和设计师提高工作效率

## 许可证

MIT License

## 贡献

欢迎提交Issue和Pull Request！

## 更新日志

### v1.0.0
- 初始版本发布
- 实现可视化边界绘制功能
- 实现自动化布局生成算法
- 实现AI驱动的边缘类型预测
- 提供完整的使用文档

## 联系方式

- 项目地址：[GitHub Repository URL]
- 作者：Mentat
- 邮箱：uran0831@qq.com

---

**感谢使用自动化建筑布局生成工具！** 🎉
