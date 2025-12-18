# 导入数值计算库，用于数组操作
import numpy as np
# 导入PyTorch深度学习框架
import torch
# 导入PyTorch功能模块，包含各种激活函数和损失函数
import torch.nn.functional as F
# 从PyTorch Geometric导入Data类，用于处理图数据
from torch_geometric.data import Data
# 从PyTorch Geometric导入图卷积层和批量归一化层
from torch_geometric.nn import SAGEConv, BatchNorm
# 从PyTorch导入1D批量归一化层
from torch.nn import BatchNorm1d
# 导入JSON库，用于数据序列化和反序列化
import json
# 导入随机数库，用于随机操作
import random
# 导入绘图库，用于数据可视化
import matplotlib.pyplot as plt
# 导入操作系统库，用于文件和目录操作
import os

# 设置设备：优先使用GPU（如果可用），否则使用CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# 全局输入文件路径：节点属性和边缘连接数据
INPUT_FILE_PATH = "node_attr_edge_conn_4GEN.json"
# 全局模型路径：预训练的GNN模型权重文件
MODEL_PATH = "best_model_HRL_GNN_NJU25F_1019_good.pth"


# 1. 数据加载函数
def load_node_attributes_and_edge_connections(file_path):
    """
    从JSON文件加载节点属性和边缘连接数据
    
    参数:
        file_path: 包含节点属性和边缘连接的JSON文件路径
        
    返回:
        node_attrs: 节点属性数组，形状为 (num_nodes, num_features)
        edge_conn: 边缘连接数组，形状为 (2, num_edges)，第一行为源节点索引，第二行为目标节点索引
    """
    try:
        # 读取JSON文件
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        print(f"Successfully loaded file: {file_path}")
        print(f"Keys in file: {list(data.keys())}")

        # 转换节点属性为NumPy数组
        node_attrs = np.array(data['node_attrs'])
        print(f"Successfully loaded node attributes from {file_path}")
        print(f"Node attributes shape: {node_attrs.shape}")
        
        # 提取边缘连接数据
        source_nodes = data['edge_conn'][0]
        target_nodes = data['edge_conn'][1]
        edge_conn = np.array([source_nodes, target_nodes])
        print(f"Successfully loaded edge connections from {file_path}")
        print(f"Edge connections shape: {edge_conn.shape}")
        
        return node_attrs, edge_conn
        
    except FileNotFoundError:
        print(f"Error: File {file_path} does not exist")
        return None, None
    except json.JSONDecodeError:
        print(f"Error: File {file_path} is not valid JSON format")
        return None, None
    except Exception as e:
        print(f"Error occurred while reading file: {str(e)}")
        return None, None

def create_data_object(node_attrs, edge_conn):
    """
    从节点属性和边缘连接创建PyTorch Geometric Data对象
    
    参数:
        node_attrs: 节点属性数组
        edge_conn: 边缘连接数组
        
    返回:
        data: PyTorch Geometric Data对象，包含节点特征和边缘索引
    """
    # 将节点属性转换为PyTorch张量
    x = torch.FloatTensor(node_attrs)
    # 将边缘连接转换为PyTorch张量
    edge_index = torch.LongTensor(edge_conn)
    
    # 归一化节点坐标（仅前2个属性）
    x[:, 0] = x[:, 0] / 200.0  # x坐标归一化到0-1范围
    x[:, 1] = x[:, 1] / 200.0  # y坐标归一化到0-1范围
    
    # 创建双向边缘索引：原始边缘 + 反向边缘
    edge_index_bidir = torch.cat([edge_index, torch.stack([edge_index[1], edge_index[0]], dim=0)], dim=1)
    
    # 创建Data对象
    data = Data(x=x, edge_index=edge_index_bidir)
    
    # 打印数据信息
    print(f"Data created successfully!")
    print(f"Number of nodes: {data.num_nodes}")
    print(f"Number of edges: {data.num_edges}")
    print(f"Number of node features: {data.num_node_features}")
    
    return data

# 2. 模型定义
class Encoder(torch.nn.Module):
    """图变分自编码器的编码器网络"""
    def __init__(self, feature_size, latent_dim=32):
        """
        初始化编码器网络
        
        参数:
            feature_size: 节点特征维度
            latent_dim: 潜在空间维度，默认32
        """
        super(Encoder, self).__init__()
        
        # 编码器嵌入尺寸
        encoder_embedding_size = 256
        
        # 定义图卷积层和批量归一化层
        self.conv1 = SAGEConv(feature_size, encoder_embedding_size)
        self.bn1 = BatchNorm(encoder_embedding_size)
        self.conv2 = SAGEConv(encoder_embedding_size, encoder_embedding_size*2)
        self.bn2 = BatchNorm(encoder_embedding_size*2)
        self.conv3 = SAGEConv(encoder_embedding_size*2, encoder_embedding_size*4)
        self.bn3 = BatchNorm(encoder_embedding_size*4)
        self.conv4 = SAGEConv(encoder_embedding_size*4, encoder_embedding_size*4)
        self.bn4 = BatchNorm(encoder_embedding_size*4)
        self.conv5 = SAGEConv(encoder_embedding_size*4, encoder_embedding_size*2)
        self.bn5 = BatchNorm(encoder_embedding_size*2)
        self.conv6 = SAGEConv(encoder_embedding_size*2, encoder_embedding_size)
        self.bn6 = BatchNorm(encoder_embedding_size)
        self.conv7 = SAGEConv(encoder_embedding_size, encoder_embedding_size//2)
        self.bn7 = BatchNorm(encoder_embedding_size//2)

        # 均值和对数方差层
        self.conv_mu = SAGEConv(encoder_embedding_size//2, latent_dim*2)
        self.conv_logvar = SAGEConv(encoder_embedding_size//2, latent_dim*2)

        # Dropout层，防止过拟合
        self.dropout = torch.nn.Dropout(p=0.3)

    def forward(self, x, edge_index):
        """
        编码器前向传播
        
        参数:
            x: 节点特征张量
            edge_index: 边缘索引张量
            
        返回:
            mu: 潜在空间均值
            logvar: 潜在空间对数方差
        """
        x = F.relu(self.conv1(x, edge_index))  # 第一层图卷积 + ReLU激活
        x = self.bn1(x)  # 批量归一化
        x = self.dropout(x)  # Dropout
        
        x = F.relu(self.conv2(x, edge_index))  # 第二层图卷积
        x = self.bn2(x)
        x = self.dropout(x)
        
        x = F.relu(self.conv3(x, edge_index))  # 第三层图卷积
        x = self.bn3(x)
        x = self.dropout(x)
        
        x = F.relu(self.conv4(x, edge_index))  # 第四层图卷积
        x = self.bn4(x)
        x = self.dropout(x)
        
        x = F.relu(self.conv5(x, edge_index))  # 第五层图卷积
        x = self.bn5(x)
        x = self.dropout(x)
        
        x = F.relu(self.conv6(x, edge_index))  # 第六层图卷积
        x = self.bn6(x)
        x = self.dropout(x)
        
        x = F.relu(self.conv7(x, edge_index))  # 第七层图卷积
        x = self.bn7(x)
        x = self.dropout(x)

        # 计算潜在空间均值和对数方差
        mu = self.conv_mu(x, edge_index)
        logvar = self.conv_logvar(x, edge_index)
        return mu, logvar

class Decoder(torch.nn.Module):
    """图变分自编码器的解码器网络"""

    def __init__(self, latent_dim=32):
        """
        初始化解码器网络
        
        参数:
            latent_dim: 潜在空间维度，默认32
        """
        super(Decoder, self).__init__()
        
        # 解码器隐藏层尺寸
        decoder_size = 128
        # 定义全连接层和批量归一化层
        self.edge_attr_decoder1 = torch.nn.Linear(latent_dim*2, decoder_size)
        self.decoder_bn_1 = BatchNorm1d(decoder_size)
        self.edge_attr_decoder2 = torch.nn.Linear(decoder_size, decoder_size)
        self.decoder_bn_2 = BatchNorm1d(decoder_size)
        self.edge_attr_decoder3 = torch.nn.Linear(decoder_size, decoder_size//2)
        self.decoder_bn_3 = BatchNorm1d(decoder_size//2)
        self.edge_attr_decoder4 = torch.nn.Linear(decoder_size//2, 6)  # 输出6种边缘类型
        self.dropout = torch.nn.Dropout(p=0.3)

    def forward(self, z, edge_index):
        """
        解码器前向传播
        
        参数:
            z: 潜在空间表示
            edge_index: 边缘索引张量
            
        返回:
            edge_attr_recon: 重构的边缘属性
        """
        # 将潜在表示移动到指定设备
        z = z.to(device)
        # 计算边缘两端节点的潜在表示之差
        z_diff = z[edge_index[0]] - z[edge_index[1]]
        
        # 全连接层 + ReLU激活
        x = F.relu(self.decoder_bn_1(self.edge_attr_decoder1(z_diff)))
        x = self.dropout(x)  # Dropout
        x = F.relu(self.decoder_bn_2(self.edge_attr_decoder2(x)))
        x = self.dropout(x)
        x = F.relu(self.decoder_bn_3(self.edge_attr_decoder3(x)))
        # 最后一层不使用激活函数，直接输出logits
        edge_attr_recon = self.edge_attr_decoder4(x)
        
        return edge_attr_recon

class GVAE(torch.nn.Module):
    """图变分自编码器 (Graph Variational Autoencoder)"""

    def __init__(self, input_dim, edge_attr_dim):
        """
        初始化图变分自编码器
        
        参数:
            input_dim: 节点特征维度
            edge_attr_dim: 边缘属性维度（类别数）
        """
        super(GVAE, self).__init__()
        # 初始化编码器
        self.encoder = Encoder(input_dim)
        # 初始化解码器，使用默认的latent_dim=32
        self.decoder = Decoder()
        
    def reparameterize(self, mu, logvar):
        """
        重参数化技巧：从正态分布中采样潜在表示
        
        参数:
            mu: 潜在空间均值
            logvar: 潜在空间对数方差
            
        返回:
            z: 采样得到的潜在表示
        """
        # 计算标准差
        std = torch.exp(0.5 * logvar)
        # 从标准正态分布中采样
        eps = torch.randn_like(std)
        # 重参数化采样
        return mu + eps * std
    
    def forward(self, x, edge_index):
        """
        图变分自编码器前向传播
        
        参数:
            x: 节点特征张量
            edge_index: 边缘索引张量
            
        返回:
            reconstructed_edge_attr: 重构的边缘属性
            mu: 潜在空间均值
            logvar: 潜在空间对数方差
        """
        # 编码：获取潜在空间均值和对数方差
        mu, logvar = self.encoder(x, edge_index)
        # 重参数化采样
        z = self.reparameterize(mu, logvar)
        # 解码：重构边缘属性
        reconstructed_edge_attr = self.decoder(z, edge_index)
        return reconstructed_edge_attr, mu, logvar

# 3. 模型加载函数
def load_model():
    """
    加载预训练模型
    
    返回:
        model: 加载了预训练权重的GVAE模型，或None（加载失败时）
    """
    # 定义模型参数
    input_dim = 18  # 节点特征维度
    edge_attr_dim = 6  # 边缘属性类别数
    
    # 创建模型实例并移动到指定设备
    model = GVAE(input_dim, edge_attr_dim).to(device)
    
    # 加载预训练模型权重
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print(f"Loaded trained model from {MODEL_PATH}")
        return model
    except FileNotFoundError:
        print(f"Error: Could not find the trained model file '{MODEL_PATH}'")
        print("Please train the model first using HRL_GNN_NJU25F_1019.py")
        return None

# 4. 单样本预测函数
def predict_single_sample(model, data, device, edge_conn):
    """
    预测单个样本并返回原始边缘的预测结果
    
    参数:
        model: 训练好的GVAE模型
        data: 图数据对象
        device: 计算设备
        edge_conn: 原始边缘连接数组
        
    返回:
        averaged_predictions: 原始边缘的预测类别
        averaged_probs: 原始边缘的预测概率
    """
    # 设置模型为评估模式
    model.eval()
    # 关闭梯度计算，节省内存并加速推理
    with torch.no_grad():
        # 将数据移动到指定设备
        data = data.to(device)
        # 前向传播，获取重构的边缘属性
        edge_attr_recon, mu, logvar = model(data.x, data.edge_index)
        # 计算边缘类型概率
        probs = F.softmax(edge_attr_recon, dim=1).cpu().numpy()

        # 平均双向边缘的概率
        averaged_probs = average_bidirectional_probabilities(probs, edge_conn)

        # 选择概率最大的类别作为预测结果
        averaged_predictions = np.argmax(averaged_probs, axis=1)
        return averaged_predictions, averaged_probs

def average_bidirectional_probabilities(probs, edge_conn):
    """
    平均双向边缘的概率，得到原始边缘的概率
    
    参数:
        probs: 所有双向边缘的概率，形状为 (440, 6)
        edge_conn: 原始边缘连接，形状为 (2, 220)
        
    返回:
        averaged_probs: 原始边缘的平均概率，形状为 (220, 6)
    """
    # 原始边缘数量
    num_original_edges = edge_conn.shape[1]
    
    # 分离原始边缘和反向边缘的概率
    original_probs = probs[:num_original_edges]
    reverse_probs = probs[num_original_edges:]
    
    # 平均双向边缘的概率
    averaged_probs = (original_probs + reverse_probs) / 2.0
    
    return averaged_probs

# 主程序执行
# 加载节点属性和边缘连接数据
node_attrs, edge_conn = load_node_attributes_and_edge_connections(INPUT_FILE_PATH)
# 创建图数据对象
data = create_data_object(node_attrs, edge_conn)
# 加载预训练模型
model = load_model()

# 进行单样本预测
pred_labels, probs = predict_single_sample(model, data, device, edge_conn)
# 打印预测结果信息
print(f"Predictions shape: {pred_labels.shape}")
print(f"Probabilities shape: {probs.shape}")

# 将预测结果转换为列表格式
edge_type_list = pred_labels.tolist()
# 打印边缘类型列表
print(edge_type_list)