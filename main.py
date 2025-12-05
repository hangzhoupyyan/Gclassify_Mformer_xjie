import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.data import Data
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
import math
from typing import List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")


class PositionalEncoding(nn.Module):
    """位置编码模块"""

    def __init__(self, d_model: int, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(0), :]


class MultiHeadAttention(nn.Module):
    """多头注意力机制"""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super(MultiHeadAttention, self).__init__()

        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len = query.size(0), query.size(1)

        # 线性变换
        Q = self.w_q(query).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        # 计算注意力得分
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # Softmax归一化
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 注意力加权
        attn_output = torch.matmul(attn_weights, V)

        # 拼接多头输出
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model)

        # 输出线性变换
        output = self.w_o(attn_output)

        return output


class SpectralClusteringGCN(nn.Module):
    """基于图卷积网络映射的谱聚类模块"""

    def __init__(self, input_dim: int, hidden_dims: List[int], num_clusters: int = 3):
        super(SpectralClusteringGCN, self).__init__()

        self.num_clusters = num_clusters

        # 构建GCN层
        gcn_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            gcn_layers.append(GCNConv(prev_dim, hidden_dim))
            prev_dim = hidden_dim

        self.gcn_layers = nn.ModuleList(gcn_layers)
        self.activation = nn.ReLU()

    def haversine_distance(self, lat1: torch.Tensor, lon1: torch.Tensor,
                           lat2: torch.Tensor, lon2: torch.Tensor) -> torch.Tensor:
        """计算哈弗辛距离"""
        R = 6371.0  # 地球半径

        lat1_rad = torch.deg2rad(lat1)
        lon1_rad = torch.deg2rad(lon1)
        lat2_rad = torch.deg2rad(lat2)
        lon2_rad = torch.deg2rad(lon2)

        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad

        a = torch.sin(dlat / 2) ** 2 + torch.cos(lat1_rad) * torch.cos(lat2_rad) * torch.sin(dlon / 2) ** 2
        c = 2 * torch.asin(torch.sqrt(a))

        return R * c

    def build_similarity_matrix(self, node_features: torch.Tensor,
                                coordinates: torch.Tensor) -> torch.Tensor:
        """构建相似度矩阵"""
        n_nodes = node_features.size(0)

        # 基于坐标计算距离
        similarity_matrix = torch.zeros((n_nodes, n_nodes))

        for i in range(n_nodes):
            for j in range(n_nodes):
                if i != j:
                    dist = self.haversine_distance(
                        coordinates[i, 0], coordinates[i, 1],
                        coordinates[j, 0], coordinates[j, 1]
                    )
                    similarity_matrix[i, j] = dist
                else:
                    similarity_matrix[i, j] = 0

        return similarity_matrix

    def compute_laplacian(self, similarity_matrix: torch.Tensor) -> torch.Tensor:
        """计算拉普拉斯矩阵"""
        degree_matrix = torch.diag(torch.sum(similarity_matrix, dim=1))
        laplacian = degree_matrix - similarity_matrix

        # 标准化拉普拉斯矩阵
        degree_sqrt_inv = torch.diag(1.0 / torch.sqrt(torch.diag(degree_matrix)))
        normalized_laplacian = degree_sqrt_inv @ laplacian @ degree_sqrt_inv

        return normalized_laplacian

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                coordinates: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播"""

        # 构建相似度矩阵和拉普拉斯矩阵
        similarity_matrix = self.build_similarity_matrix(x, coordinates)
        laplacian = self.compute_laplacian(similarity_matrix)

        # GCN特征映射
        for i, gcn_layer in enumerate(self.gcn_layers):
            x = gcn_layer(x, edge_index)
            if i < len(self.gcn_layers) - 1:
                x = self.activation(x)

        # 谱聚类
        try:
            # 使用k-means进行聚类
            kmeans = KMeans(n_clusters=self.num_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(x.detach().cpu().numpy())
            cluster_labels = torch.tensor(cluster_labels, device=device)
        except:
            # 如果聚类失败，返回随机标签
            cluster_labels = torch.randint(0, self.num_clusters, (x.size(0),), device=device)

        return x, cluster_labels


class EncoderLayer(nn.Module):
    """编码器层"""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super(EncoderLayer, self).__init__()

        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 自注意力
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))

        # 前馈网络
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))

        return x


class MultiEncoderInformer(nn.Module):
    """多编码器Informer模块"""

    def __init__(self, input_dim: int, d_model: int, n_heads: int,
                 d_ff: int, num_layers: int, num_sites: int, dropout: float = 0.1):
        super(MultiEncoderInformer, self).__init__()

        self.num_sites = num_sites
        self.d_model = d_model

        # 输入投影层
        self.input_projection = nn.Linear(input_dim, d_model)

        # 位置编码
        self.pos_encoding = PositionalEncoding(d_model)

        # 多个编码器（每个站点一个）
        self.encoders = nn.ModuleList([
            nn.ModuleList([EncoderLayer(d_model, n_heads, d_ff, dropout)
                           for _ in range(num_layers)])
            for _ in range(num_sites)
        ])

        # 图注意力网络融合
        self.gat_fusion = GATConv(d_model, d_model, heads=1, concat=False)

        # 输出层
        self.output_projection = nn.Linear(d_model, 1)  # 预测AQI值

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # x形状: [batch_size, seq_len, num_sites, input_dim]
        batch_size, seq_len, num_sites, input_dim = x.shape

        # 对每个站点的数据进行编码
        encoded_features = []
        for site_idx in range(num_sites):
            site_data = x[:, :, site_idx, :]  # [batch_size, seq_len, input_dim]

            # 输入投影和位置编码
            projected = self.input_projection(site_data)  # [batch_size, seq_len, d_model]
            encoded = self.pos_encoding(projected)

            # 通过编码器层
            for encoder_layer in self.encoders[site_idx]:
                encoded = encoder_layer(encoded)

            # 取最后一个时间步的特征
            last_step_features = encoded[:, -1, :]  # [batch_size, d_model]
            encoded_features.append(last_step_features)

        # 堆叠所有站点的特征
        all_features = torch.stack(encoded_features, dim=1)  # [batch_size, num_sites, d_model]

        # 使用GAT进行特征融合
        batch_size, num_sites, d_model = all_features.shape
        gat_input = all_features.reshape(-1, d_model)  # [batch_size * num_sites, d_model]

        # 扩展edge_index以处理批次数据
        batch_edge_index = []
        for batch_idx in range(batch_size):
            offset = batch_idx * num_sites
            batch_edges = edge_index + offset
            batch_edge_index.append(batch_edges)

        batch_edge_index = torch.cat(batch_edge_index, dim=1)

        # GAT融合
        fused_features = self.gat_fusion(gat_input, batch_edge_index)
        fused_features = fused_features.reshape(batch_size, num_sites, d_model)

        # 平均池化得到最终特征
        final_features = torch.mean(fused_features, dim=1)  # [batch_size, d_model]

        # 输出预测
        output = self.output_projection(final_features)  # [batch_size, 1]

        return output


class GCformer(nn.Module):
    """完整的GCformer模型"""

    def __init__(self, input_dim: int, gcn_hidden_dims: List[int], num_clusters: int,
                 d_model: int, n_heads: int, d_ff: int, num_encoder_layers: int,
                 num_sites: int, dropout: float = 0.1):
        super(GCformer, self).__init__()

        self.num_clusters = num_clusters
        self.num_sites = num_sites

        # 谱聚类模块
        self.spectral_clustering = SpectralClusteringGCN(
            input_dim, gcn_hidden_dims, num_clusters
        )

        # 为每个聚类创建单独的预测模型
        self.predictors = nn.ModuleList([
            MultiEncoderInformer(input_dim, d_model, n_heads, d_ff,
                                 num_encoder_layers, num_sites, dropout)
            for _ in range(num_clusters)
        ])

        # 分类到模型的映射
        self.cluster_to_model = {}

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                coordinates: torch.Tensor) -> torch.Tensor:
        """
        x: [batch_size, seq_len, num_sites, input_dim]
        edge_index: [2, num_edges]
        coordinates: [num_sites, 2] (纬度, 经度)
        """
        batch_size = x.size(0)

        # 重塑输入以进行聚类
        # 将时间序列数据视为图节点特征
        node_features = x.mean(dim=1)  # [batch_size, num_sites, input_dim]
        node_features = node_features.reshape(-1, node_features.size(-1))  # [batch_size * num_sites, input_dim]

        # 扩展坐标信息
        batch_coordinates = coordinates.unsqueeze(0).repeat(batch_size, 1, 1)
        batch_coordinates = batch_coordinates.reshape(-1, 2)  # [batch_size * num_sites, 2]

        # 扩展edge_index
        batch_edge_index = []
        for i in range(batch_size):
            offset = i * self.num_sites
            batch_edges = edge_index + offset
            batch_edge_index.append(batch_edges)

        batch_edge_index = torch.cat(batch_edge_index, dim=1)

        # 谱聚类
        _, cluster_labels = self.spectral_clustering(
            node_features, batch_edge_index, batch_coordinates
        )

        # 重塑聚类标签
        cluster_labels = cluster_labels.reshape(batch_size, self.num_sites)

        # 对每个批次中的主要聚类进行预测
        predictions = []
        for batch_idx in range(batch_size):
            # 找到该批次中最常见的聚类
            batch_clusters = cluster_labels[batch_idx]
            unique, counts = torch.unique(batch_clusters, return_counts=True)
            dominant_cluster = unique[torch.argmax(counts)].item()

            # 确保聚类到模型的映射存在
            if dominant_cluster not in self.cluster_to_model:
                self.cluster_to_model[dominant_cluster] = dominant_cluster % len(self.predictors)

            model_idx = self.cluster_to_model[dominant_cluster]

            # 使用对应的预测器进行预测
            batch_pred = self.predictors[model_idx](
                x[batch_idx:batch_idx + 1], edge_index
            )
            predictions.append(batch_pred)

        predictions = torch.cat(predictions, dim=0)
        return predictions


class AQIDataset(torch.utils.data.Dataset):
    """AQI数据集类"""

    def __init__(self, data: np.ndarray, seq_len: int, pred_len: int, num_sites: int):
        self.data = data
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.num_sites = num_sites
        self.total_len = seq_len + pred_len

    def __len__(self):
        return len(self.data) - self.total_len + 1

    def __getitem__(self, idx):
        # 获取输入序列和目标
        x = self.data[idx:idx + self.seq_len]  # [seq_len, num_sites, features]
        y = self.data[idx + self.seq_len:idx + self.total_len]  # [pred_len, num_sites, features]

        # 只取AQI值作为目标
        x_tensor = torch.FloatTensor(x)  # [seq_len, num_sites, features]
        y_tensor = torch.FloatTensor(y[:, :, 0])  # [pred_len, num_sites] 只取AQI值

        return x_tensor, y_tensor


def create_edge_index(num_sites: int, connection_radius: float = 100.0) -> torch.Tensor:
    """创建基于距离的边索引"""
    # 这里简化处理，实际应根据站点坐标计算
    edges = []
    for i in range(num_sites):
        for j in range(num_sites):
            if i != j:
                edges.append([i, j])

    if len(edges) == 0:
        edges = [[0, 1], [1, 0]]  # 默认边

    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def prepare_data(data_path: str, seq_len: int = 24, pred_len: int = 1,
                 batch_size: int = 32, num_sites: int = 4) -> Tuple[torch.utils.data.DataLoader, torch.Tensor]:
    """准备数据"""
    # 这里使用模拟数据，实际应加载真实数据
    # 模拟数据形状: [timesteps, num_sites, features]
    timesteps = 1000
    features = 5  # AQI + 气象特征

    # 生成模拟数据
    data = np.random.randn(timesteps, num_sites, features)

    # 创建数据集
    dataset = AQIDataset(data, seq_len, pred_len, num_sites)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    coordinates = torch.tensor([
        [39.9, 116.4],
        [30.2, 120.2],
        [22.5, 114.0],
        [29.5, 106.5]
    ], dtype=torch.float)

    return dataloader, coordinates


def train_model(model: nn.Module, dataloader: torch.utils.data.DataLoader,
                coordinates: torch.Tensor, num_epochs: int = 100):
    """训练模型"""
    model.to(device)
    coordinates = coordinates.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # 创建边索引
    edge_index = create_edge_index(model.num_sites).to(device)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch_idx, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)

            # 前向传播
            pred = model(x, edge_index, coordinates)

            # 计算损失
            loss = criterion(pred.squeeze(), y[:, -1, :].mean(dim=1))  # 预测最后一个时间步

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f'Epoch: {epoch + 1}, Batch: {batch_idx}, Loss: {loss.item():.4f}')

        avg_loss = total_loss / len(dataloader)
        print(f'Epoch: {epoch + 1}, Average Loss: {avg_loss:.4f}')


def evaluate_model(model: nn.Module, test_dataloader: torch.utils.data.DataLoader,
                   coordinates: torch.Tensor):
    """评估模型"""
    model.eval()
    coordinates = coordinates.to(device)
    edge_index = create_edge_index(model.num_sites).to(device)

    predictions = []
    targets = []

    with torch.no_grad():
        for x, y in test_dataloader:
            x, y = x.to(device), y.to(device)

            pred = model(x, edge_index, coordinates)
            predictions.append(pred.cpu().numpy())
            targets.append(y[:, -1, :].mean(dim=1).cpu().numpy())

    predictions = np.concatenate(predictions)
    targets = np.concatenate(targets)

    # 计算评估指标
    mae = np.mean(np.abs(predictions - targets))
    rmse = np.sqrt(np.mean((predictions - targets) ** 2))
    smape = 100 * np.mean(2 * np.abs(predictions - targets) / (np.abs(predictions) + np.abs(targets) + 1e-8))

    print(f'MAE: {mae:.4f}, RMSE: {rmse:.4f}, SMAPE: {smape:.4f}%')
    return mae, rmse, smape


def main():
    """主函数"""
    # 超参数
    seq_len = 24
    pred_len = 1
    batch_size = 32
    num_sites = 4
    num_epochs = 50

    # 准备数据
    print("准备数据...")
    train_dataloader, coordinates = prepare_data("simulated_data.npy", seq_len, pred_len, batch_size, num_sites)

    # 初始化模型
    print("初始化模型...")
    model = GCformer(
        input_dim=5,  # AQI + 气象特征
        gcn_hidden_dims=[64, 32, 16],
        num_clusters=3,
        d_model=64,
        n_heads=4,
        d_ff=128,
        num_encoder_layers=2,
        num_sites=num_sites,
        dropout=0.1
    )

    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 训练模型
    print("开始训练...")
    train_model(model, train_dataloader, coordinates, num_epochs)

    # 评估模型
    print("评估模型...")
    test_dataloader, _ = prepare_data("simulated_data.npy", seq_len, pred_len, batch_size, num_sites)
    evaluate_model(model, test_dataloader, coordinates)


if __name__ == "__main__":
    main()

import matplotlib.pyplot as plt
import seaborn as sns


def visualize_predictions(model: nn.Module, dataloader: torch.utils.data.DataLoader,
                          coordinates: torch.Tensor, num_samples: int = 10):
    """可视化预测结果"""
    model.eval()
    coordinates = coordinates.to(device)
    edge_index = create_edge_index(model.num_sites).to(device)

    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.ravel()

    with torch.no_grad():
        for i, (x, y) in enumerate(dataloader):
            if i >= num_samples:
                break

            x, y = x.to(device), y.to(device)
            pred = model(x, edge_index, coordinates)

            # 转换为numpy
            pred_np = pred.cpu().numpy().flatten()
            target_np = y[:, -1, :].mean(dim=1).cpu().numpy()

            # 绘制散点图
            axes[i].scatter(target_np, pred_np, alpha=0.6)
            axes[i].plot([target_np.min(), target_np.max()],
                         [target_np.min(), target_np.max()], 'r--')
            axes[i].set_xlabel('真实值')
            axes[i].set_ylabel('预测值')
            axes[i].set_title(f'样本 {i + 1}')

    plt.tight_layout()
    plt.show()


def plot_training_loss(loss_history: list):
    """绘制训练损失曲线"""
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('训练损失曲线')
    plt.grid(True)
    plt.show()