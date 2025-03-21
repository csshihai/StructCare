import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_scatter import scatter_mean


class GATWithEdgeEmbedding(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim, output_dim, heads=8):
        super(GATWithEdgeEmbedding, self).__init__()
        # 节点和边的特征转换层
        self.edge_linear = nn.Linear(edge_dim, node_dim)  # 将边特征转化为与节点特征同维度
        self.gat1 = GATConv(node_dim, hidden_dim, heads=heads, concat=True)  # 第一层 GAT
        self.gat2 = GATConv(hidden_dim * heads, output_dim, heads=1, concat=False)  # 第二层 GAT

    def forward(self, x, edge_index, edge_attr):
        # 将边特征映射到与节点特征相同的维度
        edge_embedding = self.edge_linear(edge_attr)

        # print("Edge Index:", edge_index)
        # print("Edge Index Shape:", edge_index.shape)
        # 聚合边嵌入到相应的节点特征中
        src, dst = edge_index
        # 使用 scatter_mean 将边特征聚合到目标节点 (dst)!!!!!
        edge_to_node = scatter_mean(edge_embedding, dst, dim=0, dim_size=x.size(0))

        # 将聚合的边嵌入加到节点特征上
        x = x + edge_to_node

        # 第一层 GAT + 激活函数
        x = F.relu(self.gat1(x, edge_index))

        # 第二层 GAT
        x = self.gat2(x, edge_index)

        # 对所有节点进行平均池化，获取全局图表示
        graph_embedding = x.mean(dim=0)

        return graph_embedding
# 1层 GAT
class GATWithEdgeEmbedding_1Layer(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim, output_dim, heads=8):
        super(GATWithEdgeEmbedding_1Layer, self).__init__()
        # 边特征转换层
        self.edge_linear = nn.Linear(edge_dim, node_dim)  # 将边特征转化为与节点特征同维度
        # 只使用一层GAT
        self.gat1 = GATConv(node_dim, output_dim, heads=1, concat=False)  # 只有1层GAT

    def forward(self, x, edge_index, edge_attr):
        # 将边特征映射到与节点特征相同的维度
        edge_embedding = self.edge_linear(edge_attr)

        # 聚合边嵌入到目标节点特征
        src, dst = edge_index
        edge_to_node = scatter_mean(edge_embedding, dst, dim=0, dim_size=x.size(0))

        # 将聚合的边嵌入加到节点特征上
        x = x + edge_to_node

        # 第一层 GAT
        x = self.gat1(x, edge_index)

        # 返回图表示
        graph_embedding = x.mean(dim=0)
        return graph_embedding


# 4层 GAT
class GATWithEdgeEmbedding_4Layer(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim, output_dim, heads=8):
        super(GATWithEdgeEmbedding_4Layer, self).__init__()
        # 边特征转换层
        self.edge_linear = nn.Linear(edge_dim, node_dim)  # 将边特征转化为与节点特征同维度
        # 4层GAT，每一层的输出为下一层的输入
        self.gat1 = GATConv(node_dim, hidden_dim, heads=heads, concat=True)  # 第一层GAT
        self.gat2 = GATConv(hidden_dim * heads, hidden_dim, heads=heads, concat=True)  # 第二层GAT
        self.gat3 = GATConv(hidden_dim * heads, hidden_dim, heads=heads, concat=True)  # 第三层GAT
        self.gat4 = GATConv(hidden_dim * heads, output_dim, heads=1, concat=False)  # 第四层GAT

    def forward(self, x, edge_index, edge_attr):
        # 将边特征映射到与节点特征相同的维度
        edge_embedding = self.edge_linear(edge_attr)

        # 聚合边嵌入到目标节点特征
        src, dst = edge_index
        edge_to_node = scatter_mean(edge_embedding, dst, dim=0, dim_size=x.size(0))

        # 将聚合的边嵌入加到节点特征上
        x = x + edge_to_node

        # 第一层 GAT + 激活函数
        x = F.relu(self.gat1(x, edge_index))

        # 第二层 GAT + 激活函数
        x = F.relu(self.gat2(x, edge_index))

        # 第三层 GAT + 激活函数
        x = F.relu(self.gat3(x, edge_index))

        # 第四层 GAT
        x = self.gat4(x, edge_index)

        # 对所有节点进行平均池化，获取全局图表示
        graph_embedding = x.mean(dim=0)

        return graph_embedding
def main():
    # 示例输入
    num_nodes = 5
    input_dim = 10
    hidden_dim = 8
    output_dim = 6
    edge_dim = 4  # 边特征的维度

    # 节点特征
    x = torch.randn(num_nodes, input_dim)
    # 边特征
    edge_attr = torch.randn(4, edge_dim)  # 对应每条边的特征
    # 边的索引
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]], dtype=torch.long)

    # 初始化 GATWithEdgeEmbedding 模型
    gat_model = GATWithEdgeEmbedding(input_dim, edge_dim, hidden_dim, output_dim)

    # 前向传播，获取图的全局表示
    graph_embedding = gat_model(x, edge_index, edge_attr)

    print("图的全局表示：", graph_embedding)

if __name__ == '__main__':
    main()
