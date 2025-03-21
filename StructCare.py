
import numpy as np
import torch
from cuml.cluster import HDBSCAN
import cupy as cp
from cuml.cluster import KMeans
from torch_geometric.data import Data
from Graph_Features_Learning import GATWithEdgeEmbedding
import torch.nn as nn
import torch.nn


def concatenate_patient_embeddings(patient_emb_dict):
    """
    将 patient_emb_dict 中的所有张量在最后一个维度上拼接起来。

    参数:
    patient_emb_dict (dict of torch.Tensor): 包含每个特征的患者嵌入字典

    返回:
    torch.Tensor: 拼接后的张量
    """
    # 提取所有张量
    emb_list = list(patient_emb_dict.values())
    # 在最后一个维度上拼接
    patient_emb = torch.cat(emb_list, dim=-1)
    return patient_emb

def pad_tensors(tensors):
    """
    将不同尺寸的tensor填充成相同尺寸的大tensor。

    参数:
    tensors (list of torch.Tensor): 输入的多个张量列表，每个张量的形状为 (visit, monitor, dim)

    返回:
    torch.Tensor: 填充后的大张量，形状为 (batch, max_visits, max_monitors, dim)
    """
    # 找到各个维度的最大值
    max_visits = max(t.size(0) for t in tensors)
    max_monitors = max(t.size(1) for t in tensors)
    dim = tensors[0].size(2)

    # 初始化一个大的tensor，用0填充
    batch_size = len(tensors)
    padded_tensor = torch.zeros((batch_size, max_visits, max_monitors, dim))

    # 将各个tensor填充到大tensor中
    for i, tensor in enumerate(tensors):
        v, m, d = tensor.size()
        padded_tensor[i, :v, :m, :] = tensor

    return padded_tensor





def get_rel_emb(map_cluster_rel):
    rel_emb = []

    for i in range(len(map_cluster_rel.keys())):
        rel_emb.append(map_cluster_rel[str(i)]['embedding'][0])

    rel_emb = np.array(rel_emb)
    return torch.tensor(rel_emb)
def get_node_emb(map_cluster):
    node_emb = []

    for i in range(len(map_cluster.keys())):
        node_emb.append(map_cluster[str(i)]['embedding'][0])  # 假设节点嵌入存储在 'embedding' 字段中

    node_emb = np.array(node_emb)
    return torch.tensor(node_emb)





def cluster_nodes(node_features,min_samples,min_cluster_size):
    # 如果 node_features 是 PyTorch 张量，先转换为 CuPy 数组
    if isinstance(node_features, torch.Tensor):
        node_features_cpu = node_features.detach().cpu()  # 移到CPU并分离梯度
        node_features_gpu = cp.asarray(node_features_cpu)  # 转换为CuPy数组
    else:
        node_features_gpu = node_features  # 如果已是 CuPy 数组，直接使用

    # 执行 DBSCAN 聚类
    dbscan = HDBSCAN(min_samples=min_samples,min_cluster_size=min_cluster_size)
    cluster_labels = dbscan.fit_predict(node_features_gpu)
    membership_strengths =dbscan.probabilities_
    # 返回聚类标签
    return cluster_labels, membership_strengths

# def cluster_nodes(node_features, n_clusters):
#     # 将整个特征张量转移到 GPU
#     node_features_gpu = cp.asarray(node_features.cpu().detach().numpy())
#
#     # 执行 KMeans 聚类
#     kmeans = KMeans(n_clusters=n_clusters)
#     kmeans.fit(node_features_gpu)
#
#     # 获取聚类标签
#     cluster_labels = kmeans.predict(node_features_gpu)
#
#     # 返回标签
#     return cluster_labels


### 开始
class StructCare(nn.Module):
    def __init__(
            self,
            num_nodes,
            num_rels,
            Tokenizers_visit_event,
            Tokenizers_monitor_event,
            output_size,
            device,
            freeze=False,
            embedding_dim=384,
            node_emb=None,
            rel_emb=None,
            dropout=0.7,

    ):
        super(StructCare, self).__init__()
        # 初始化一些参数，常规操作
        self.embedding_dim = embedding_dim
        # 三个字典分别是老三样con、diagnose、drug
        self.visit_event_token = Tokenizers_visit_event
        self.feature_visit_event_keys = Tokenizers_visit_event.keys()
        self.monitor_event_token = Tokenizers_monitor_event
        self.feature_monitor_event_keys = Tokenizers_monitor_event.keys()
        self.dropout = torch.nn.Dropout(p=dropout)

        self.device = device

        self.embeddings = nn.ModuleDict()

        # 为每种event添加一种嵌入
        for feature_key in self.feature_visit_event_keys:
            tokenizer = self.visit_event_token[feature_key]
            self.embeddings[feature_key] = nn.Embedding(
                tokenizer.get_vocabulary_size(),
                self.embedding_dim,
                padding_idx=tokenizer.get_padding_index(),
            )
        for feature_key in self.feature_monitor_event_keys:
            tokenizer = self.monitor_event_token[feature_key]
            self.embeddings[feature_key] = nn.Embedding(
                tokenizer.get_vocabulary_size(),
                self.embedding_dim,
                padding_idx=tokenizer.get_padding_index(),
            )

        self.num_nodes = num_nodes
        self.num_rels = num_rels
        # 初始化大图G的边和节点的嵌入
        if node_emb is None:
            self.node_emb = nn.Embedding(num_nodes, embedding_dim)
        else:
            self.node_emb = nn.Embedding.from_pretrained(node_emb, freeze=freeze)

        if rel_emb is None:
            self.rel_emb = nn.Embedding(num_rels, embedding_dim)
        else:
            self.rel_emb = nn.Embedding.from_pretrained(rel_emb, freeze=freeze)
        self.visit_gru = nn.ModuleDict()
        # 为每种visit_event添加一种gru
        for feature_key in self.feature_visit_event_keys:
            self.visit_gru[feature_key] = torch.nn.GRU(self.embedding_dim, self.embedding_dim, batch_first=True)
            self.visit_gru[feature_key].flatten_parameters()  # 在每次前向传播之前调用
        for feature_key in self.feature_monitor_event_keys:
            self.visit_gru[feature_key] = torch.nn.GRU(self.embedding_dim, self.embedding_dim, batch_first=True)
        # 实际并没有用，忘记注释了
        for feature_key in ['weight', 'age']:
            self.visit_gru[feature_key] = torch.nn.GRU(self.embedding_dim, self.embedding_dim, batch_first=True)
            self.visit_gru[feature_key].flatten_parameters()  # 在每次前向传播之前调用

        # 实际并没有用,可以加上但不影响最后结果
        self.fc_age = nn.Linear(1, self.embedding_dim)
        self.fc_weight = nn.Linear(1, self.embedding_dim)

        self.graph_visit_Learning = torch.nn.GRU(self.embedding_dim, self.embedding_dim, batch_first=True)
        self.graph_visit_Learning.flatten_parameters()
        self.graph_monitor_Learning = torch.nn.GRU(self.embedding_dim, self.embedding_dim, batch_first=True)
        # 这里写关于图的表征学习的初始化
        self.GATWithEdgeEmbedding = GATWithEdgeEmbedding(768, embedding_dim, embedding_dim, embedding_dim,
                                                         heads=8)  #node_dim, edge_dim, hidden_dim, output_dim
        self.linear_layer = nn.Linear(768, 768)
        # 得到结果前一步的传递
        item_num = 6
        self.fc_patient = nn.Sequential(
            torch.nn.ReLU(),
            nn.Linear(item_num * self.embedding_dim, output_size)
        )

    def generate_graph_based_on_monitor(self, original_graph, monitor_info, node_emb):
        """
        根据给定的 monitor 信息生成新的图，通过结合节点嵌入和 monitor 信息更新节点特征，并进行聚类。

        :param original_graph: 原始的 visit 图，包含节点索引和边信息
        :param monitor_info: 当前 monitor 的信息，形状为 (1, 384)
        :param node_emb: 节点嵌入信息
        :return: 基于 monitor 信息生成的新图
        """

        # 拼接节点嵌入和 monitor 信息
        monitor_info_repeated = monitor_info.repeat(node_emb.size(0), 1)
        node_monitor_features = torch.cat([node_emb, monitor_info_repeated], dim=-1)
        node_monitor_features = self.linear_layer(node_monitor_features)

        # 聚类操作（假设 cluster_nodes 函数已定义）
        cluster_labels, membership_strengths = cluster_nodes(node_monitor_features,5,10)
        cluster_labels = torch.tensor(cluster_labels)  # 将 NumPy 数组转换为 PyTorch 张量

        # 获取唯一的聚类ID
        unique_clusters = cluster_labels.unique()

        cluster_centers = []
        new_node_indices = {}

        for cluster_id in unique_clusters:
            if cluster_id == -1:  # DBSCAN中的噪声标记为-1
                continue

            cluster_indices = (cluster_labels == cluster_id).nonzero(as_tuple=True)[0]  # 获取当前聚类的索引

            if cluster_indices.numel() > 0:  # 确保聚类不为空
                if cluster_indices.numel() == 1:
                    # 如果只有一个节点，直接使用该节点的特征作为聚类中心
                    cluster_center = node_monitor_features[cluster_indices.item()]
                else:
                    # 多个节点时计算平均值
                    cluster_center = node_monitor_features[cluster_indices].mean(dim=0)

                cluster_centers.append(cluster_center)
                # 更新新的节点索引映射
                new_index = len(cluster_centers) - 1
                for index in cluster_indices:
                    new_node_indices[index.item()] = new_index

        # 若有聚类中心，生成新节点特征；否则保留原始特征
        if len(cluster_centers) > 0:
            new_node_features = torch.stack(cluster_centers)
        else:
            # 返回与原始节点特征同样形状的全零张量
            new_node_features = torch.zeros_like(node_monitor_features)  # 保持形状和 dtype 一致

        # 更新边的索引和关系
        new_edge_index = {}
        for edge, rel in zip(original_graph.edge_index.t().tolist(), original_graph.relation.tolist()):
            source, target = edge

            # 获取源节点和目标节点的聚类标签
            source_cluster = new_node_indices.get(source)
            target_cluster = new_node_indices.get(target)

            # 只保留不同聚类之间的边
            if source_cluster is not None and target_cluster is not None and source_cluster != target_cluster:
                edge_key = (min(source_cluster, target_cluster), max(source_cluster, target_cluster))
                if edge_key not in new_edge_index:
                    new_edge_index[edge_key] = []  # 初始化边的关系列表
                new_edge_index[edge_key].append(rel)  # 添加关系

        # 创建新的边列表和合并后的关系
        edge_index_list = []
        relation_list = []
        for (source, target), rel_list in new_edge_index.items():
            edge_index_list.append([source, target])
            # 合并关系，例如取最大值或平均值
            # merged_relation = max(rel_list)  # 示例：取最大关系

            # merged_relation = torch.mean(rel_list)  # 示例：取平均关系
            merged_relation = max(rel_list)   # 示例：取加和关系
            relation_list.append(merged_relation)

        # 创建新图
        new_graph = Data(
            x=new_node_features,
            edge_index=torch.tensor(edge_index_list, dtype=torch.long).t().contiguous(),
            relation=torch.tensor(relation_list, dtype=torch.long)
        )

        return new_graph, cluster_labels

    def calculate_graph_clustering_loss(self, node_embeddings, cluster_labels):
        """
        计算无监督聚类损失，适用于 DBSCAN。

        :param node_embeddings: 节点的嵌入，形状为 (num_nodes, embedding_dim)
        :param cluster_labels: 每个节点的聚类标签，形状为 (num_nodes,)
        :return: 计算得到的损失值
        """
        n_clusters = cluster_labels.max().item() + 1  # 聚类数量
        loss = 0.0
        cluster_count = 0  # 记录有效聚类的数量

        # 遍历每个聚类
        for cluster_id in range(n_clusters):
            # 获取当前聚类的节点索引
            cluster_indices = (cluster_labels == cluster_id).nonzero(as_tuple=True)[0]

            if len(cluster_indices) > 0:
                # 计算当前聚类中节点到核心点的距离
                cluster_nodes = node_embeddings[cluster_indices]
                distances = torch.norm(cluster_nodes - cluster_nodes.mean(dim=0), dim=1)

                # 只考虑核心点和非噪声点的聚类
                loss += distances.sum()
                cluster_count += 1

        # 计算平均损失，防止损失值过大
        return loss / cluster_count if cluster_count > 0 else loss

    def calculate_hdbscan_loss(node_embeddings, cluster_labels, membership_strengths):
        """
        计算 HDBSCAN 风格的无监督聚类损失
        :param node_embeddings: 形状为 (num_nodes, embedding_dim) 的节点嵌入
        :param cluster_labels: 形状为 (num_nodes,) 的聚类标签
        :param membership_strengths: 形状为 (num_nodes,) 的成员度量（HDBSCAN 额外输出的）
        :return: 计算得到的损失值
        """
        n_clusters = cluster_labels.max().item() + 1  # 获取聚类数量
        loss = 0.0
        cluster_count = 0  # 记录有效聚类的数量

        for cluster_id in range(n_clusters):
            # 获取当前簇的点索引
            cluster_indices = (cluster_labels == cluster_id).nonzero(as_tuple=True)[0]

            if len(cluster_indices) > 0:
                cluster_nodes = node_embeddings[cluster_indices]
                cluster_membership = membership_strengths[cluster_indices]  # 获取成员度

                # 计算密度加权的均值中心
                weighted_mean = (cluster_nodes * cluster_membership[:, None]).sum(dim=0) / cluster_membership.sum()

                # 计算密度感知的紧凑损失
                distances = torch.norm(cluster_nodes - weighted_mean, dim=1) * cluster_membership
                loss += distances.sum()
                cluster_count += 1

        return loss / cluster_count if cluster_count > 0 else loss

    # def calculate_graph_clustering_loss(self,node_embeddings, cluster_labels):
    #     """
    #        计算无监督聚类损失。
    #
    #        :param node_embeddings: 节点的嵌入，形状为 (num_nodes, embedding_dim)
    #        :param cluster_labels: 每个节点的聚类标签，形状为 (num_nodes,)
    #        :return: 计算得到的损失值
    #        """
    #     n_clusters = cluster_labels.max().item() + 1  # 聚类数量
    #     loss = 0.0
    #
    #     # 遍历每个聚类
    #     for cluster_id in range(n_clusters):
    #         # 获取当前聚类的节点索引
    #         cluster_indices = (cluster_labels == cluster_id).nonzero(as_tuple=True)[0]
    #
    #         if len(cluster_indices) > 0:
    #             # 计算当前聚类的中心
    #             cluster_center = node_embeddings[cluster_indices].mean(dim=0)
    #
    #             # 计算当前聚类中每个节点到聚类中心的距离
    #             distances = torch.norm(node_embeddings[cluster_indices] - cluster_center, dim=1)
    #             loss += distances.sum()  # 将所有距离相加
    #
    #     # 可以选择对损失进行平均
    #     return loss / len(node_embeddings) if len(node_embeddings) > 0 else loss
    def forward(self, batch_data):
        batch_size = len(batch_data['visit_id'])
        # patient_emb_list = []
        # 病人的原始嵌入字典，和下面的嵌入字典没看出来差别，先继续往下看！！！！！看出来了，这里是保存老三样的生成的嵌入
        patient_emb_list = []
        """处理lab,这里是针对监测水平医学事件单独建模，利用GRU学习"""
        feature_paris = list(zip(*[iter(self.feature_monitor_event_keys)] * 2))
        # 迭代处理每一对
        for feature_key1, feature_key2 in feature_paris:
            monitor_emb_list = []
            # 先聚合monitor层面，生成batch_size个病人的多次就诊的表征，batch_size * (1, visit, embedding)
            for patient in range(batch_size):
                x1 = self.monitor_event_token[feature_key1].batch_encode_3d(
                    batch_data[feature_key1][patient], max_length=(400, 1024)
                )
                x1 = torch.tensor(x1, dtype=torch.long, device=self.device)
                x2 = self.monitor_event_token[feature_key2].batch_encode_3d(
                    batch_data[feature_key2][patient], max_length=(400, 1024)
                )
                x2 = torch.tensor(x2, dtype=torch.long, device=self.device)
                # (visit, monitor, event)

                x1 = self.dropout(self.embeddings[feature_key1](x1))
                x2 = self.dropout(self.embeddings[feature_key2](x2))
                # (visit, monitor, event, embedding_dim)

                x = torch.mul(x1, x2)
                # (visit, monitor, event, embedding_dim)

                x = torch.sum(x, dim=2)
                # (visit, monitor, embedding_dim)

                monitor_emb_list.append(x)

            # 聚合多次的monitor
            monitor_tensor = pad_tensors(monitor_emb_list)
            # (patient, visit, monitor, embedding_dim)

            batch_size, max_visits, max_monitors, _ = monitor_tensor.size()
            monitor_tensor = monitor_tensor.view(-1, max_monitors, self.embedding_dim).to(self.device)
            # (patient * visit, monitor, embedding_dim)

            # 扩展张量
            # expended_visit_emb_list = expand_tensors(visit_emb_list, monitor_tensor.shape)
            # list[3 * (patient, visit, monitor, embedding_dim)]

            # expended_visit_emb_list.append(monitor_tensor)
            # list[4 * (patient, visit, monitor, embedding_dim)]

            output, hidden = self.visit_gru[feature_key1](monitor_tensor)
            # output: (patient * visit, monitor, embedding_dim), hidden:(1, batch_size * max_visits, embedding_dim)
            hidden = hidden.squeeze(0)
            # (batch_size * max_visits, dim)
            visit_tensor = hidden.view(batch_size, max_visits, self.embedding_dim)
            # (patient, visit, embedding_dim)

            output, hidden = self.visit_gru[feature_key1](visit_tensor)
            # output:(patient, visit, embedding_dim), hidden:(1, patient, embedding_dim)

            patient_emb_list.append(hidden.squeeze(dim=0))
            # (patient, event)
        """处理cond, proc, drug"""  # 这个和base2_1一样的处理方式，目的都是得到一个嵌入，并且把老三样cond、diag和drug用上
        for feature_key in self.feature_visit_event_keys:
            x = self.visit_event_token[feature_key].batch_encode_3d(
                batch_data[feature_key]
            )
            x = torch.tensor(x, dtype=torch.long, device=self.device)
            # (patient, visit, event)

            x = self.dropout(self.embeddings[feature_key](x))
            # (patient, visit, event, embedding_dim)

            x = torch.sum(x, dim=2)
            # (patient, visit, embedding_dim)

            output, hidden = self.visit_gru[feature_key](x)
            # output:(patient, visit, embedding_dim), hidden:(1, patient, embedding_dim)
            patient_emb_list.append(hidden.squeeze(dim=0))
            #(patient, embedding_dim)
        # 这里写表征学习的具体实现，我们先学visit图
        total_clustering_loss = 0  # 初始化图损失
        # feature_paris = list(zip(*[iter(self.feature_monitor_event_keys)] * 2))
        # 根据monitor信息，这里我们用lab信息，进行聚类和结构学习
        feature_key1 = 'lab_item'
        feature_key2= 'lab_flag'
        generated_patient_vectors = []
        for patient in range(batch_size):
            x1 = self.monitor_event_token[feature_key1].batch_encode_3d(
                batch_data[feature_key1][patient], max_length=(400, 1024)
            )
            # 变成张量
            x1 = torch.tensor(x1, dtype=torch.long, device=self.device)
            # 同X1对lab_flag
            x2 = self.monitor_event_token[feature_key2].batch_encode_3d(
                batch_data[feature_key2][patient], max_length=(400, 1024)
            )
            x2 = torch.tensor(x2, dtype=torch.long, device=self.device)
            # (visit, monitor, event)
            # 传入嵌入曾并且增强加入dropout正则化
            x1 = self.dropout(self.embeddings[feature_key1](x1))
            x2 = self.dropout(self.embeddings[feature_key2](x2))
            # (visit, monitor, event, embedding_dim)

            x = torch.mul(x1, x2)
            # (visit, monitor, event, embedding_dim)
            # x为lab对的生成的嵌入
            x = torch.sum(x, dim=2)
            # 获取该患者的 visit 图和 monitor 信息
            visits = batch_data["visit_graph"][patient]
            generated_visit_vectors = []
            for visit_id in range(len(visits)):
                visit_graph = visits[visit_id][0]
                visit_monitors = x[visit_id]  # 该 visit 的所有 monitor 信息

                # 将 monitor 信息通过 GRU 聚合
                monitor_embeddings = []
                for monitor_info in visit_monitors:
                    # 这里假设 monitor_info 是一个合适的张量输入到 GRU 中
                    monitor_embeddings.append(monitor_info)

                # 将 monitor 嵌入堆叠成一个张量
                monitor_tensor = torch.stack(monitor_embeddings, dim=0).to(self.device)

                # 通过 GRU 进行聚合
                gru_output, hidden = self.graph_monitor_Learning(monitor_tensor)  # 假设 self.gru 是已定义的 GRU 网络
                aggregated_monitor_info = hidden.squeeze(dim=0)  # 获取聚合后的信息

                # 获取节点索引
                node_ids = visit_graph.x.to(self.device)

                # 获取节点嵌入
                original_node_emb = self.node_emb(node_ids).float().to(self.device)

                # 根据聚合的 monitor 信息生成新的图
                new_graph, cluster_labels = self.generate_graph_based_on_monitor(visit_graph, aggregated_monitor_info,
                                                                                 original_node_emb)

                # 获取节点和边的嵌入
                node_em = new_graph.x.to(self.device)
                rel_ids = new_graph.relation.to(self.device)
                edge_attr = self.rel_emb(rel_ids).float().to(self.device)

                # 检查边索引是否为空
                if new_graph.edge_index.numel() == 0:
                    continue  # 跳过此次循环

                # 使用 GNN 对新图进行表征学习
                x1 = self.GATWithEdgeEmbedding(node_em, new_graph.edge_index.to(self.device), edge_attr)

                # 计算聚类损失
                clustering_loss = self.calculate_graph_clustering_loss(original_node_emb, cluster_labels)
                # clustering_loss = self.calculate_graph_clustering_loss(original_node_emb, cluster_labels, membership_strengths)
                total_clustering_loss += clustering_loss  # 累加聚类损失

                # 将每个 visit 的 GNN 输出保存
                generated_visit_vectors.append(x1)

            # 将多个 visit 的表示输入到 RNN 中
            # 在进行堆叠之前检查generated_visit_vectors是否为空
            if len(generated_visit_vectors) > 0:
                # 将多个 visit 的表示输入到 RNN 中
                stacked_visit_vectors = torch.stack(generated_visit_vectors, dim=0)
                output, hidden = self.graph_visit_Learning(stacked_visit_vectors)
                generated_patient_vectors.append(hidden.squeeze(dim=0))
            else:
                # 如果 generated_visit_vectors 为空，用默认值填充
                default_tensor = torch.zeros(self.embedding_dim).to(self.device)  # 假设 self.hidden_dim 是隐藏层维度
                generated_patient_vectors.append(default_tensor)

        # 堆叠所有病人的向量
        stacked_patient_vectors = torch.stack(generated_patient_vectors, dim=0)
        patient_emb_list.append(stacked_patient_vectors)
        #考虑患者的年龄和体重

        # for feature_key in ['weight', 'age']:
        #     x = batch_data[feature_key]
        #
        #     # 找出最长列表的长度
        #     max_length = max(len(sublist) for sublist in x)
        #     # 将每个子列表的元素转换为浮点数，并使用0对齐长度
        #     x = [[float(item) for item in sublist] + [0] * (max_length - len(sublist)) for sublist in x]
        #     # (patient, visit)
        #
        #     x = torch.tensor(x, dtype=torch.float, device=self.device)
        #     # (patient, visit)
        #
        #     num_patients, num_visits = x.shape
        #     x = x.view(-1, 1)  # 变成 (patient * visit, 1)
        #
        #     # 创建一个掩码用于标记输入为0的位置
        #     mask = (x == 0)
        #
        #     if feature_key == 'weight':
        #         x = self.dropout(self.fc_weight(x))
        #     elif feature_key == 'age':
        #         x = self.dropout(self.fc_age(x))
        #     # 对输入为0的位置输出也设为0
        #     x = x * (~mask)
        #     # (patient * visit, embedding_dim)
        #
        #     x = x.view(num_patients, num_visits, -1)
        #     # (patient, visit, embedding_dim)
        #
        #     output, hidden = self.visit_gru[feature_key](x)
        #     # output:(patient, visit, embedding_dim), hidden:(1, patient, embedding_dim)
        #
        #     patient_emb_list.append(hidden.squeeze(dim=0))
        # 合并患者的表示
        patient_emb = torch.cat(patient_emb_list, dim=-1)

        # 计算最终预测 logits
        logits = self.fc_patient(patient_emb)
        # 返回预测结果和聚类损失
        return logits, total_clustering_loss

