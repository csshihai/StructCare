import numpy as np
from sklearn.cluster import KMeans
import networkx as nx
import matplotlib.pyplot as plt

# 假设我们有节点的嵌入向量
node_embeddings = np.array([
    [0.1, 0.2],
    [0.2, 0.3],
    [0.9, 0.8],
    [0.8, 0.9],
    [0.4, 0.5],
    [0.5, 0.6]
])

# 计算节点的相似性矩阵（这里使用嵌入向量的余弦相似度）
def cosine_similarity_matrix(embeddings):
    normed_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    return np.dot(normed_embeddings, normed_embeddings.T)

similarity_matrix = cosine_similarity_matrix(node_embeddings)

# 使用K均值聚类对节点进行聚类
num_clusters = 2
kmeans = KMeans(n_clusters=num_clusters)
clusters = kmeans.fit_predict(node_embeddings)

# 创建一个示例知识图谱
G = nx.Graph()
edges = [(0, 1), (1, 4), (2, 3), (3, 5), (4, 5)]
G.add_edges_from(edges)

# 根据聚类结果细化知识图谱
refined_subgraphs = []
for cluster_id in range(num_clusters):
    nodes_in_cluster = [i for i, cluster in enumerate(clusters) if cluster == cluster_id]
    subgraph = G.subgraph(nodes_in_cluster)
    refined_subgraphs.append(subgraph)

# 可视化原始知识图谱
pos = nx.spring_layout(G)
plt.figure(figsize=(12, 6))
plt.subplot(121)
nx.draw(G, pos, with_labels=True, node_color=[clusters[n] for n in G.nodes()])
plt.title("Original Knowledge Graph")

# 可视化细化后的子图
plt.subplot(122)
colors = ['red', 'blue']
for i, subgraph in enumerate(refined_subgraphs):
    nx.draw(subgraph, pos, with_labels=True, node_color=colors[i], edge_color=colors[i], node_size=500, alpha=0.7, label=f"Cluster {i}")
plt.title("Refined Subgraphs")
plt.legend()

plt.show()
