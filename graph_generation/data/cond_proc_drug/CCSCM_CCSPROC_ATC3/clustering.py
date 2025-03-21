import json
import pickle
from collections import defaultdict

import numpy as np
from sklearn.cluster import AgglomerativeClustering


def load_embeddings(task):
    if task == "drug_rec":
        with open('../../../../graphs/cond_proc_drug/CCSCM_CCSPROC_ATC3/ent2id.json', 'r') as file:
            ent2id = json.load(file)
        with open('../../../../graphs/cond_proc_drug/CCSCM_CCSPROC_ATC3/rel2id.json', 'r') as file:
            rel2id = json.load(file)
        with open('../../../../graphs/cond_proc_drug/CCSCM_CCSPROC_ATC3/entity_embedding.pkl', 'rb') as file:
            ent_emb = pickle.load(file)
        with open('../../../../graphs/cond_proc_drug/CCSCM_CCSPROC_ATC3/relation_embedding.pkl', 'rb') as file:
            rel_emb = pickle.load(file)
    else:
        with open('../../../../graphs/cond_proc_drug/CCSCM_CCSPROC_ATC3/ent2id.json', 'r') as file:
            ent2id = json.load(file)
        with open('../../../../graphs/cond_proc_drug/CCSCM_CCSPROC_ATC3/rel2id.json', 'r') as file:
            rel2id = json.load(file)
        with open('../../../../graphs/cond_proc_drug/CCSCM_CCSPROC_ATC3/entity_embedding.pkl', 'rb') as file:
            ent_emb = pickle.load(file)
        with open('../../../../graphs/cond_proc_drug/CCSCM_CCSPROC_ATC3/relation_embedding.pkl', 'rb') as file:
            rel_emb = pickle.load(file)
    return ent2id, rel2id, ent_emb, rel_emb


def clustering(task, ent_emb, rel_emb, threshold=0.15, load_cluster=False, save_cluster=False):
    if task == "drug_rec":
        path = "./"
    else:
        # 这个几个任务我们用不到
        path = "./"
    # 如果文件已经保存
    #"/home/tcg/sunqizheng/Structure_care/graph_generation/data/cond_proc_drug/CCSCM_CCSPROC_ATC3/"
    if load_cluster:
        with open(f'{path}/clusters_th015.json', 'r', encoding='utf-8') as f:
            map_cluster = json.load(f)
        with open(f'{path}/clusters_inv_th015.json', 'r', encoding='utf-8') as f:
            map_cluster_inv = json.load(f)
        with open(f'{path}/clusters_rel_th015.json', 'r', encoding='utf-8') as f:
            map_cluster_rel = json.load(f)
        with open(f'{path}/clusters_inv_rel_th015.json', 'r', encoding='utf-8') as f:
            map_cluster_inv_rel = json.load(f)

    else:
        # 根据阈值聚类实体和关系（即实体和边），linkage参数是指测量不同簇之间距离的方式，metric则使用余弦函数测量相似度
        # 这里的距离指的是余弦距离，即1-余弦相似度，平均距离则是指两簇之间所有的余弦距离之和，除以总数
        cluster_alg = AgglomerativeClustering(n_clusters=None, distance_threshold=threshold, linkage='average',
                                              metric='cosine')
        cluster_labels = cluster_alg.fit_predict(ent_emb)
        cluster_labels_rel = cluster_alg.fit_predict(rel_emb)

        def nested_dict():
            return defaultdict(list)

        # 初始化和聚类节点
        map_cluster = defaultdict(nested_dict)

        for unique_l in np.unique(cluster_labels):
            # cluster_labels = [0, 1, 0, 1, 2, 2, 0, 1, 2]
            # cur是# cluster_labels的索引
            for cur in range(len(cluster_labels)):
                if cluster_labels[cur] == unique_l:
                    map_cluster[str(unique_l)]['nodes'].append(cur)
        # 计算每个聚类的平均嵌入向量
        for unique_l in map_cluster.keys():
            nodes = map_cluster[unique_l]['nodes']
            nodes = np.array(nodes)
            embedding_mean = np.mean(ent_emb[nodes], axis=0)
            map_cluster[unique_l]['embedding'].append(embedding_mean.tolist())
        # 最终生成的结果形式如下
        # map_cluster = {
        #     '0': {'nodes': [0, 2, 6], 'embedding': []},
        #     '1': {'nodes': [1, 3, 7], 'embedding': []},
        #     '2': {'nodes': [4, 5, 8], 'embedding': []}
        # }
        # 创建节点到聚类标签的反向映射
        map_cluster_inv = {}
        for cluster_label, item in map_cluster.items():
            for node in item['nodes']:
                map_cluster_inv[str(node)] = cluster_label
        # 初始化聚类关系
        map_cluster_rel = defaultdict(nested_dict)

        for unique_l in np.unique(cluster_labels_rel):
            for cur in range(len(cluster_labels_rel)):
                if cluster_labels_rel[cur] == unique_l:
                    map_cluster_rel[str(unique_l)]['relations'].append(cur)

        for unique_l in map_cluster_rel.keys():
            nodes = map_cluster_rel[unique_l]['relations']
            nodes = np.array(nodes)
            embedding_mean = np.mean(ent_emb[nodes], axis=0)
            map_cluster_rel[unique_l]['embedding'].append(embedding_mean.tolist())
        # 创建关系到聚类标签的反向映射
        map_cluster_inv_rel = {}
        for cluster_label, item in map_cluster_rel.items():
            for node in item['relations']:
                map_cluster_inv_rel[str(node)] = cluster_label
        # 是否保存聚类
        if save_cluster:
            with open(f'{path}/clusters_th015.json', 'w', encoding='utf-8') as f:
                json.dump(map_cluster, f, indent=6)
            with open(f'{path}/clusters_inv_th015.json', 'w', encoding='utf-8') as f:
                json.dump(map_cluster_inv, f, indent=6)
            with open(f'{path}/clusters_rel_th015.json', 'w', encoding='utf-8') as f:
                json.dump(map_cluster_rel, f, indent=6)
            with open(f'{path}/clusters_inv_rel_th015.json', 'w', encoding='utf-8') as f:
                json.dump(map_cluster_inv_rel, f, indent=6)

    return map_cluster, map_cluster_inv, map_cluster_rel, map_cluster_inv_rel


def main():
    load_cluster = False
    save_cluster = True
    task ="drug_rec"
    ent2id, rel2id, ent_emb, rel_emb = load_embeddings(task)
    map_cluster, map_cluster_inv, map_cluster_rel, map_cluster_inv_rel = clustering(task, ent_emb, rel_emb,
                                                                                    threshold=0.15,
                                                                                    load_cluster=load_cluster,
                                                                                    save_cluster=save_cluster)
if __name__ == '__main__':
    main()