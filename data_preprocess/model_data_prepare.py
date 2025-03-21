
import csv
import os

import pickle
import json
from sklearn.cluster import AgglomerativeClustering
from collections import defaultdict
import networkx as nx
from torch_geometric.utils import from_networkx
from tqdm import tqdm
from data_preprocess.drug_recommendation_mimic34_fn import drug_recommendation_mimic3_fn, drug_recommendation_mimic4_fn
from data_preprocess.diag_prediction_mimic34_fn import diag_prediction_mimic3_fn, diag_prediction_mimic4_fn
from utils import *
from data_preprocess.data_load import *


# 这里对应StructCare的 utils加载数据集


def label_ehr_nodes(sample_dataset, ccscm_id2clus, ccsproc_id2clus, atc3_id2clus,task):
    for patient in tqdm(sample_dataset):
        for visit_idx in range(len(patient['procedures'])):
            # nodes = []
            # 如果是诊断的话应该还会有判断
            if task == "drug_rec":
                conditions = patient['conditions'][visit_idx]
                procedures = patient['procedures'][visit_idx]
                drugs = patient['drugs_hist'][visit_idx]
            else:
                conditions = patient['cond_hist'][visit_idx]
                procedures = patient['procedures'][visit_idx]
                drugs = patient['drugs'][visit_idx]

            for condition in conditions:
                ehr_node = ccscm_id2clus[condition]
                # nodes.append(int(ehr_node))
                patient['visit_node_set'][visit_idx].append(int(ehr_node))

            for procedure in procedures:
                ehr_node = ccsproc_id2clus[procedure]
                # nodes.append(int(ehr_node))
                patient['visit_node_set'][visit_idx].append(int(ehr_node))

            for drug in drugs:
                ehr_node = atc3_id2clus[drug]
                # nodes.append(int(ehr_node))
                patient['visit_node_set'][visit_idx].append(int(ehr_node))
    return sample_dataset


# 没人调用，都是直接粘贴
def load_mappings():
    condition_mapping_file = "../data/resources/CCSCM.csv"
    procedure_mapping_file = "../data/resources/CCSPROC.csv"
    drug_file = "../data/resources/ATC.csv"

    condition_dict = {}
    with open(condition_mapping_file, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            condition_dict[row['code']] = row['name'].lower()

    procedure_dict = {}
    with open(procedure_mapping_file, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            procedure_dict[row['code']] = row['name'].lower()

    drug_dict = {}
    with open(drug_file, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row['level'] == '3.0':
                drug_dict[row['code']] = row['name'].lower()

    return condition_dict, procedure_dict, drug_dict


# 列表展开
def flatten(lst):
    result = []
    for item in lst:
        if isinstance(item, list):
            result.extend(flatten(item))
        else:
            result.append(item)
    return result


def load_embeddings(task):
    if task == "drug_rec":
        with open('../graphs/cond_proc_drug/CCSCM_CCSPROC_ATC3/ent2id.json', 'r') as file:
            ent2id = json.load(file)
        with open('../graphs/cond_proc_drug/CCSCM_CCSPROC_ATC3/rel2id.json', 'r') as file:
            rel2id = json.load(file)
        with open('../graphs/cond_proc_drug/CCSCM_CCSPROC_ATC3/entity_embedding.pkl', 'rb') as file:
            ent_emb = pickle.load(file)
        with open('../graphs/cond_proc_drug/CCSCM_CCSPROC_ATC3/relation_embedding.pkl', 'rb') as file:
            rel_emb = pickle.load(file)
    else:
        with open('../graphs/cond_proc_drug/CCSCM_CCSPROC_ATC3/ent2id.json', 'r') as file:
            ent2id = json.load(file)
        with open('../graphs/cond_proc_drug/CCSCM_CCSPROC_ATC3/rel2id.json', 'r') as file:
            rel2id = json.load(file)
        with open('../graphs/cond_proc_drug/CCSCM_CCSPROC_ATC3/entity_embedding.pkl', 'rb') as file:
            ent_emb = pickle.load(file)
        with open('../graphs/cond_proc_drug/CCSCM_CCSPROC_ATC3/relation_embedding.pkl', 'rb') as file:
            rel_emb = pickle.load(file)
    return ent2id, rel2id, ent_emb, rel_emb


# 多热编码 用下面两个方法的目的就是生成一个病人在所有药物里面开到的药的0，1 向量
def multihot(label, num_labels):
    multihot = np.zeros(num_labels)
    for l in label:
        multihot[l] = 1
    return multihot


#
def prepare_drug_label(sample_dataset, drugs):
    label_tokenizer = Tokenizer(
        sample_dataset.get_all_tokens(key='drugs')
    )

    labels_index = label_tokenizer.convert_tokens_to_indices(drugs)
    num_labels = label_tokenizer.get_vocabulary_size()
    labels = multihot(labels_index, num_labels)
    return labels
def prepare_diag_label(sample_dataset, drugs):
    label_tokenizer = Tokenizer(
        sample_dataset.get_all_tokens(key='conditions')
    )

    labels_index = label_tokenizer.convert_tokens_to_indices(drugs)
    num_labels = label_tokenizer.get_vocabulary_size()
    labels = multihot(labels_index, num_labels)
    return labels

def prepare_drug_indices(sample_dataset):
    for patient in tqdm(sample_dataset):
        patient['drugs_ind'] = torch.tensor(prepare_drug_label(sample_dataset, patient['drugs']))
    return sample_dataset


def prepare_diag_indices(sample_dataset):
    for patient in tqdm(sample_dataset):
        patient['diag_ind'] = torch.tensor(prepare_diag_label(sample_dataset, patient['conditions']))
    return sample_dataset


#############聚类方法
def clustering(task, ent_emb, rel_emb, threshold=0.15, load_cluster=False, save_cluster=False):
    if task == "drug_rec":
        path = "../graph_generation/data/cond_proc_drug/CCSCM_CCSPROC_ATC3"
    else:
        # 这个几个任务我们用不到
        path = "../graph_generation/data/cond_proc_drug/CCSCM_CCSPROC_ATC3"
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


# 生成网络图
def process_graph(dataset, task, sample_dataset, ent2id, rel2id, map_cluster, map_cluster_inv, map_cluster_rel,
                  map_cluster_inv_rel, save_graph=False, dev=False):
    if dataset == "mimic3":
        if task == "drug_rec":
            # path = "./graph_generation/data/ccscm_ccsproc"
            path = "../graph_generation/data/cond_proc_drug/CCSCM_CCSPROC_ATC3"
        else:
            path = "../graph_generation/data/cond_proc_drug/CCSCM_CCSPROC_ATC3"
    # dai
    if dataset == "mimic4":
        if task == "drug_rec":
            # path = "./graph_generation/data/ccscm_ccsproc"
            path = "../graph_generation/data/cond_proc_drug/CCSCM_CCSPROC_ATC3"
        else:
            path = "../graph_generation/data/cond_proc_drug/CCSCM_CCSPROC_ATC3"
    G = nx.Graph()
    # 将每个聚类的代表性信息作为一个节点添加到图G中
    for cluster_label, item in map_cluster.items():
        G.add_nodes_from([
            (int(cluster_label), {'x': int(cluster_label)})
        ])
        G.add_nodes_from([int(cluster_label)])

    for patient in tqdm(sample_dataset):
        triple_set = set()
        # 遍历每一个条件
        if task == "drug_rec":
            conditions = flatten(patient['conditions'])
        else:
            conditions = flatten(patient['cond_hist'])
        for condition in conditions:
            cond_file = f'../graphs/condition/CCSCM/{condition}.txt'
            with open(cond_file, 'r') as f:
                lines = f.readlines()
            for line in lines:
                # in case the map and emb is not up-to-date
                try:
                    items = line.split('\t')
                    if len(items) == 3:
                        h, r, t = items
                        # 去掉可能的换行符
                        t = t[:-1]
                        h = ent2id[h]
                        r = rel2id[r]
                        t = ent2id[t]
                        triple = (h, r, t)
                        if triple not in triple_set:
                            edge = (int(map_cluster_inv[h]), int(map_cluster_inv[t]))
                            # *edge添加边，后面添加边的属性
                            G.add_edge(*edge, relation=int(map_cluster_inv_rel[r]))
                        triple_set.add(triple)
                except:
                    continue

        procedures = flatten(patient['procedures'])
        for procedure in procedures:
            proc_file = f'../graphs/procedure/CCSPROC/{procedure}.txt'
            with open(proc_file, 'r') as f:
                lines = f.readlines()
            for line in lines:
                try:
                    items = line.split('\t')
                    if len(items) == 3:
                        h, r, t = items
                        t = t[:-1]
                        h = ent2id[h]
                        r = rel2id[r]
                        t = ent2id[t]
                        triple = (h, r, t)
                        if triple not in triple_set:
                            edge = (int(map_cluster_inv[h]), int(map_cluster_inv[t]))
                            G.add_edge(*edge, relation=int(map_cluster_inv_rel[r]))
                            triple_set.add(triple)
                except KeyError:
                    # 忽略映射中未找到的实体或关系
                    continue
                except:
                    # 忽略类型转换错误
                    continue
        if task == "drug_rec":
            drugs = flatten(patient['drugs_hist'])
        else:
            drugs = flatten(patient['drugs'])
        for drug in drugs:
            proc_file = f'../graphs/drug/ATC3/{drug}.txt'
            with open(proc_file, 'r') as f:
                lines = f.readlines()
            for line in lines:
                try:
                    items = line.split('\t')
                    if len(items) == 3:
                        h, r, t = items
                        t = t[:-1]
                        h = ent2id[h]
                        r = rel2id[r]
                        t = ent2id[t]
                        triple = (h, r, t)
                        if triple not in triple_set:
                            edge = (int(map_cluster_inv[h]), int(map_cluster_inv[t]))
                            G.add_edge(*edge, relation=int(map_cluster_inv_rel[r]))
                            triple_set.add(triple)
                except:
                    continue
    if save_graph:
        if dev == True:
            with open(f'{path}/graph_{dataset}_{task}_developer_th015.pkl', 'wb') as f:
                pickle.dump(G, f)
        else:
            with open(f'{path}/graph_{dataset}_{task}_th015.pkl', 'wb') as f:
                pickle.dump(G, f)
    return G


#
def pad_and_convert(visits, max_visits, max_nodes):
    # 填充函数，
    padded_visits = []
    # 填充一次就诊的长度
    for idx in range(len(visits) - 1, -1, -1):
        # 创建一个全0的张量
        visit_multi_hot = torch.zeros(max_nodes)
        for idx, med_code in enumerate(visits[idx]):
            visit_multi_hot[med_code] = 1
        padded_visits.append(visit_multi_hot)
    # 填充最大就诊次数
    while len(padded_visits) < max_visits:
        padded_visits.append(torch.zeros(max_nodes))
    return torch.stack(padded_visits, dim=0)


# 这个方法很重要！！！！
def process_sample_dataset(dataset, task, sample_dataset, G_tg, ent2id, rel2id, map_cluster, map_cluster_inv,
                           map_cluster_rel, map_cluster_inv_rel, save_dataset=False, dev=False):
    if task == "drug_rec":
        if dataset == "mimic3":
            path = "../data/mimic3/processed_data/drug_rec"
        else:
            path = "../data/mimic4/processed_data/drug_rec"

    else:
        if dataset == "mimic3":
            path = "../data/mimic3/processed_data/diag_pred"
        else:
            path = "../data/mimic4/processed_data/diag_pred"

    c_v = []
    for patient in sample_dataset:
        c_v.append(len(patient['procedures']))

    max_visits = max(c_v)
    # 这里是一条记录
    for patient in tqdm(sample_dataset):
        node_set_all = set()
        node_set_list = []
        for visit_i in range(len(patient['procedures'])):
            triple_set = set()
            node_set = set()
            if task == "drug_rec":
                conditions = patient['conditions'][visit_i]
                procedures = patient['procedures'][visit_i]
                drugs = patient['drugs_hist'][visit_i]
            else:
                conditions = patient['cond_hist'][visit_i]
                procedures = patient['procedures'][visit_i]
                drugs = patient['drugs'][visit_i]
            for condition in conditions:
                cond_file = f'../graphs/condition/CCSCM/{condition}.txt'
                with open(cond_file, 'r') as f:
                    lines = f.readlines()
                for line in lines:
                    try:
                        items = line.split('\t')
                        if len(items) == 3:
                            h, r, t = items
                            t = t[:-1]
                            h = ent2id[h]
                            # r = int(rel2id[r]) + len(ent_emb)
                            t = ent2id[t]
                            triple = (h, r, t)
                            if triple not in triple_set:
                                triple_set.add(triple)
                                node_set.add(int(map_cluster_inv[h]))
                                # node_set.add(r)
                                node_set.add(int(map_cluster_inv[t]))
                    except:
                        continue
            for procedure in procedures:
                proc_file = f'../graphs/procedure/CCSPROC/{procedure}.txt'
                with open(proc_file, 'r') as f:
                    lines = f.readlines()
                for line in lines:
                    try:
                        items = line.split('\t')
                        if len(items) == 3:
                            h, r, t = items
                            t = t[:-1]
                            h = ent2id[h]
                            # r = int(rel2id[r]) + len(ent_emb)
                            t = ent2id[t]
                            # 得到的三元组h和t是id，上面也有关系映射的但不知道这里为啥注释了
                            triple = (h, r, t)
                            if triple not in triple_set:
                                triple_set.add(triple)
                                node_set.add(int(map_cluster_inv[h]))
                                # node_set.add(r)
                                node_set.add(int(map_cluster_inv[t]))
                    except:
                        continue
            for drug in drugs:
                drug_file = f'../graphs/drug/ATC3/{drug}.txt'
                with open(drug_file, 'r') as f:
                    lines = f.readlines()
                for line in lines:
                    try:
                        items = line.split('\t')
                        if len(items) == 3:
                            h, r, t = items
                            t = t[:-1]
                            h = ent2id[h]
                            # r = int(rel2id[r]) + len(ent_emb)
                            t = ent2id[t]
                            triple = (h, r, t)
                            if triple not in triple_set:
                                triple_set.add(triple)
                                node_set.add(int(map_cluster_inv[h]))
                                # node_set.add(r)
                                node_set.add(int(map_cluster_inv[t]))
                    except:
                        continue
            # 将一次访问的节点都加入一个新列表，[*node_set]这个操作是将集合里面的元素取出来并放到一个列表中
            # node_set_list和node_set_all最终保存的是对应一条sample的诊断和手术信息
            node_set_list.append([*node_set])
            node_set_all.update(node_set)
        # 将
        padded_visits = pad_and_convert(node_set_list, max_visits, len(G_tg.x))
        # 这里修改了sample_dataset这个数据集
        patient['visit_node_set'] = node_set_list
        patient['visit_padded_node'] = padded_visits

    if save_dataset:
        if dev == True:
            with open(f'{path}/processed_addVisitNode_developer_data.pkl', 'wb') as f:
                pickle.dump(sample_dataset, f)
        else:
            with open(f'{path}/processed_addVisitNode_data.pkl', 'wb') as f:
                pickle.dump(sample_dataset, f)
        # with open(f'{path}/sample_dataset_{dataset}_{task}_th015.pkl', 'wb') as f:
        #     pickle.dump(sample_dataset, f)

    return sample_dataset


def run(dataset, task):
    if task == "drug_rec":
        load_processed_dataset = False
    else:
        load_processed_dataset = False
    load_cluster = False
    save_cluster = True
    save_graph = True
    save_processed_dataset = True
    dev = False

    print(f"Dataset: {dataset}, Task: {task}")
    print(f"Load processed dataset: {load_processed_dataset}")
    print(f"Load cluster: {load_cluster}")
    print(f"Save cluster: {save_cluster}")
    print(f"Save graph: {save_graph}")
    print(f"Save processed dataset: {save_processed_dataset}")
    print("Loading dataset...")
    if dataset == 'mimic3':
        if task == 'drug_rec':
            task_dataset = load_dataset(dataset,
                                        tables=['DIAGNOSES_ICD', 'PROCEDURES_ICD', 'PRESCRIPTIONS', "LABEVENTS",
                                                "INPUTEVENTS_MV"],
                                        root=f"../data/{dataset}/raw_data",
                                        task_fn=drug_recommendation_mimic3_fn,
                                        dev=dev)
        elif task == 'diag_pred':
            task_dataset = load_dataset(dataset,
                                        tables=['DIAGNOSES_ICD', 'PROCEDURES_ICD', 'PRESCRIPTIONS', "LABEVENTS",
                                                "INPUTEVENTS_MV"],
                                        root=f"../data/{dataset}/raw_data",
                                        task_fn=diag_prediction_mimic3_fn,
                                        dev=dev)
            sample_dataset = task_dataset
        else:
            raise ValueError("检查一下这个task")
    elif dataset == 'mimic4':
        if task == 'drug_rec':
            task_dataset = load_dataset(dataset,
                                        tables=['diagnoses_icd', 'procedures_icd', 'prescriptions', 'labevents', 'inputevents_mv'],
                                        root=f"../data/{dataset}/raw_data",
                                        task_fn=drug_recommendation_mimic4_fn,
                                        dev=False)
        elif task == 'diag_pred':
            task_dataset = load_dataset(dataset,
                                        tables=['diagnoses_icd', 'procedures_icd', 'prescriptions', 'labevents', 'inputevents_mv'],
                                        root=f"../data/{dataset}/raw_data",
                                        task_fn=diag_prediction_mimic4_fn,
                                        dev=False)
        else:
            raise ValueError("检查一下这个task")
    else:
        raise ValueError("检查一下dataset，没有这个数据集")
    if dev == True:
        save_preprocessed_data(task_dataset,
                               f"../data/{dataset}/processed_data/{task}/processed_devoloper_data.pkl")
    else:
        save_preprocessed_data(task_dataset, f"../data/{dataset}/processed_data/{task}/processed_data.pkl")
    print("Loading embeddings...")
    ent2id, rel2id, ent_emb, rel_emb = load_embeddings(task)

    if task == "drug_rec" and not load_processed_dataset:
        print("Preparing drug indices...")
        sample_dataset = prepare_drug_indices(task_dataset)
    else:
        print("Preparing diag indices...")
        sample_dataset = prepare_diag_indices(task_dataset)
    map_cluster, map_cluster_inv, map_cluster_rel, map_cluster_inv_rel = clustering(task, ent_emb, rel_emb,
                                                                                    threshold=0.15,
                                                                                    load_cluster=load_cluster,
                                                                                    save_cluster=save_cluster)
    save_preprocessed_data(task_dataset, f"../data/{dataset}/processed_data/{task}/processed_indices_data.pkl")
    print("Processing graph...")
    G = process_graph(dataset, task, sample_dataset, ent2id, rel2id, map_cluster, map_cluster_inv, map_cluster_rel,
                      map_cluster_inv_rel, save_graph=save_graph, dev=dev)
    G_tg = from_networkx(G)

    print("Processing dataset...")
    sample_dataset = process_sample_dataset(dataset, task, sample_dataset, G_tg, ent2id, rel2id, map_cluster,
                                            map_cluster_inv, map_cluster_rel, map_cluster_inv_rel,
                                            save_dataset=save_processed_dataset, dev=dev)


def main():
    current_path = os.getcwd()
    datasets = [
        # "mimic3",
        "mimic4"
    ]
    tasks = [
        # "drug_rec",
        "diag_pred"
    ]

    for dataset in datasets:
        for task in tasks:
            run(dataset, task)


if __name__ == "__main__":
    main()
