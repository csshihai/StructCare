from torch.utils.data import Dataset
from torch_geometric.data import HeteroData
from torch_geometric.utils import k_hop_subgraph, from_networkx
from tqdm import *
import os
from data_preprocess.data_load import load_preprocessed_data, save_preprocessed_data
import torch
import torch.nn as nn


class Sample_copy_Graph(torch.utils.data.Dataset):

    def __init__(self, Tokenizer_visit_event, Tokenizer_monitor_event, dataset, G, task):
        self.visit_event_token = Tokenizer_visit_event
        self.monitor_event_token = Tokenizer_monitor_event
        self.feature_visit_event_keys = list(Tokenizer_visit_event.keys())
        self.feature_monitor_event_keys = list(Tokenizer_monitor_event.keys())
        self.dataset = dataset
        self.G = G
        self.task = task
        self.all_data = self._process_()

    def __getitem__(self, index):
        return self.all_data[index]

    def __len__(self):
        return len(self.all_data)

    def get_subgraph(self, idx):
        patient = self.dataset[idx]
        subgraphs = []
        for visit_i in range(len(patient['procedures'])):
            if len(patient['visit_node_set'][visit_i]) == 0:
                continue
            nodes, _, _, edge_mask = k_hop_subgraph(torch.tensor(patient['visit_node_set'][visit_i]), 2,
                                                    self.G.edge_index)
            mask_idx = torch.where(edge_mask)[0]
            L = self.G.edge_subgraph(mask_idx)
            P = L.subgraph(torch.tensor(patient['visit_node_set'][visit_i]))
            if(self.task == "drug_rec"):
                P.label = patient['drugs_ind']
            else:
                P.label = patient['diag_ind']
            P.patient_id = patient['patient_id']

            subgraphs.append(P)

        return subgraphs

    # def get_subgraph(self, idx):
    #     patient = self.dataset[idx]
    #     subgraphs = []
    #     for visit_i in range(len(patient['conditions'])):
    #         if len(patient['visit_node_set'][visit_i]) == 0:
    #             continue
    #
    #         nodes, _, _, edge_mask = k_hop_subgraph(torch.tensor(patient['visit_node_set'][visit_i]), 2,
    #                                                 self.G.edge_index)
    #         mask_idx = torch.where(edge_mask)[0]
    #         L = self.G.edge_subgraph(mask_idx)
    #         P = L.subgraph(torch.tensor(patient['visit_node_set'][visit_i]))
    #
    #         # 传递边特征（如果存在）
    #         P.edge_attr = L.edge_attr[mask_idx]  # 确保边特征在子图中被保留
    #
    #         if self.task == "drug_rec":
    #             P.label = patient['drugs_ind']
    #         else:
    #             P.label = patient['diag_ind']
    #         P.patient_id = patient['patient_id']
    #
    #         subgraphs.append(P)
    #
    #     return subgraphs

    def _process_(self):
        graph_list = []
        for patient in tqdm(self.dataset):
            patient_graph_dict = {}
            # patient_graph_dict['inj_item_visit_graph'] = []
            # patient_graph_dict['lab_item_visit_graph'] = []
            patient_graph_dict['visit_graph'] = []
            for visit_id in range(len(patient['procedures'])):
                # 这行是为了判断一个病人的visit个数
                # l = len(patient['procedures'])
                # 获取子图
                subgraphs = self.get_subgraph(visit_id)
                patient_graph_dict['visit_graph'].append(subgraphs)
                # feature_pairs = list(zip(*[iter(self.feature_monitor_event_keys)] * 2))
                # for feature_key1, feature_key2 in feature_pairs:
                #     # 获取 graph_node_dict[feature_key1] 的长度
                #     monitor_length = len(patient[feature_key1][visit_id])
                #     visit_time_graph_list = []
                #     # 复制子图
                #     for _ in range(monitor_length):
                #         visit_time_graph_list.extend(subgraphs)
                #     if feature_key1 == 'inj_item':
                #         patient_graph_dict['inj_item_visit_graph'].append(visit_time_graph_list)
                #     elif feature_key1 == 'lab_item':
                #         patient_graph_dict['lab_item_visit_graph'].append(visit_time_graph_list)
                #     else:
                #         exit("没有这个东西")
            graph_list.append(patient_graph_dict)
            # print(1)
        return graph_list


def dataset_collate(dataset, Tokenizers_event, Tokenizers_monitor, dataset_name,task):
    processed_Graph_path = f"./graph_generation/data/cond_proc_drug/CCSCM_CCSPROC_ATC3/graph_{dataset_name}_{task}_th015.pkl"
    print(processed_Graph_path)
    if os.path.exists(processed_Graph_path):
        G = load_preprocessed_data(processed_Graph_path)
        G_tg = from_networkx(G)
    graph_dataset = Sample_copy_Graph(Tokenizers_event, Tokenizers_monitor, dataset.samples,G_tg,task).all_data
    print("图抽取完了")
    combined_dataset = combine_datasets(dataset.samples, graph_dataset)
    dataset.samples = combined_dataset
    return dataset


def combine_datasets(sequence_dataset, graph_dataset):
    combined_dataset = []
    for seq_data, graph_data in zip(sequence_dataset, graph_dataset):
        combined_data = {**seq_data, **graph_data}
        combined_dataset.append(combined_data)
    return combined_dataset


def process_data_with_graph(task_dataset, Tokenizers_visit_event, Tokenizers_monitor_event, args):
    """数据加入图"""

    # 判断是否是developer
    if args.developer:
        processed_data_path = f'./data/{args.dataset}/processed_data/{args.task}/data_with_graph/' \
                              f'processed_developer_data.pkl'
    else:
        processed_data_path = f'./data/{args.dataset}/processed_data/{args.task}/data_with_graph/processed_data.pkl'

    # 判断是否有处理好的数据
    if os.path.exists(processed_data_path):
        print("Processed graph data exists, loading directly.")
        task_dataset_with_graph = load_preprocessed_data(processed_data_path)
    else:
        print("Graph data not processed, reconstructing the graph.")
        task_dataset_with_graph = dataset_collate(task_dataset, Tokenizers_visit_event, Tokenizers_monitor_event,args.dataset,args.task)
        save_preprocessed_data(task_dataset_with_graph, processed_data_path)

    return task_dataset_with_graph
