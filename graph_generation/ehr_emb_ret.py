import csv
from get_emb import embedding_retriever
import numpy as np
from tqdm import tqdm
import pickle
import json
# 创建三个映射字典，分别是诊断手术和药物映射代码对应的字典
ccscm_id2name = {}
with open('../resources/CCSCM.csv', 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines[1:]:
        line = line.strip().split(',')
        ccscm_id2name[line[0]] = line[1].lower()

ccsproc_id2name = {}
with open('../resources/CCSPROC.csv', 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines[1:]:
        line = line.strip().split(',')
        ccsproc_id2name[line[0]] = line[1].lower()

atc3_id2name = {}
with open("../resources/ATC.csv", newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        if row['level'] == '3.0':
            atc3_id2name[row['code']] = row['name'].lower()
path_0 = "./data"
# 利用api生成三个嵌入字典（一共多少种，就有多少个嵌入）,并且保存为pikle文件

ccscm_id2emb = {}
ccsproc_id2emb = {}
atc3_id2emb = {}


for key in tqdm(ccscm_id2name.keys()):
    emb = embedding_retriever(term=ccscm_id2name[key])
    ccscm_id2emb[key] = emb

for key in tqdm(ccsproc_id2name.keys()):
    emb = embedding_retriever(term=ccsproc_id2name[key])
    ccsproc_id2emb[key] = emb

for key in tqdm(atc3_id2name.keys()):
    emb = embedding_retriever(term=atc3_id2name[key])
    atc3_id2emb[key] = emb

with open(f"{path_0}/ccscm_id2emb.pkl", "wb") as f:
    pickle.dump(ccscm_id2emb, f)

with open(f"{path_0}/ccsproc_id2emb.pkl", "wb") as f:
    pickle.dump(ccsproc_id2emb, f)

with open(f"{path_0}/atc3_id2emb.pkl", "wb") as f:
    pickle.dump(atc3_id2emb, f)

# 通过相似度来聚类
def cosine_similarity(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))


# path_1 = "/data/ccscm_ccsproc"
# path_1_ = "../../graphs/cond_proc/CCSCM_CCSPROC"
#
# ent2id_file = f"{path_1_}/ent2id.json"
# ent_emb_file = f"{path_1_}/entity_embedding.pkl"
# # 正向聚类文件路径
# map_cluster_file = f"{path_1}/clusters_th015.json"
# # 反向聚类文件路径
# map_cluster_inv = f"{path_1}/clusters_inv_th015.json"
#
# with open(ent2id_file, "r") as f:
#     ent2id = json.load(f)
#
# with open(ent_emb_file, "rb") as f:
#     ent_emb = pickle.load(f)
#
# with open(map_cluster_file, "r") as f:
#     map_cluster = json.load(f)
#
# with open(map_cluster_inv, "r") as f:
#     map_cluster_inv = json.load(f)
# # 存储诊断和手术到聚类的映射
# ccscm_id2clus = {}
# ccsproc_id2clus = {}
#
# for key in tqdm(ccscm_id2emb.keys()):
#     emb = ccscm_id2emb[key]
#     emb = np.array(emb)
#     max_sim = 0
#     max_id = None
#     for i in range(ent_emb.shape[0]):
#         emb_compare = ent_emb[i]
#         sim = cosine_similarity(emb, emb_compare)
#         if sim > max_sim:
#             max_sim = sim
#             max_id = i
#
#     cluster_id = map_cluster_inv[str(max_id)]
#     ccscm_id2clus[key] = cluster_id
#
# for key in tqdm(ccsproc_id2emb.keys()):
#     emb = ccsproc_id2emb[key]
#     emb = np.array(emb)
#     max_sim = 0
#     max_id = None
#     for i in range(ent_emb.shape[0]):
#         emb_compare = ent_emb[i]
#         sim = cosine_similarity(emb, emb_compare)
#         if sim > max_sim:
#             max_sim = sim
#             max_id = i
#
#     cluster_id = map_cluster_inv[str(max_id)]
#     ccsproc_id2clus[key] = cluster_id
#
# with open(f"{path_1}/ccscm_id2clus.json", "w") as f:
#     json.dump(ccscm_id2clus, f)
#
# with open(f"{path_1}/ccsproc_id2clus.json", "w") as f:
#     json.dump(ccsproc_id2clus, f)
path_2 = "./data/cond_proc_drug/CCSCM_CCSPROC_ATC3"
path_2_ = "../graphs/cond_proc_drug/CCSCM_CCSPROC_ATC3"

ent2id_file = f"{path_2_}/ent2id.json"
ent_emb_file = f"{path_2_}/entity_embedding.pkl"
map_cluster_file = f"{path_2}/clusters_th015.json"
map_cluster_inv = f"{path_2}/clusters_inv_th015.json"

with open(ent2id_file, "r") as f:
    ent2id = json.load(f)

with open(ent_emb_file, "rb") as f:
    ent_emb = pickle.load(f)

with open(map_cluster_file, "r") as f:
    map_cluster = json.load(f)

with open(map_cluster_inv, "r") as f:
    map_cluster_inv = json.load(f)

ccscm_id2clus = {}
ccsproc_id2clus = {}
atc3_id2clus = {}

for key in tqdm(ccscm_id2emb.keys()):
    emb = ccscm_id2emb[key]
    emb = np.array(emb)
    max_sim = 0
    max_id = None
    for i in range(ent_emb.shape[0]):
        emb_compare = ent_emb[i]
        sim = cosine_similarity(emb, emb_compare)
        if sim > max_sim:
            max_sim = sim
            max_id = i

    cluster_id = map_cluster_inv[str(max_id)]
    ccscm_id2clus[key] = cluster_id

for key in tqdm(ccsproc_id2emb.keys()):
    emb = ccsproc_id2emb[key]
    emb = np.array(emb)
    max_sim = 0
    max_id = None
    for i in range(ent_emb.shape[0]):
        emb_compare = ent_emb[i]
        sim = cosine_similarity(emb, emb_compare)
        if sim > max_sim:
            max_sim = sim
            max_id = i

    cluster_id = map_cluster_inv[str(max_id)]
    ccsproc_id2clus[key] = cluster_id

for key in tqdm(atc3_id2emb.keys()):
    emb = atc3_id2emb[key]
    emb = np.array(emb)
    max_sim = 0
    max_id = None
    for i in range(ent_emb.shape[0]):
        emb_compare = ent_emb[i]
        sim = cosine_similarity(emb, emb_compare)
        if sim > max_sim:
            max_sim = sim
            max_id = i

    cluster_id = map_cluster_inv[str(max_id)]
    atc3_id2clus[key] = cluster_id

with open(f"{path_2}/ccscm_id2clus.json", "w") as f:
    json.dump(ccscm_id2clus, f)

with open(f"{path_2}/ccsproc_id2clus.json", "w") as f:
    json.dump(ccsproc_id2clus, f)

with open(f"{path_2}/atc3_id2clus.json", "w") as f:
    json.dump(atc3_id2clus, f)