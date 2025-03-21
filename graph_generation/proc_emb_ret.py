import csv
import json
from get_emb import embedding_retriever
import numpy as np
from tqdm import tqdm
import pickle

condition_mapping_file = "../resources/CCSCM.csv"
procedure_mapping_file = "../resources/CCSPROC.csv"
drug_file = "../resources/ATC.csv"

condition_dict = {}
with open(condition_mapping_file, newline='',encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        condition_dict[row['code']] = row['name'].lower()

procedure_dict = {}
with open(procedure_mapping_file, newline='',encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        procedure_dict[row['code']] = row['name'].lower()

drug_dict = {}
with open(drug_file, newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        if row['level'] == '5.0':
            drug_dict[row['code']] = row['name'].lower()


proc_ent = set()
proc_rel = set()

file_dir = "../graphs/procedure/CCSPROC"

for key in procedure_dict.keys():
    file = f"{file_dir}/{key}.txt"
    with open(file=file, mode='r',encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines:
        parsed = line.split('\t')
        if len(parsed) == 3:
            h, r, t = line.split('\t')
            t = t[:-1]
            proc_ent.add(h)
            proc_ent.add(t)
            proc_rel.add(r)

proc_id2ent = {index: value for index, value in enumerate(proc_ent)}
proc_ent2id = {value: index for index, value in enumerate(proc_ent)}
proc_id2rel = {index: value for index, value in enumerate(proc_rel)}
proc_rel2id = {value: index for index, value in enumerate(proc_rel)}

out_file_id2ent = f"{file_dir}/id2ent.json"
out_file_ent2id = f"{file_dir}/ent2id.json"
out_file_id2rel = f"{file_dir}/id2rel.json"
out_file_rel2id = f"{file_dir}/rel2id.json"

with open(out_file_id2ent, 'w') as file:
    json.dump(proc_id2ent, file, indent=6)
with open(out_file_ent2id, 'w') as file:
    json.dump(proc_ent2id, file, indent=6)
with open(out_file_id2rel, 'w') as file:
    json.dump(proc_id2rel, file, indent=6)
with open(out_file_rel2id, 'w') as file:
    json.dump(proc_rel2id, file, indent=6)

file_dir = "../graphs/procedure/CCSPROC"

file_id2ent = f"{file_dir}/id2ent.json"
file_ent2id = f"{file_dir}/ent2id.json"
file_id2rel = f"{file_dir}/id2rel.json"
file_rel2id = f"{file_dir}/rel2id.json"

with open(file_id2ent, 'r') as file:
    proc_id2ent = json.load(file)
with open(file_ent2id, 'r') as file:
    proc_ent2id = json.load(file)
with open(file_id2rel, 'r') as file:
    proc_id2rel = json.load(file)
with open(file_rel2id, 'r') as file:
    proc_rel2id = json.load(file)


## get embedding for procedure entities
proc_ent_emb = []

for idx in tqdm(range(len(proc_id2ent))):
    ent = proc_id2ent[str(idx)]
    embedding = embedding_retriever(term=ent)
    embedding = np.array(embedding)
    proc_ent_emb.append(embedding)

stacked_embedding = np.vstack(proc_ent_emb)

emb_pkl = f"{file_dir}/entity_embedding.pkl"

with open(emb_pkl, "wb") as file:
    pickle.dump(stacked_embedding, file)
from get_emb import embedding_retriever
import numpy as np
from tqdm import tqdm
import pickle

## get embedding for procedure relations
proc_rel_emb = []

for idx in tqdm(range(len(proc_id2rel))):
    rel = proc_id2rel[str(idx)]
    embedding = embedding_retriever(term=rel)
    embedding = np.array(embedding)
    proc_rel_emb.append(embedding)

stacked_embedding = np.vstack(proc_rel_emb)

emb_pkl = f"{file_dir}/relation_embedding.pkl"

with open(emb_pkl, "wb") as file:
    pickle.dump(stacked_embedding, file)