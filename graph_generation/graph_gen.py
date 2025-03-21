import csv
import re
from ChatGPT import ChatGPT
import json
from tqdm import tqdm
import os

def extract_data_in_brackets(input_string):
    # 一种正则表达式目的是提取[]里面的内容
    pattern = r"\[(.*?)\]"
    matches = re.findall(pattern, input_string)
    return matches

def divide_text(long_text, max_len=800):
    sub_texts = []
    start_idx = 0
    while start_idx < len(long_text):
        end_idx = start_idx + max_len
        sub_text = long_text[start_idx:end_idx]
        sub_texts.append(sub_text)
        start_idx = end_idx
    return sub_texts


#定义了一个图生成方法，来生成每次交互问答的三元组
def graph_gen(term: str, mode: str):
    if mode == "condition":
        example = \
        """
        Example:
        prompt: systemic lupus erythematosus
        updates: [[systemic lupus erythematosus, is an, autoimmune condition], [systemic lupus erythematosus, may cause, nephritis], [anti-nuclear antigen, is a test for, systemic lupus erythematosus], [systemic lupus erythematosus, is treated with, steroids], [methylprednisolone, is a, steroid]]
        """
    elif mode == "procedure":
        example = \
        """
        Example:
        prompt: endoscopy
        updates: [[endoscopy, is a, medical procedure], [endoscopy, used for, diagnosis], [endoscopic biopsy, is a type of, endoscopy], [endoscopic biopsy, can detect, ulcers]]
        """
    elif mode == "drug":
        example = \
        """
        Example:
        prompt: iobenzamic acid
        updates: [[iobenzamic acid, is a, drug], [iobenzamic acid, may have, side effects], [side effects, can include, nausea], [iobenzamic acid, used as, X-ray contrast agent], [iobenzamic acid, formula, C16H13I3N2O3]]
        """
    chatgpt = ChatGPT()
    response = chatgpt.chat(
        f"""
            Given a prompt (a medical condition/procedure/drug), extrapolate as many relationships as possible from it and provide a list of updates.
            The relationships should be helpful for healthcare prediction (e.g., drug recommendation, mortality prediction, readmission prediction, etc.).
            Each update should be exactly in the format of [ENTITY 1, RELATIONSHIP, ENTITY 2]. The relationship is directed, so the order matters.
            Both ENTITY 1 and ENTITY 2 should be nouns.
            Any element in [ENTITY 1, RELATIONSHIP, ENTITY 2] should be conclusive, make it as short as possible.
            Do this in both breadth and depth. Expand [ENTITY 1, RELATIONSHIP, ENTITY 2] until the size reaches 100.
            All generated triples should be in English.

            {example}:
            prompt: Diabetes
            updates:
            [Diabetes, increases risk, Heart Disease]
            [Diabetes, causes, Hyperglycemia]
            [Hyperglycemia, leads to, Nerve Damage]
            [Nerve Damage, causes, Neuropathy]
            ...

            prompt: Hypertension
            updates:
            [Hypertension, increases risk, Stroke]
            [Hypertension, leads to, Heart Failure]
            [Heart Failure, requires, Medication]
            [Hypertension, associated with, Kidney Disease]
            [Kidney Disease, leads to, Dialysis]
        """
        )
    json_string = str(response)
    # json_data = json.loads(json_string)

    triples = extract_data_in_brackets(json_string)
    outstr = ""
    for triple in triples:
        outstr += triple.replace('[', '').replace(']', '').replace(', ', '\t') + '\n'

    return outstr

condition_mapping_file = "../resources/CCSCM.csv"
procedure_mapping_file = "../resources/CCSPROC.csv"
drug_file = "../resources/ATC.csv"
# name是名字，code是编码1，2，3.... ，生成三个字典
condition_dict = {}
with open(condition_mapping_file, newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        condition_dict[row['code']] = row['name'].lower()


procedure_dict = {}
with open(procedure_mapping_file, newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        procedure_dict[row['code']] = row['name'].lower()


drug_dict = {}
with open(drug_file, newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        if row['level'] == '3.0':
            drug_dict[row['code']] = row['name'].lower()

for key in tqdm(condition_dict.keys()):
    file = f'../graphs/condition/CCSCM/{key}.txt'
    # 这里是不是就是取并集的过程
    if os.path.exists(file):
        with open(file=file, mode="r", encoding='utf-8') as f:
            prev_triples = f.read()
        if len(prev_triples.split('\n')) < 100:
            outstr = graph_gen(term=condition_dict[key], mode="condition")
            outfile = open(file=file, mode='w', encoding='utf-8')
            outstr = prev_triples + outstr
            # print(outstr)
            outfile.write(outstr)
    else:
        outstr = graph_gen(term=condition_dict[key], mode="condition")
        outfile = open(file=file, mode='w', encoding='utf-8')
        outstr = outstr
        # print(outstr)
        outfile.write(outstr)


for key in tqdm(procedure_dict.keys()):
    file = f'../graphs/procedure/CCSPROC/{key}.txt'
    if os.path.exists(file):
        with open(file=file, mode="r", encoding='utf-8') as f:
            prev_triples = f.read()
        if len(prev_triples.split('\n')) < 150:
            outstr = graph_gen(term=procedure_dict[key], mode="procedure")
            outfile = open(file=file, mode='w', encoding='utf-8')
            outstr = prev_triples + outstr
            # print(outstr)
            outfile.write(outstr)
    else:
        outstr = graph_gen(term=procedure_dict[key], mode="procedure")
        outfile = open(file=file, mode='w', encoding='utf-8')
        outstr = outstr
        # print(outstr)
        outfile.write(outstr)


for key in tqdm(drug_dict.keys()):
    file = f'../graphs/drug/ATC3/{key}.txt'
    if os.path.exists(file):
        with open(file=file, mode="r", encoding='utf-8') as f:
            prev_triples = f.read()
        if len(prev_triples.split('\n')) < 150:
            outstr = graph_gen(term=drug_dict[key], mode="drug")
            outfile = open(file=file, mode='w', encoding='utf-8')
            outstr = prev_triples + outstr
            # print(outstr)
            outfile.write(outstr)
        # continue
    else:
        outstr = graph_gen(term=drug_dict[key], mode="drug")
        outfile = open(file=file, mode='w', encoding='utf-8')
        outstr = outstr
        # print(outstr)
        outfile.write(outstr)