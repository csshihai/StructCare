import random
from datetime import datetime

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
# from joblib import load
from  pyhealth.datasets import split_by_patient, get_dataloader
from pyhealth.medcode import InnerMap
from pyhealth.tokenizer import Tokenizer
from torch.utils.data import random_split, DataLoader, Subset



# 此文件修改的地方为67行加入了LABEVENTS以及将cond_hist对应变为drugs_hist
def get_label_tokenizer(label_tokens):
    special_tokens = []
    label_tokenizer = Tokenizer(
        label_tokens,
        special_tokens=special_tokens,
    )
    return label_tokenizer


def batch_to_multihot(label, num_labels: int) -> torch.tensor:
    multihot = torch.zeros((len(label), num_labels))
    for i, l in enumerate(label):
        multihot[i, l] = 1
    return multihot


def prepare_labels(
        labels,
        label_tokenizer: Tokenizer,
) -> torch.Tensor:
    labels_index = label_tokenizer.batch_encode_2d(
        labels, padding=False, truncation=False
    )
    num_labels = label_tokenizer.get_vocabulary_size()
    labels = batch_to_multihot(labels_index, num_labels)
    return labels


def parse_datetimes(datetime_strings):
    # print(datetime_strings)
    return [datetime.strptime(dt_str, "%Y-%m-%d %H:%M") for dt_str in datetime_strings]


def timedelta_to_str(tdelta):
    days = tdelta.days
    seconds = tdelta.seconds
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    return days * 1440 + hours * 60 + minutes


def convert_to_relative_time(datetime_strings):
    datetimes = parse_datetimes(datetime_strings)
    base_time = min(datetimes)
    return [timedelta_to_str(dt - base_time) for dt in datetimes]




# def get_init_tokenizers(task_dataset, keys=['drugs_hist', 'procedures', 'drugs']):
#     Tokenizers = {key: Tokenizer(tokens=task_dataset.get_all_tokens(key), special_tokens=["<pad>"]) for key in keys}
#     return Tokenizers
def get_init_tokenizers(task_dataset, keys=None):
    Tokenizers = {key: Tokenizer(tokens=task_dataset.get_all_tokens(key), special_tokens=["<pad>"]) for key in keys}
    return Tokenizers


def get_parent_tokenizers(task_dataset, keys=['cond_hist', 'procedures']):
    parent_tokenizers = {}
    dictionary = {'cond_hist': InnerMap.load("ICD9CM"), 'procedures': InnerMap.load("ICD9PROC")}
    for feature_key in keys:
        assert feature_key in dictionary.keys()
        tokens = task_dataset.get_all_tokens(feature_key)
        parent_tokens = set()
        for token in tokens:
            try:
                parent_tokens.update(dictionary[feature_key].get_ancestors(token))
            except:
                continue
        parent_tokenizers[feature_key + '_parent'] = Tokenizer(tokens=list(parent_tokens), special_tokens=["<pad>"])
    return parent_tokenizers


def split_dataset(dataset, train_ratio=0.75, valid_ratio=0.1, test_ratio=0.15):
    # Ensure the ratios sum to 1
    total = train_ratio + valid_ratio + test_ratio
    if total != 1.0:
        raise ValueError("Ratios must sum to 1.")
    total_size = len(dataset)
    train_size = int(total_size * train_ratio)
    valid_size = int(total_size * valid_ratio)
    test_size = total_size - train_size - valid_size

    # Randomly splitting the dataset
    train_set, valid_set, test_set = random_split(dataset, [train_size, valid_size, test_size])
    return train_set, valid_set, test_set


def custom_collate_fn(batch):
    sequence_data_list = [item[0] for item in batch]
    graph_data_list = [item[1] for item in batch]

    sequence_data_batch = {key: [d[key] for d in sequence_data_list if d[key] != []] for key in sequence_data_list[0]}

    graph_data_batch = graph_data_list

    return sequence_data_batch, graph_data_batch


# def mm_dataloader(trainset, validset, testset, batch_size=64):
#     train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
#     val_loader = DataLoader(validset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
#     test_loader = DataLoader(testset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
#
#     return train_loader, val_loader, test_loader


def seq_dataloader(dataset, split_ratio=[0.75, 0.1, 0.15], batch_size=64):
    train_dataset, val_dataset, test_dataset = split_by_patient(dataset, split_ratio)
    train_loader = get_dataloader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = get_dataloader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = get_dataloader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


# 纯gpt写的
def get_sample_loader(data_loader, sample_size):
    sample_size = round(len(data_loader.dataset) * sample_size)
    dataset = data_loader.dataset
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    random.shuffle(indices)  # 随机打乱索引
    sample_indices = indices[:sample_size]  # 取前 sample_size 个索引
    subset = Subset(dataset, sample_indices)
    sample_loader = DataLoader(subset, batch_size=data_loader.batch_size, shuffle=False,
                               collate_fn=data_loader.collate_fn)
    return sample_loader

def code_level(labels, predicts):
    labels = np.array(labels)
    total_labels = np.where(labels == 1)[0].shape[0]
    top_ks = [10, 20, 30, 40, 50, 60, 70, 80]
    total_correct_preds = []
    for k in top_ks:
        correct_preds = 0
        for i, pred in enumerate(predicts):
            index = np.argsort(-pred)[:k]
            for ind in index:
                if labels[i][ind] == 1:
                    correct_preds = correct_preds + 1
        total_correct_preds.append(float(correct_preds))

    total_correct_preds = np.array(total_correct_preds) / total_labels
    return total_correct_preds


def visit_level(labels, predicts):
    labels = np.array(labels)
    predicts = np.array(predicts)
    top_ks = [10, 20, 30, 40, 50, 60, 70, 80]
    precision_at_ks = []
    for k in top_ks:
        precision_per_patient = []
        for i in range(len(labels)):
            actual_positives = np.sum(labels[i])
            denominator = min(k, actual_positives)
            top_k_indices = np.argsort(-predicts[i])[:k]
            true_positives = np.sum(labels[i][top_k_indices])
            precision = true_positives / denominator if denominator > 0 else 0
            precision_per_patient.append(precision)
        average_precision = np.mean(precision_per_patient)
        precision_at_ks.append(average_precision)
    return precision_at_ks
def plot_losses(epoch_list, train_losses, val_losses, png_path):
    plt.figure(figsize=(10, 6))
    plt.plot(epoch_list, train_losses, label='Train Loss')
    plt.plot(epoch_list, val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss over Epochs')
    plt.legend()
    plt.grid(True)
    # 设置纵坐标范围
    plt.autoscale(True)
    # 保存绘图
    plt.savefig(png_path)
    plt.close()

def log_results(epoch, train_loss, val_loss, metrics, log_path):
    with open(log_path, 'a') as log_file:
        log_file.write(f'Epoch {epoch + 1}\n')
        log_file.write(f'Train Loss: {train_loss:.4f}\n')
        log_file.write(f'Validation Loss: {val_loss:.4f}\n')
        log_file.write(f'F1: {metrics["f1"]:.4f}, '
                       f'Jaccard: {metrics["jaccard"]:.4f}, '
                       f'ROC-AUC: {metrics["roc_auc"]:.4f}, '
                       f'PR-AUC: {metrics["pr_auc"]:.4f}\n')
        log_file.write('--------------------------------------\n')
def set_random_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # 这三句我都不知道干啥的
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True