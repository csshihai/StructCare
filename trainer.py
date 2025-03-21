import dill
import numpy as np
import torch
import torch.nn.functional as F
from pyhealth.metrics import binary_metrics_fn
from tqdm import tqdm
from torch import autograd
# from pyhealth.datasets import MIMIC4Dataset, MIMIC3Dataset
from utils import prepare_labels, get_sample_loader, visit_level, code_level
# def train(data_loader, model, label_tokenizer, optimizer, device):
#     train_loss = 0
#     for data in data_loader:
#         model.train()
#         optimizer.zero_grad()
#         if type(data) == dict:
#             label = prepare_labels(data['conditions'], label_tokenizer).to(device)
#         else:
#             label = prepare_labels(data[0]['conditions'], label_tokenizer).to(device)
#         out = model(data)
#         loss = F.binary_cross_entropy_with_logits(out, label)
#         # y_prob = torch.sigmoid(out)
#         loss.backward()
#         optimizer.step()
#         train_loss += loss.detach().cpu().numpy()
#     return train_loss
import torch
import torch.nn.functional as F
from tqdm import tqdm


def training(data_loader, model, label_tokenizer, optimizer,label_name,device, hp):
    model.train()
    train_loss = 0
    clustering_loss_total = 0  # 初始化聚类损失
    total_classification_loss = 0  # 新增总分类损失
    with tqdm(total=len(data_loader), desc="Training", unit="batch") as pbar:
        for batch_idx, data in enumerate(data_loader):
            optimizer.zero_grad()

            if type(data) == dict:
                label = prepare_labels(data[label_name], label_tokenizer).to(device)
            else:
                label = prepare_labels(data[0][label_name], label_tokenizer).to(device)

            out, clustering_loss = model(data)
            if(clustering_loss != 0):
                clustering_loss_total += clustering_loss.item()  # 累加聚类损失

            classification_loss = F.binary_cross_entropy_with_logits(out, label)
            total_classification_loss += classification_loss.item()  # 新增总分类损失
            # 动态调整聚类损失的权重
            lambda_clustering = hp  # 这个值可以根据实验调整
            # 计算总损失
            total_loss = classification_loss + lambda_clustering * clustering_loss  # 这里将聚类损失加入总损失
            # 反向传播总损失
            total_loss.backward()
            optimizer.step()

            train_loss += total_loss.detach().cpu().numpy()


            # 清理显存
            del data, out, classification_loss, clustering_loss
            torch.cuda.empty_cache()

            # 更新进度条
            pbar.update(1)
            pbar.set_postfix(avg_loss=f"{train_loss / (batch_idx + 1):.4f}",
                             avg_clustering_weight_loss=f"{clustering_loss_total * lambda_clustering / (batch_idx + 1):.4f}",
                             avg_classification_loss=f"{total_classification_loss / (batch_idx + 1):.4f}", )

        avg_loss = train_loss / len(data_loader)




    return avg_loss


def training_base(data_loader, model, label_tokenizer, optimizer, label_name, device):
    model.train()
    train_loss = 0

    with tqdm(total=len(data_loader), desc="Training", unit="batch") as pbar:
        for batch_idx, data in enumerate(data_loader):
            optimizer.zero_grad()

            if type(data) == dict:
                label = prepare_labels(data[label_name], label_tokenizer).to(device)
            else:
                label = prepare_labels(data[0][label_name], label_tokenizer).to(device)

            out = model(data)

            # 计算分类损失
            classification_loss = F.binary_cross_entropy_with_logits(out, label)

            # 反向传播总损失
            classification_loss.backward()
            optimizer.step()

            train_loss += classification_loss.item()
            avg_loss = train_loss / (batch_idx + 1)

            # 清理显存
            del data, out, classification_loss
            torch.cuda.empty_cache()

            # 更新进度条
            pbar.update(1)
            pbar.set_postfix(avg_loss=f"{avg_loss:.4f}")

    return avg_loss

def evaluating_base(data_loader, model, label_tokenizer, label_name, device):
    model.eval()
    val_loss = 0
    y_t_all, y_p_all = [], []
    with torch.no_grad():
        with tqdm(total=len(data_loader), desc="Evaluating", unit="batch") as pbar:
            for batch_idx, data in enumerate(data_loader):
                if type(data) == dict:
                    label = prepare_labels(data[label_name], label_tokenizer).to(device)
                else:
                    label = prepare_labels(data[0][label_name], label_tokenizer).to(device)

                out = model(data)
                loss = F.binary_cross_entropy_with_logits(out, label)

                val_loss += loss.detach().cpu().numpy()
                avg_loss = val_loss / (batch_idx + 1)

                y_t = label.cpu().numpy()
                y_p = torch.sigmoid(out).detach().cpu().numpy()
                y_t_all.append(y_t)
                y_p_all.append(y_p)

                # 网上搜的，清理显存，不然显存会叠起来，100g也不够用
                del data, out, loss
                torch.cuda.empty_cache()

                # 更新进度条
                pbar.update(1)
                pbar.set_postfix(avg_loss=f"{avg_loss:.4f}")

            y_true = np.concatenate(y_t_all, axis=0)
            y_prob = np.concatenate(y_p_all, axis=0)


            code_level_results = code_level(y_true, y_prob)
            visit_level_results = visit_level(y_true, y_prob)

            y_true = y_true.ravel()
            y_prob = y_prob.ravel()

            metrics = binary_metrics_fn(y_true, y_prob,
                                        metrics=["f1", "jaccard", "roc_auc", "pr_auc"])

    return avg_loss, metrics, code_level_results, visit_level_results


def evaluating(data_loader, model, label_tokenizer, label_name,device, hp):
    model.eval()
    val_loss = 0
    clustering_loss_total = 0  # 初始化聚类损失
    total_classification_loss = 0  # 新增总分类损失
    y_t_all, y_p_all = [], []

    with torch.no_grad():
        with tqdm(total=len(data_loader), desc="Evaluating", unit="batch") as pbar:
            for batch_idx, data in enumerate(data_loader):
                if type(data) == dict:
                    label = prepare_labels(data[label_name], label_tokenizer).to(device)
                else:
                    label = prepare_labels(data[0][label_name], label_tokenizer).to(device)

                out, clustering_loss = model(data)  # 获取聚类损失
                classification_loss = F.binary_cross_entropy_with_logits(out, label)
                total_classification_loss += classification_loss.item()  # 新增总分类损失
                # 动态调整聚类损失的权重
                lambda_clustering = hp  # 这个值可以根据实验调整
                # 计算总损失
                total_loss = classification_loss + lambda_clustering * clustering_loss  # 这里将聚类损失加入总损失
                val_loss += total_loss.detach().cpu().numpy()
                if clustering_loss is not None:
                    if isinstance(clustering_loss, torch.Tensor):
                        clustering_loss_total += clustering_loss.item()
                    else:
                        clustering_loss_total += clustering_loss
                # 累加聚类损失

                y_t = label.cpu().numpy()
                y_p = torch.sigmoid(out).detach().cpu().numpy()
                y_t_all.append(y_t)
                y_p_all.append(y_p)

                # 清理显存
                del data, out, classification_loss, clustering_loss
                torch.cuda.empty_cache()

                # 更新进度条
                pbar.update(1)
                pbar.set_postfix(avg_loss=f"{val_loss / (batch_idx + 1):.4f}",avg_clustering_weight_loss=f"{clustering_loss_total*lambda_clustering / (batch_idx + 1):.4f}",avg_classification_loss=f"{total_classification_loss / (batch_idx + 1):.4f}",)

            avg_loss = val_loss / len(data_loader)
            avg_clustering_weight_loss = clustering_loss_total * lambda_clustering / len(data_loader)
            avg_classification_loss = total_classification_loss / len(data_loader)  # 计算平均分类损失

            y_true = np.concatenate(y_t_all, axis=0)
            y_prob = np.concatenate(y_p_all, axis=0)


            code_level_results = code_level(y_true, y_prob)
            visit_level_results = visit_level(y_true, y_prob)

            y_true = y_true.ravel()
            y_prob = y_prob.ravel()

            metrics = binary_metrics_fn(y_true, y_prob,
                                        metrics=["f1", "jaccard", "roc_auc", "pr_auc"])
            with open('./logs/y_t_all.pkl', 'wb') as file:
                dill.dump(y_t_all, file)
            with open('./logs/y_p_all.pkl', 'wb') as file:
                dill.dump(y_p_all, file)
            with open('./logs/test_data', 'wb') as file:
                dill.dump(data_loader, file)
        return avg_loss, avg_clustering_weight_loss, avg_classification_loss, metrics,code_level_results, visit_level_results


def testing(data_loader, test_epochs, model, label_tokenizer, sample_size, label_name, device,hp):
    results = []

    for epoch in range(test_epochs):
        print(f'\nTesting Epoch {epoch + 1}/{test_epochs}')
        # 这里测试的时候打乱过，为了做case study shuffle等于false
        sample_loader = get_sample_loader(data_loader, sample_size)

        avg_loss, _, _, metrics,code_level_results, visit_level_results = evaluating(sample_loader, model, label_tokenizer, label_name, device,hp)

        # 打印结果
        print(f'F1: {metrics["f1"]:.4f}, '
              f'Jaccard: {metrics["jaccard"]:.4f}, '
              f'ROC-AUC: {metrics["roc_auc"]:.4f}, '
              f'PR-AUC: {metrics["pr_auc"]:.4f}, '
              f'Avg Loss: {avg_loss:.4f}')

        results.append([metrics["f1"], metrics["jaccard"], metrics["roc_auc"],
                        metrics["pr_auc"]] + list(code_level_results) + list(visit_level_results)
            )

    results = np.array(results)
    mean, std = results.mean(axis=0), results.std(axis=0)

    metric_list = ['F1', 'Jaccard', 'ROC-AUC', 'PR-AUC',
                   'code-10', 'code-20', 'code-30', 'code-40', 'code-50', 'code-60', 'code-70', 'code-80',
                   'visit-10', 'visit-20', 'visit-30', 'visit-40', 'visit-50', 'visit-60', 'visit-70', 'visit-80'
                  ]
    outstring = ''.join([
        "{}:\t{:.4f} $\\pm$ {:.4f} & \n".format(metric_list[idx], m, s)
        for idx, (m, s) in enumerate(zip(mean, std))
    ])
    return outstring


def testing_base(data_loader, test_epochs, model, label_tokenizer, sample_size, label_name, device):
    results = []
    for epoch in range(test_epochs):
        print(f'\nTesting Epoch {epoch + 1}/{test_epochs}')
        sample_loader = get_sample_loader(data_loader, sample_size)

        _, metrics,code_level_results, visit_level_results = evaluating_base(sample_loader, model, label_tokenizer, label_name, device)

        # 打印结果
        print(f'F1: {metrics["f1"]:.4f}, '
              f'Jaccard: {metrics["jaccard"]:.4f}, '
              f'ROC-AUC: {metrics["roc_auc"]:.4f}, '
              f'PR-AUC: {metrics["pr_auc"]:.4f}')

        results.append([metrics["f1"], metrics["jaccard"], metrics["roc_auc"],
                        metrics["pr_auc"]] + list(code_level_results) + list(visit_level_results)
                       )

    results = np.array(results)
    mean, std = results.mean(axis=0), results.std(axis=0)

    metric_list = ['F1', 'Jaccard', 'ROC-AUC', 'PR-AUC',
                   'code-10', 'code-20', 'code-30', 'code-40', 'code-50', 'code-60', 'code-70', 'code-80',
                   'visit-10', 'visit-20', 'visit-30', 'visit-40', 'visit-50', 'visit-60', 'visit-70', 'visit-80'
                   ]
    outstring = ''.join([
        "{}:\t{:.4f} $\\pm$ {:.4f} & \n".format(metric_list[idx], m, s)
        for idx, (m, s) in enumerate(zip(mean, std))
    ])

    return outstring
