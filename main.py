import argparse
import pickle

from torch_geometric.utils import from_networkx
from torch.optim.lr_scheduler import StepLR

# from StructCare_gnn1 import StructCareGnn1
# from StructCare_gnn4 import StructCareGnn4
from Struct_demo2 import StructDemo2
from Struct_demo3 import StructDemo3
from Struct_demo4 import StructDemo4
from Task import initialize_task
from my_baselines.Base2_1 import Base2_1
from my_baselines.Base2_2 import Base2_2
from utils import *
from baselines_.baselines import *
from my_baselines import *
from trainer import training, evaluating, testing, training_base, evaluating_base, testing_base
from data_preprocess.data_load import *
from subgraphs_sample import process_data_with_graph
from StructCare import StructCare, get_rel_emb
import os
import json
from data_preprocess.model_data_prepare import label_ehr_nodes

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=30, help='Number of epochs to train.')
parser.add_argument('--test_epochs', type=int, default=10, help='Number of epochs to test.')
parser.add_argument('--lr', type=float, default=5e-4, help='learning rate.')
parser.add_argument('--model', type=str, default="StructCare",
                    help='Transformer,StructDemo3,Struct'
                         'Demo4,StructDemo2,StructDemo5,Demo,RETAIN, StageNet, KAME, GCT, DDHGNN, TRANS, '
                         'GRU, Base, StructCare')
parser.add_argument('--device_id', type=int, default=1, help="选gpu编号的")
parser.add_argument('--seed', type=int, default=65)
parser.add_argument('--dataset', type=str, default="mimic3", choices=['mimic3', 'mimic4'])
parser.add_argument('--task', type=str, default="drug_rec", choices=['drug_rec', 'diag_pred'])
parser.add_argument('--batch_size', type=int, default=16, help='batch size')
# parser.add_argument('--dim', type=int, default=128, help='embedding dim')
parser.add_argument('--hp', type=float, default=0.000001, help='Hyperparameter')
parser.add_argument('--freeze', type=bool, default=False, help='freeze')
parser.add_argument('--dropout', type=float, default=0.7, help='dropout rate')
parser.add_argument('--gamma', type=float, default=0.2, help='gamma')
parser.add_argument('--step_size', type=int, default=20, help='step_size')
parser.add_argument('--developer', type=bool, default=False, help='developer mode')
parser.add_argument('--test', type=bool, default=True, help='test mode')

args = parser.parse_args()


def main(args):
    if args.developer:
        args.epochs = 3
        args.test_epochs = 2
        if not torch.cuda.is_available():
            args.batch_size = 2
    set_random_seed(args.seed)
    print('{}--{}--{}--{}--{}--{}'.format(args.model, args.task, args.dataset, args.batch_size, args.lr, args.hp
                                          ))
    # if torch.cuda.device_count() > 1:
    #     cuda_id = "cuda"  # 使用所有可用的 GPU
    # else:
    #     cuda_id = "cuda:" + str(args.device_id)  # 使用指定的 GPU
    cuda_id = args.device_id
    device = torch.device(cuda_id if torch.cuda.is_available() else "cpu")

    # 直接存在处理好的图和数据
    if args.model in ['StructCare', 'StructDemo2', 'StructDemo3', 'StructDemo4', 'StructCareGnn1', 'StructCareGnn4']:
        if os.path.exists(f"./data/{args.dataset}/processed_data/{args.task}/data_with_graph/processed_data.pkl"):
            print("Processed graph data exists, loading directly.")
            task_dataset_with_graph = load_preprocessed_data(
                f"./data/{args.dataset}/processed_data/{args.task}/data_with_graph/processed_data.pkl")
            # 任务定义
            Tokenizers_visit_event, Tokenizers_monitor_event, label_tokenizer, label_size = initialize_task(
                task_dataset_with_graph, args)
            for feature_key in Tokenizers_visit_event:
                tokenizer = Tokenizers_visit_event
            with open(f'./graph_generation/data/cond_proc_drug/CCSCM_CCSPROC_ATC3/clusters_rel_th015.json', 'r',
                      encoding='utf-8') as f:
                map_cluster_rel = json.load(f)
            with open(f'./graph_generation/data/cond_proc_drug/CCSCM_CCSPROC_ATC3/clusters_th015.json', 'r',
                      encoding='utf-8') as f:
                map_cluster = json.load(f)
            print("Getting embedding...")
            rel_emb = get_rel_emb(map_cluster_rel)
            node_emb = get_rel_emb(map_cluster)
            num_nodes = node_emb.shape[0]
            num_rels = rel_emb.shape[0]
        else:
            if args.developer:
                processed_data_path = f'./data/{args.dataset}/processed_data/{args.task}/processed_addVisitNode_developer_data.pkl'
            else:
                processed_data_path = f'./data/{args.dataset}/processed_data/{args.task}/processed_addVisitNode_data.pkl'

            if os.path.exists(processed_data_path):
                task_dataset = load_preprocessed_data(processed_data_path)
            else:
                print("文件出现错误请检查路径和model_data_prepare.py文件")

            with open(f'./graph_generation/data/cond_proc_drug/CCSCM_CCSPROC_ATC3/clusters_rel_th015.json', 'r',
                      encoding='utf-8') as f:
                map_cluster_rel = json.load(f)
            with open(f'./graph_generation/data/cond_proc_drug/CCSCM_CCSPROC_ATC3/clusters_th015.json', 'r',
                      encoding='utf-8') as f:
                map_cluster = json.load(f)
            print("Getting embedding...")
            rel_emb = get_rel_emb(map_cluster_rel)
            node_emb = get_rel_emb(map_cluster)
            num_nodes = node_emb.shape[0]
            num_rels = rel_emb.shape[0]

            # 再次处理数据集,对基础数据进行标记，这一步可以理解未补充大模型生成的医药代码
            with open('graph_generation/data/cond_proc_drug/CCSCM_CCSPROC_ATC3/ccscm_id2clus.json', 'r') as f:
                ccscm_id2clus = json.load(f)
            with open('graph_generation/data/cond_proc_drug/CCSCM_CCSPROC_ATC3/ccsproc_id2clus.json', 'r') as f:
                ccsproc_id2clus = json.load(f)
            with open('graph_generation/data/cond_proc_drug/CCSCM_CCSPROC_ATC3/atc3_id2clus.json', 'r') as f:
                atc3_id2clus = json.load(f)
            task_dataset = label_ehr_nodes(task_dataset, ccscm_id2clus, ccsproc_id2clus,
                                           atc3_id2clus, args.task)
            # 任务定义
            Tokenizers_visit_event, Tokenizers_monitor_event, label_tokenizer, label_size = initialize_task(
                task_dataset, args)
            for feature_key in Tokenizers_visit_event:
                tokenizer = Tokenizers_visit_event

            # 抽取子图，并加入数据
            # 这部分数据存在最终的data里面
            task_dataset_with_graph = \
                process_data_with_graph(task_dataset, Tokenizers_visit_event, Tokenizers_monitor_event, args)
    else:
        if args.developer:
            processed_data_path = f'./data/{args.dataset}/processed_data/{args.task}/processed_developer_data.pkll'
        else:
            processed_data_path = f'./data/{args.dataset}/processed_data/{args.task}/processed_data.pkl'

        if os.path.exists(processed_data_path):
            task_dataset_with_graph = load_preprocessed_data(processed_data_path)
    # 切分数据
    train_loader, val_loader, test_loader = seq_dataloader(task_dataset_with_graph, batch_size=args.batch_size)
    Tokenizers_visit_event, Tokenizers_monitor_event, label_tokenizer, label_size = initialize_task(
        task_dataset_with_graph, args)
    for feature_key in Tokenizers_visit_event:
        tokenizer = Tokenizers_visit_event

    """模型定义"""
    # TODO
    # 调通baseline
    # 还需要加入molerec，safedrug，trans（molerec和safedrug在pyhealth库里有，trans的代码在github上，也是用的这个库）
    if args.model == 'Transformer':
        model = Transformer(Tokenizers_visit_event, label_size, device)

    elif args.model == 'GRU':
        model = GRU(Tokenizers_visit_event, label_size, device)

    elif args.model == 'RETAIN':
        model = RETAIN(Tokenizers_visit_event, label_size, device)

    elif args.model == 'KAME':
        Tokenizers_visit_event.update(get_parent_tokenizers(task_dataset_with_graph))
        model = KAME(Tokenizers_visit_event, label_size, device)

    elif args.model == 'StageNet':
        model = StageNet(Tokenizers_visit_event, label_size, device)


    elif args.model == 'Base2_1':
        model = Base2_1(Tokenizers_visit_event, Tokenizers_monitor_event, label_size, device)

    elif args.model == 'Base2_2':
        model = Base2_2(Tokenizers_visit_event, Tokenizers_monitor_event, label_size, device)

    elif args.model == 'StructDemo2':
        model = StructDemo2(num_nodes=num_nodes, num_rels=num_rels, Tokenizers_visit_event=Tokenizers_visit_event,
                            Tokenizers_monitor_event=Tokenizers_monitor_event, output_size=label_size, device=device,
                            freeze=True, node_emb=node_emb,
                            rel_emb=rel_emb)
    elif args.model == 'StructDemo3':
        model = StructDemo3(Tokenizers_visit_event=Tokenizers_visit_event,
                            Tokenizers_monitor_event=Tokenizers_monitor_event, output_size=label_size, device=device
                            )
    elif args.model == 'StructDemo4':
        model = StructDemo4(num_nodes=num_nodes, num_rels=num_rels, Tokenizers_visit_event=Tokenizers_visit_event,
                            Tokenizers_monitor_event=Tokenizers_monitor_event, output_size=label_size, device=device,
                            freeze=True, node_emb=node_emb,
                            rel_emb=rel_emb)
    elif args.model == 'StructCare':
        model = StructCare(num_nodes=num_nodes, num_rels=num_rels, Tokenizers_visit_event=Tokenizers_visit_event,
                           Tokenizers_monitor_event=Tokenizers_monitor_event, output_size=label_size, device=device,
                           freeze=True, node_emb=node_emb,
                           rel_emb=rel_emb)
    # elif args.model == 'StructCareGnn1':
    #     model = StructCareGnn1(num_nodes=num_nodes, num_rels=num_rels, Tokenizers_visit_event=Tokenizers_visit_event,
    #                         Tokenizers_monitor_event=Tokenizers_monitor_event, output_size=label_size, device=device,
    #                         freeze=True, node_emb=node_emb,
    #                         rel_emb=rel_emb)
    # elif args.model == 'StructCareGnn4':
    #     model = StructCareGnn4(num_nodes=num_nodes, num_rels=num_rels, Tokenizers_visit_event=Tokenizers_visit_event,
    #                         Tokenizers_monitor_event=Tokenizers_monitor_event, output_size=label_size, device=device,
    #                         freeze=True, node_emb=node_emb,
    #                         rel_emb=rel_emb)
    else:
        print("没有这个模型")
        return

    if args.task == "drug_rec":
        label_name = 'drugs'
    else:
        label_name = 'conditions'
    # 保存checkpoint的路径
    folder_path = './logs/{}_{}_{}_{}_{}_{}'.format(args.model, args.task, args.dataset, args.epochs, args.batch_size,
                                                    args.lr)
    os.makedirs(folder_path, exist_ok=True)
    ckpt_path = f'{folder_path}/best_model.ckpt'
    png_path = f'{folder_path}/loss.png'
    txt_path = f'{folder_path}/final_result.txt'
    log_txt_path = f'{folder_path}/log.txt'
    # # 使用 DataParallel
    # model = nn.parallel.DataParallel(model)
    # model = model.to(device)
    if not args.test:
        # 记录 loss 的列表
        epoch_list = []
        train_losses = []
        val_losses = []
        print('--------------------Begin Training--------------------')
        # with open(
        #         "/home/workstation/sunqizheng/StructCare/logs/StructDemo2_drug_rec_mimic3_16_0.0005/best_model_epoch14_loss0.3898.ckpt",
        #         'rb') as Fin:model.load_state_dict(torch.load(Fin,map_location=device))
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        sched = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
        # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
        best = float('inf')  # 无限大
        best_model = None
        # best_jaccard = float('-inf')  # Initialize to a very low value
        for epoch in range(args.epochs):
            model = model.to(device)
            print(f'\nTraining Epoch {epoch + 1}/{args.epochs}')
            if args.model in ['StructDemo2', 'StructCare', 'StructCareGnn1', 'StructCareGnn4']:
                train_loss = training(train_loader, model, label_tokenizer, optimizer, label_name, device, args.hp)
                val_loss, avg_clustering_loss, avg_classification_loss, metrics, code_level_results, visit_level_results = evaluating(
                    val_loader, model,
                    label_tokenizer, label_name, device, args.hp)
                code_level_results = ', '.join(map(lambda x: f"{x:.4f}", code_level_results))
                visit_level_results = ', '.join(map(lambda x: f"{x:.4f}", visit_level_results))
                # 打印结果，包括聚类损失
                print(f'Validation Loss: {val_loss:.4f}, Avg Clustering Loss: {avg_clustering_loss:.4f}, Avg '
                      f'Classification_loss: {avg_classification_loss:.4f}')
                print(f'F1: {metrics["f1"]:.4f}, '
                      f'Jaccard: {metrics["jaccard"]:.4f}, '
                      f'ROC-AUC: {metrics["roc_auc"]:.4f}, '
                      f'PR-AUC: {metrics["pr_auc"]:.4f}, '
                      f'code_level: {code_level_results}, '
                      f'visit_level: {visit_level_results},'
                      )

                # 记录结果到 log.txt
                log_results(epoch, train_loss, val_loss, metrics, log_txt_path)
                if val_loss < best:
                    best = val_loss
                    ckpt_path = f'{folder_path}/best_model_epoch{epoch + 1}_loss{val_loss:.4f}.ckpt'
                    best_model = model.state_dict()
                # if (epoch+1)%10==0:
                #     torch.save(best_model, ckpt_path)
                # if  metrics["jaccard"] > best_jaccard:
                #     ckpt_path = f'{folder_path}/best_model_epoch{epoch + 1}_loss{val_loss:.4f}.ckpt'
                #     best_jaccard = metrics["jaccard"]
                #     best_model = model.state_dict()

                # 记录损失
                epoch_list.append(epoch + 1)
                train_losses.append(train_loss)
                val_losses.append(val_loss)

                # 每个epoch都绘制一次，绘制损失曲线
                plot_losses(epoch_list, train_losses, val_losses, png_path)
            else:
                train_loss = training_base(train_loader, model, label_tokenizer, optimizer, label_name, device)
                val_loss, metrics, code_level_results, visit_level_results = evaluating_base(val_loader, model,
                                                                                             label_tokenizer,
                                                                                             label_name, device)
                # 新指标
                code_level_results = ', '.join(map(lambda x: f"{x:.4f}", code_level_results))
                visit_level_results = ', '.join(map(lambda x: f"{x:.4f}", visit_level_results))
                # 打印结果
                print(f'F1: {metrics["f1"]:.4f}, '
                      f'Jaccard: {metrics["jaccard"]:.4f}, '
                      f'ROC-AUC: {metrics["roc_auc"]:.4f}, '
                      f'PR-AUC: {metrics["pr_auc"]:.4f}, '
                      f'code_level: {code_level_results}, '
                      f'visit_level: {visit_level_results},'
                      )
                # 记录结果到 log.txt
                log_results(epoch, train_loss, val_loss, metrics, log_txt_path)
                if val_loss < best:
                    best = val_loss
                    ckpt_path = f'{folder_path}/best_model_epoch{epoch + 1}_loss{val_loss:.4f}.ckpt'
                    best_model = model.state_dict()
                # if  metrics["jaccard"] > best_jaccard:
                #     ckpt_path = f'{folder_path}/best_model_epoch{epoch + 1}_loss{val_loss:.4f}.ckpt'
                #     best_jaccard = metrics["jaccard"]
                #     best_model = model.state_dict()

                # 记录损失
                epoch_list.append(epoch + 1)
                train_losses.append(train_loss)
                val_losses.append(val_loss)

                # 每个epoch都绘制一次，绘制损失曲线
                plot_losses(epoch_list, train_losses, val_losses, png_path)
            sched.step()
        torch.save(best_model, ckpt_path)
    print('--------------------Begin Testing--------------------')
    if args.test:
        ckpt_path = './logs/best_model_epoch16_loss0.2435.ckpt'
    # 读取最新的model
    best_model = torch.load(ckpt_path)
    model.load_state_dict(best_model)
    model = model.to(device)

    # 开始测试
    sample_size = 0.8  # 国际惯例选取0.8
    if args.model in ['StructDemo2', 'StructCare', 'StructCareGnn1', 'StructCareGnn4']:
        outstring = testing(test_loader, args.test_epochs, model, label_tokenizer, sample_size, label_name, device,
                            args.hp)
    else:
        outstring = testing_base(test_loader, args.test_epochs, model, label_tokenizer, sample_size, label_name, device)
    # 输出结果
    print("\nFinal test result:")
    print(outstring)
    with open(txt_path, 'w+') as file:
        file.write("model_path:")
        file.write(ckpt_path)
        file.write('\n')
        file.write(outstring)


if __name__ == '__main__':
    main(args)
