import torch
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score, roc_auc_score
import pandas as pd
import numpy as np
from torch_geometric.data import Data


def get_datasets():
    x = torch.tensor(pd.read_csv('data/attribute.csv').values,
                     dtype=torch.float)
    # 数据标准化
    x = (x - x.mean(0)) / x.std(0)
    #边
    edge = torch.tensor(pd.read_csv('data/graph.csv').values,
                        dtype=torch.long).T
    y = torch.full((x.shape[0], ), -1, dtype=torch.long)
    # 训练集
    train_data = pd.read_csv('data/train.csv')
    train_idx = torch.tensor(train_data['id'], dtype=torch.long)
    y[train_idx] = torch.tensor(train_data['label'], dtype=torch.long)
    # 验证集
    val_data = pd.read_csv('data/eval.csv')
    val_idx = torch.tensor(val_data['id'], dtype=torch.long)
    y[val_idx] = torch.tensor(val_data['label'], dtype=torch.long)
    # 测试集
    test_data = pd.read_csv('data/test_nolabel.csv')
    test_idx = torch.tensor(test_data['id'], dtype=torch.long)

    data = Data(x=x,
                y=y,
                edge_index=edge,
                train_idx=train_idx,
                val_idx=val_idx,
                test_idx=test_idx)
    return data


def metrics(target, pred):
    soft_pred = torch.softmax(pred.data, dim=1)[:, 1].cpu().numpy()
    pred = torch.argmax(pred.data, dim=1)
    # print(target.float().mean())
    # acc = (pred == target).float().mean().cpu().item() # 样本不均衡 正样本数量为0.0979 不能用准确率作为依据
    target, pred = target.cpu().numpy(), pred.cpu().numpy()
    f1 = f1_score(target, pred).item()
    recall = recall_score(target, pred).item()
    precision = precision_score(target, pred).item()
    auc = roc_auc_score(target, soft_pred).item()
    return dict(f1=f1, recall=recall, precision=precision, auc=auc)


if __name__ == '__main__':
    data = get_datasets()
    print(data)
