import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torchmetrics import MetricCollection, Accuracy, Precision, Recall, F1Score

def get_metrics(n_classes):
    single_metric = MetricCollection({
        'acc_top1': Accuracy(task='multiclass', num_classes=n_classes, average='macro'),
        'acc_top5': Accuracy(task='multiclass', num_classes=n_classes, average='macro', top_k=5),
        'precision': Precision(task='multiclass', num_classes=n_classes, average='macro'),
        'recall': Recall(task='multiclass', num_classes=n_classes, average='macro')
        })
    return single_metric

def args_from_yaml(args, yml):
    with open(yml, 'r', encoding='utf-8') as f:
        dic = yaml.load(f.read(), Loader=yaml.Loader)
        for k in dic:
            setattr(args, k, dic[k])