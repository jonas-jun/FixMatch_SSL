import timm
import torch.nn as nn

def load_convnext(n_classes=41, pretrained=True):
    model = timm.create_model('convnext_small_384_in221ft1k', pretrained=pretrained)
    model.head.fc = nn.Linear(model.head.fc.in_features, n_classes, bias=True)
    return model