from .allcnn import *
from .resnet import *
from .vgg import *
from .vision_transformer import *
from .swin_transformer import *

import torch
from torchvision import models

def get_model(model_name, num_classes):
    if model_name == 'resnet18':
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'resnet50':
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "allcnn":
        model = AllCNN(n_channels=3, num_classes=num_classes, filters_percentage=0.5)
    elif model_name == "my-resnet18":
        model = resnet18(num_classes=num_classes)   # 默认不适用imagenet架构，代码中已经定义了fc
    elif model_name == "vgg16":
        model = vgg16_bn(num_classes=num_classes)  # NOTE: 必须使用bn版本的vgg，否则性能非常差
    elif model_name == "vgg11":
        model = vgg11_bn(num_classes=num_classes)
    elif model_name == "vit-s-16":  # 需要resize到224*224
        model = _vision_transformer(
            num_classes=num_classes,
            patch_size=16,
            num_layers=12, 
            num_heads=6,   
            hidden_dim=384, 
            mlp_dim=1536, 
            progress=False,
            weights=None, 
        )
    elif model_name == "swin-t":
        model = swin_tiny_patch4_window7_224(pretrained=False, num_classes=10)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return model

def load_model(model_path, model_name, num_classes):
    model_ckpt = torch.load( model_path , map_location="cuda")
    if isinstance(model_ckpt, dict):  # 为state dict
        model = get_model(model_name, num_classes)
        model_ckpt = {k.replace('module.', ''): v for k, v in model_ckpt.items()}
        model.load_state_dict(model_ckpt)
    else:
        model = model_ckpt
    model = model.to("cuda")
    return model