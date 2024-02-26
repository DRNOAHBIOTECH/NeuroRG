from sys import exit
import torch
import torch.nn as nn
import timm
from torchvision.models import densenet201, inception_v3, resnet50, swin_v2_t
from efficientnet_pytorch import EfficientNet
import torch.optim as optim
from torch.optim import lr_scheduler


class ModelUtils:
    def __init__(self):
        pass

    @staticmethod
    def set_parameter_requires_grad(model, feature_extracting):
        if feature_extracting:
            for param in model.parameters():
                param.requires_grad = False

    @staticmethod
    def initialize_model(model_name, num_classes, use_pretrained=True):
        model_ft = None
        
        if model_name == 'convnext_tiny':
            model_ft = timm.create_model('convnext_nano', pretrained=use_pretrained, num_classes=num_classes)

        elif model_name == 'densenet201':
            model_ft = densenet201(pretrained = use_pretrained)
            num_features = model_ft.classifier.in_features
            model_ft.classifier = nn.Linear(num_features, 6)

        elif model_name == 'efficientnet_b0':
            model_ft = EfficientNet.from_pretrained('efficientnet-b0', num_classes=num_classes)

        elif model_name == 'efficientnet_b1':
            model_ft = EfficientNet.from_pretrained('efficientnet-b1', num_classes=num_classes)

        elif model_name == 'efficientnet_b2':
            model_ft = EfficientNet.from_pretrained('efficientnet-b2', num_classes=num_classes)

        elif model_name == 'efficientnet_b3':
            model_ft = EfficientNet.from_pretrained('efficientnet-b3', num_classes=num_classes)
            
        elif model_name == 'efficientnet_b4': 
            model_ft = EfficientNet.from_pretrained('efficientnet-b4', num_classes=num_classes)

        elif model_name == 'efficientnet_b5': 
            model_ft = EfficientNet.from_pretrained('efficientnet-b5', num_classes=num_classes)

        elif model_name == 'efficientnet_v2_s': 
            model_ft = efficientnet_v2_s(pretrained = use_pretrained)
            num_features = model_ft.classifier[1].in_features
            model_ft.classifier[1] = nn.Linear(num_features, num_classes)

        elif model_name == 'inception_v3': 
            model_ft = inception_v3(pretrained = use_pretrained)
            num_features = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_features, num_classes)

        elif model_name == 'maxvit_t':        
            model_ft = timm.create_model('maxvit_tiny_224', pretrained=use_pretrained, num_classes=num_classes)

        elif model_name == 'regnet_y_3_2gf':
            model_ft = regnet_y_3_2gf(pretrained = use_pretrained)
            num_features = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_features, num_classes)

        elif model_name == 'resnet50':
            model_ft = resnet50(pretrained = use_pretrained)
            num_features = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_features, num_classes)

        elif model_name == 'swin_v2_t':
            model_ft = swin_v2_t(pretrained = use_pretrained)
            num_features = model_ft.head.in_features
            model_ft.head = nn.Linear(num_features, num_classes)

        else:
            print("Invalid model name, exiting...")
            exit()
        
        return model_ft
    
    def get_optimizer(model_ft, model_name):    
        if model_name == 'convnext_tiny':
            optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.015, momentum=0.9)
        elif model_name == 'densenet201':
            optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.015, momentum=0.9)
        elif model_name.startswith('efficientnet'):
            optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.015, momentum=0.9)
        elif model_name == 'efficientnet_v2_s':
            optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.015, momentum=0.9)
        elif model_name == 'inception_v3':
            optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.015, momentum=0.9)
        elif model_name == 'mnasnet1_3':
            optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.015, momentum=0.9)
        elif model_name == 'maxvit_t':
            optimizer_ft = optim.Adam(model_ft.parameters(), lr=3e-2, weight_decay=0.05)
        elif model_name == 'regnet_y_3_2gf':
            optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.015, momentum=0.9)
        elif model_name == 'resnet50':
            optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.015, momentum=0.9)
        elif model_name == 'swin_v2_t':
            optimizer_ft = optim.AdamW(model_ft.parameters(), lr=0.01, weight_decay=0.05)
        else:
            print("Invalid model name, exiting...")
            exit()

        return optimizer_ft