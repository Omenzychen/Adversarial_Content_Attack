import torchvision
import torch
import os
import timm

def get_model(model):
    home_path = './'
    if model == 'resnet50':
        net = torchvision.models.resnet50()
        net.load_state_dict(torch.load(os.path.join(home_path, 'checkpoints/resnet50-0676ba61.pth')))
    elif model == 'mnv2':
        net = torchvision.models.mobilenet_v2()
        net.load_state_dict(torch.load(os.path.join(home_path, 'checkpoints/mobilenet_v2-b0353104.pth')))
    return net