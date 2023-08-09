from __future__ import print_function

import sys
import torch
import torch.nn as nn
from torch.nn.modules import loss
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import os
import argparse
from tqdm import tqdm, trange
import numpy as np
from PIL import Image
from get_model import get_model


'''
CUDA_VISIBLE_DEVICES=0 python3 test_image.py
'''

############## Initialize #####################
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--model', type=str, default='resnet50', help='cnn')
parser.add_argument('--img_path', type=str, default='', help='cnn')
parser.add_argument('--seed', default=0, type=int, help='random seed')

args = parser.parse_args()
print(args)

home_path = './'

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)

device = 'cuda' if torch.cuda.is_available() else 'cpu'


if __name__ == '__main__':
    torch.set_printoptions(precision=7)
    img_root = args.img_path
    print(img_root)
    img_list = os.listdir(img_root)
    img_list.sort()
    print('Total Images', len(img_list))

    # models = ['mnv2',  'inception_v3', 'resnet50', 'densenet161', 'resnet152', 'ef_b7', 'mvit', 'vit', 'swint', 'pvtv2'] #]
    models = ['mnv2'] # 

    for model in models:
        if model == 'vit':
            print('Using 0.5 Nor...')
            mean = [0.5, 0.5, 0.5]
            std = [0.5, 0.5, 0.5]
        elif model == 'mvit' or model == 'vitb_adv' or model == 'covnext_l_adv': 
            mean = [0, 0, 0]
            std = [1, 1, 1]
        else:
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]

        mean = torch.Tensor(mean).cuda()
        std = torch.Tensor(std).cuda()
            
        # Model
        net = get_model(model)

        if device == 'cuda':
            net.to(device)
            cudnn.benchmark = True
        net.eval()
        net.cuda()
        if model == 'inception_v3':
            image_size = (299, 299)
        elif model == 'mvit':
            image_size = (320, 320)
        else:
            image_size = (224, 224)
        # print(img_list)
        cnt = 0
        labels_f = open('third_party/Natural-Color-Fool/dataset/labels.txt').readlines()
        acc = 0

        for (i, img_p) in enumerate(img_list):
            pil_image = Image.open(os.path.join(img_root, img_p)).convert('RGB').resize(image_size)
            img = (torch.tensor(np.array(pil_image), device=device).unsqueeze(0)/255.).permute(0, 3, 1, 2)
            img = img - mean[None,:,None,None]
            img = img / std[None,:,None,None]
            out = net(img.cuda())
            _, predicted = out.max(1)
            idx = int(img_p.split('.')[0]) 
            label = int(labels_f[idx]) - 1
            if predicted[0] == label:
                acc += 1
        print('-' * 60)
        print(model, "ASR:", 100-acc/len(img_list)*100)


