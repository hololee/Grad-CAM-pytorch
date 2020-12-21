import torch
import torchvision
import os
from module import gradcam

if __name__ == '__main__':
    # download resnet weight to ./models
    os.environ['TORCH_HOME'] = './models/resnet50'

    # load resnet50. is pretrained on ImageNet.
    resnet = torchvision.models.resnet50(pretrained=True)

    grad_cam = gradcam(resnet, activation_module='layer4', activation_sub_module='2')

    # # last conv layer is layer4.
    # print(list(resnet._modules))  # 'layer4'
    #
    # # layer4th last block is 2.
    # print(list(resnet.layer4._modules))  # '2'
    # last layer of resnet is Y

    grad_cam.forward()
