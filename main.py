import torch
import torchvision
import os
from module import gradcam
from imageio import imread
import numpy as np
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
import cv2


def Normalize(img, inplace=False):
    changed_dim = np.transpose(img, (2, 0, 1)) / 255.
    Tensor = torch.from_numpy(changed_dim)

    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]

    normalized = F.normalize(Tensor, means, stds, inplace)

    return normalized.unsqueeze(0)


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

    origin = imread('/data_ssd3/LJH/pytorch_project/Grad-CAM-pytorch/data/cat.jpg')
    # normalize for pre-trained network.
    img = Normalize(origin.astype(np.float32))

    # for calculate gradient.
    img.requires_grad_(True)

    # forward.
    grad_cam.forward(img)

    # backward.
    grad_cam.backward()

    # calculate dy/dA
    with torch.no_grad():
        alpha_c_k = np.mean(grad_cam.get_activation_gradient()[0].numpy(), axis=(2, 3))
        A_k = grad_cam.get_activation()[0].detach().numpy()

        alpha_c_k = alpha_c_k.squeeze()
        A_k = A_k.squeeze()

        L_Grad_CAM = torch.nn.ReLU()(torch.from_numpy(cv2.resize(np.mean(np.multiply(A_k.transpose(1, 2, 0), alpha_c_k), axis=2), (224, 224), interpolation=cv2.INTER_LANCZOS4)))

        plt.subplot(121)
        plt.imshow(origin)
        plt.subplot(122)
        plt.imshow(L_Grad_CAM, cmap='jet')
        plt.show()
