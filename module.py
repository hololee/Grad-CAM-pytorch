import torch.nn as nn
import torch


class gradcam(nn.Module):
    def __init__(self, model, activation_module, activation_sub_module):
        super().__init__()
        # for saved gradient
        self.list_gradient = []

        # for get activation.
        self.activation_a = []

        self.model = model
        self.activation_module = activation_module
        self.activation_sub_module = activation_sub_module

        self.output = None

    def func_grad(self, gradient):
        self.list_gradient.append(gradient)

    def get_activation_gradient(self):
        return self.list_gradient

    def get_activation(self):
        return self.activation_a

    def forward(self, x):

        # find activations and regist hook for gradient.
        for index_module, module in self.model._modules.items():

            # apply module.
            x = module(x)

            if index_module == 'avgpool':
                x = x.view(x.shape[0], -1)

            # find activations.
            if index_module == self.activation_module:
                for index_sub_module, sub_module in module._modules.items():

                    if index_sub_module == self.activation_sub_module:
                        # find gradient of last Conv activation and regist grad hook for get  gradient of activation.
                        x.register_hook(self.func_grad)
                        self.activation_a.append(x)

        self.output = x

    def backward(self):
        # TODO: set to 0 except argmax(output), to make gradcam for specific class
        output = torch.sum(self.output)
        output.backward(retain_graph=True)
