import torch.nn as nn


class gradcam(nn.Module):
    def __init__(self, model, activation_module, activation_sub_module):
        super().__init__()
        self.list_gradient = []
        self.activation_a = []
        self.model = model
        self.activation_module = activation_module
        self.activation_sub_module = activation_sub_module

    def func_grad(self, gradient):
        self.list_gradient.append(gradient)

    def forward(self, x):

        # find activations and regist hook for gradient.
        for index_module, module in self.model._modules.items():

            # apply module.
            x = module(x)

            # find activations.
            if index_module == self.activation_module:
                for index_sub_module, sub_module in module._modules.items():

                    if index_sub_module == self.activation_sub_module:
                        x.register_hook(self.func_grad)
                        self.activation_a.append(x)
        return x
