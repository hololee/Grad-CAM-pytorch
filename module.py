import torch.nn as nn
import torch


class gradcam(nn.Module):
    def __init__(self, model, activation_module):
        super().__init__()
        # for saved gradient
        self.list_gradient = []

        # for get activation.
        self.activation_a = []

        self.model = model
        self.activation_module = activation_module

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
            print(index_module)

            # apply module.
            x = module(x)

            if index_module == 'avgpool':
                # in avgpool case should change shape.
                x = x.view(x.shape[0], -1)

            # find activations.
            if index_module == self.activation_module:
                print(f'----- last conv -----')
                x.register_hook(self.func_grad)
                self.activation_a.append(x)

        self.output = x

    def backward(self):
        # TODO: set to 0 except argmax(output), to make gradcam for specific class
        class_filter = torch.zeros(self.output.shape)
        class_filter[:, torch.argmax(self.output)] = 1
        one_class = class_filter * self.output
        
        output = torch.sum(one_class)
        output.backward()
