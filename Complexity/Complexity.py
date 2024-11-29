"""
========================================================================================================================
Package
========================================================================================================================
"""
import time
import math

from typing import Literal

import torch
from torch import nn
from torch import Tensor
from torch.nn import Module

from torchvision import models
"""
========================================================================================================================
Complexity Calculator
========================================================================================================================
"""
class Complexity():

    """
    ====================================================================================================================
    Initialization
    ====================================================================================================================
    """
    def __init__(self, model: Module, dummy: Tensor) -> None:
    
        # Model and Dummy Input
        self.model = model
        self.dummy = dummy

        # FLOPs
        self.flops = 0

        # Calculate FLOPs
        self.hook_layer()

        return
    
    """
    ====================================================================================================================
    Register Forword Hook
    ====================================================================================================================
    """
    def hook_layer(self) -> None:

        """
        ----------------------------------------------------------------------------------------------------------------
        Calculate FLOPs
        ----------------------------------------------------------------------------------------------------------------
        """
        def forward_hook(module: Module, feature_in: tuple[Tensor], feature_out: Tensor) -> None:

            # Conver Tuple to Tensor
            feature_in = feature_in[0]

            # Convolution 2D & Transpose Convolution 2D
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                # Input Channels
                channel_in = feature_in.shape[1]
                # Kernel Size
                kernel = module.kernel_size[0]
                # FLOPs
                flops = 2 * kernel * kernel * channel_in * feature_out.numel()

            # Fully Connected
            elif isinstance(module, nn.Linear):
                # FLOPs
                flops = 2 * feature_in.numel() * feature_out.numel()

            # Batch Normalization
            elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                # FLOPs
                flops = 4 * feature_out.numel()

            # Instance Normalization
            elif isinstance(module, (nn.InstanceNorm1d, nn.InstanceNorm2d)):
                # FLOPs
                flops = 7 * feature_out.numel()

            # Group Normalization
            elif isinstance(module, nn.GroupNorm):
                # FLOPs
                flops = 4 * feature_out.numel()

            # # Layer Normalization (Self-Defined)
            # elif isinstance(module, LayerNorm):
            #     # FLOPs
            #     flops = 7 * feature_out.numel()

            # Dropout
            elif isinstance(module, (nn.Dropout, nn.Dropout1d, nn.Dropout2d)):
                # FLOPs
                flops = 2 * feature_out.numel()

            # Activation (Only Estimation)
            elif isinstance(module, (nn.ReLU, nn.LeakyReLU, nn.GELU, nn.SELU, nn.SiLU, nn.Tanh, nn.Sigmoid, nn.Softmax2d, nn.Softmax)):
                # FLOPs
                flops = 4 * feature_out.numel()

            # Pooling 
            elif isinstance(module, (nn.MaxPool2d, nn.AvgPool2d)):
                # Kernel Size
                kernel = module.kernel_size
                # FLOPs
                flops = kernel * kernel * feature_out.numel()

            # Pooling
            elif isinstance(module, (nn.AdaptiveAvgPool2d, nn.AdaptiveMaxPool2d)):
                # Check Shape
                if (feature_in.shape[2] != feature_in.shape[3] or feature_out.shape[2] != feature_out.shape[3]):
                    raise TypeError('Invalid Input or Output Shape')
                # kernel Size
                kernel = math.ceil(feature_in.shape[3] / feature_out.shape[3])
                # FLOPs
                flops = kernel * kernel * feature_out.numel()

            # Upsample
            elif isinstance(module, nn.Upsample):

                # Nearest Interpolation
                if module.mode == 'nearest':
                    # FLOPs
                    flops = 0
                # Bilinear Interpolation
                elif module.mode == 'bilinear':
                    # FLOPs
                    flops = 4 * feature_out.numel()
                else:
                    raise TypeError('Invalid Mode for Upsample')
            
            # Nearest Interpolation
            elif isinstance(module, nn.UpsamplingNearest2d):
                # FLOPs
                flops = 0

            # Bilinear Interpolation
            elif isinstance(module, nn.UpsamplingBilinear2d):
                # FLOPs
                flops = 4 * feature_out.numel()

            # Flatten
            elif isinstance(module, nn.Flatten):
                # FLOPs
                flops = 0

            # Others
            else:
                # Check Layer Name
                print(module._get_name())
                # FLOPs
                flops = 0

            # Total FLOPs
            self.flops += flops

            return

        """
        ----------------------------------------------------------------------------------------------------------------
        Break Down Model into Layers
        ----------------------------------------------------------------------------------------------------------------
        """
        def break_block(block: Module) -> None:
            
            # Block is Indeed a Block
            if list(block.children()):
                # Iterate Through Layer in Block
                for child in block.children():
                    # Recursive Break Down
                    break_block(child)

            # Block is a Layer
            else:
                # Register Hook
                block.register_forward_hook(forward_hook)

                return
                
        return break_block(self.model)

    """
    ====================================================================================================================
    Profile
    ====================================================================================================================
    """
    def profile(self, order: str | Literal['G', 'M', 'k'] = 'M', num_input: int = 1) -> None:

        """
        ----------------------------------------------------------------------------------------------------------------
        FLOPs and FLOPS
        ----------------------------------------------------------------------------------------------------------------
        """
        # Dummy Input
        dummy = torch.rand((1, *self.dummy))
        dummy = [dummy for _ in range(num_input)]

        # Start Time
        time_start = time.time()

        # Forward Pass
        for _ in range(10):
            self.flops = 0
            self.model(*dummy)

        # End Time
        time_end = time.time()

        # FLOPs and FLOPs per Second
        flops = self.flops
        flops_per_sec = self.flops / ((time_end - time_start) / 10)

        if order == 'G':
            flops = round(flops * 1e-9, 3)
            flops_per_sec = round(flops_per_sec * 1e-9, 3)
        elif order == 'M':
            flops = round(flops * 1e-6, 3)
            flops_per_sec = round(flops_per_sec * 1e-6, 3)
        elif order == 'k':
            flops = round(flops * 1e-3, 3)
            flops_per_sec = round(flops_per_sec * 1e-3, 3)
        else:
            raise TypeError('Invalid Order')

        """
        ----------------------------------------------------------------------------------------------------------------
        Parameter
        ----------------------------------------------------------------------------------------------------------------
        """
        # Total Parameter & Trainable Parameter
        num_params = sum(params.numel() for params in self.model.parameters())
        num_params_train = sum(params.numel() for params in self.model.parameters() if params.requires_grad)

        """
        ----------------------------------------------------------------------------------------------------------------
        Output Log
        ----------------------------------------------------------------------------------------------------------------
        """
        # Output Format
        space = "{:^25}|{:^25}|{:^25}|{:^25}"

        # Title
        print('-------------------------------------------------------------------------------------------------------')
        print(space.format(order + 'FLOPs', order + 'FLOPs', 'Total Parameters', 'Trainable Parameters'))
        print('-------------------------------------------------------------------------------------------------------')

        # Output Log
        print(space.format(flops, flops_per_sec, num_params, num_params_train))
        print()

        return 


"""
========================================================================================================================
Main Function
========================================================================================================================
"""
if __name__ == '__main__':

    # Model
    all_model = {
                 'vgg11': models.vgg11(),
                 'vgg13': models.vgg13(),
                 'vgg16': models.vgg16(),
                 'vgg19': models.vgg19(),
                 'resnet18': models.resnet18(),
                 'resnet34': models.resnet34(),
                 'resnet50': models.resnet50(),
                 'resnet101': models.resnet101(),
                 'resnet152': models.resnet152(),
                 'densenet121': models.densenet121(),
                 'densenet169': models.densenet169(),
                 'densenet201': models.densenet201(),
                 'densenet161': models.densenet161(),
                }

    for name, model in all_model.items():

        print()
        print(name)

        # Compute Complexity
        calculator = Complexity(model, (3, 224, 224))
        calculator.profile('G')