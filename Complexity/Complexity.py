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
    def __init__(self, model: Module, dummy: tuple[int, int, int], device: torch.device = None) -> None:

        # Device: CPU or GPU
        self.device = device or torch.device('cpu')
    
        # Model and Dummy Input
        self.model = model.to(self.device)
        self.dummy = dummy

        # Activations
        self.actvs = 0

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

            # FLOPs
            flops = None

            # Convolution 2D & Transpose Convolution 2D
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                # Input Channels
                channel_in = feature_in.shape[1]
                # Kernel Size
                kernel = module.kernel_size
                # FLOPs
                flops = 2 * kernel[0] * kernel[1] * channel_in * feature_out.numel()

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
                    raise ValueError('Invalid Input or Output Shape')
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
                    raise ValueError('Invalid Mode for Upsample')
            
            # Nearest Interpolation
            elif isinstance(module, nn.UpsamplingNearest2d):
                # FLOPs
                flops = 0

            # Bilinear Interpolation
            elif isinstance(module, nn.UpsamplingBilinear2d):
                # FLOPs
                flops = 4 * feature_out.numel()

            # Pixel Shuffle
            elif isinstance(module, nn.PixelShuffle):
                # FLOPs
                if module.upscale_factor == 2:
                    flops = 0
                elif module.upscale_factor == 3:
                    flops = 9 * feature_in.shape[1] * feature_out.shape[1]

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
            self.actvs += feature_out.numel()
            self.flops += flops or 0

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
    def profile(self, order: str | Literal['G', 'M', 'k'] = 'M', num_input: int = 1, batch_size: int = 16) -> None:

        """
        ----------------------------------------------------------------------------------------------------------------
        FLOPs and FLOPS
        ----------------------------------------------------------------------------------------------------------------
        """
        # Dummy Input
        dummy = torch.rand((1, *self.dummy)).to(self.device)
        dummy = [dummy for _ in range(num_input)]

        # Start Time
        time_start = time.time()

        # Forward Pass
        for _ in range(10):
            self.actvs = 0
            self.flops = 0
            self.model(*dummy)

        # End Time
        time_end = time.time()

        # Activations
        actvs = self.actvs

        # FLOPs and FLOPs per Second
        flops = self.flops
        flops_per_sec = self.flops / ((time_end - time_start) / 10)

        # Memory Cost
        memory = int((actvs * 2 * 4 * batch_size) / (1024 ** 2))

        # Magnitude
        actvs = round(actvs * 1e-6, 3)
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
            raise ValueError('Invalid Order')

        """
        ----------------------------------------------------------------------------------------------------------------
        Parameter
        ----------------------------------------------------------------------------------------------------------------
        """
        # Total Parameter & Trainable Parameter
        num_param = sum(param.numel() for param in self.model.parameters())
        num_param_train = sum(param.numel() for param in self.model.parameters() if param.requires_grad)

        """
        ----------------------------------------------------------------------------------------------------------------
        Output Log
        ----------------------------------------------------------------------------------------------------------------
        """
        # Output Format
        title = "{:^21}|{:^21}|{:^21}|{:^21}|{:^21}|{:^21}"
        space = "{:^21}|{:^21}|{:^21}|{:^21,}|{:^21,}|{:^21,}"

        # Title
        print('-' * 130)
        print(title.format(order + ' FLOPs', order + ' FLOPS', 'M Acts', 'Memory (MB)', 'Params', 'Trainable Params'))
        print('-' * 130)

        # Output Log
        print(space.format(flops, flops_per_sec, actvs, memory, num_param, num_param_train))
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
                    'vgg11':       models.vgg11(),
                    'vgg13':       models.vgg13(),
                    'vgg16':       models.vgg16(),
                    'vgg19':       models.vgg19(),
                    'resnet18':    models.resnet18(),
                    'resnet34':    models.resnet34(),
                    'resnet50':    models.resnet50(),
                    'resnet101':   models.resnet101(),
                    'resnet152':   models.resnet152(),
                    'densenet121': models.densenet121(),
                    'densenet169': models.densenet169(),
                    'densenet201': models.densenet201(),
                    'densenet161': models.densenet161(),
                }

    for name, model in all_model.items():

        print()
        print(name)

        # Compute Complexity
        calculator = Complexity(model, (3, 224, 224), torch.device('cuda'))
        calculator.profile('G', num_input = 1, batch_size = 1)