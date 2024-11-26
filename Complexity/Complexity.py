"""
========================================================================================================================
Package
========================================================================================================================
"""
import time

from typing import Literal

import torch
from torch import nn
from torch import Tensor
from torch.nn import Module


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
        self.flop = 0

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
                # Input and Output Channels
                channel_in = feature_in.shape[1]
                # Output Channels, Height and Width
                channel_out, height_out, width_out = feature_out.shape[1:]
                # Kernel Size
                kernel = module.kernel_size[0]
                # FLOPs
                flop = 2 * kernel * kernel * channel_in * channel_out * height_out * width_out

            # Fully Connected
            elif isinstance(module, nn.Linear):
                # FLOPs
                flop = 2 * feature_in.shape[1] * feature_out.shape[1]

            # Batch Normalization 1D
            elif isinstance(module, nn.BatchNorm1d):
                # Output Channels
                channel_out = feature_out.shape[1]
                # FLOPs
                flop = 4 * channel_out

            # Batch Normalization 2D
            elif isinstance(module, nn.BatchNorm2d):
                # Output Channels, Height and Width
                channel_out, height_out, width_out = feature_out.shape[1:]
                # FLOPs
                flop = 4 * channel_out * height_out * width_out

            # Instance Normalization 1D
            elif isinstance(module, nn.InstanceNorm1d):
                # Output Channels and Length
                channel_out, length_out = feature_out.shape[1:]
                # FLOPs
                flop = 7 * channel_out * length_out
            
            # Instance Normalization 2D
            elif isinstance(module, nn.InstanceNorm2d):
                # Output Channels, Height and Width
                channel_out, height_out, width_out = feature_out.shape[1:]
                # FLOPs
                flop = 7 * channel_out * height_out * width_out

            # # Layer Normalization
            # elif isinstance(module, LayerNorm):
            #     # Output Channels, Height and Width
            #     channel_out, height_out, width_out = feature_out.shape[1:]
            #     # FLOPs
            #     flop = 7 * channel_out * height_out * width_out

            # Dropout 1D
            elif isinstance(module, nn.Dropout):
                # Output Channels
                channel_out = feature_out.shape[1]
                # FLOPs
                flop = 2 * channel_out

            # Dropout 2D
            elif isinstance(module, nn.Dropout2d):
                # Output Channels, Height and Width
                channel_out, height_out, width_out = feature_out.shape[1:]
                # FLOPs
                flop = 2 * channel_out * height_out * width_out

            # Activation
            elif isinstance(module, (nn.ReLU, nn.LeakyReLU, nn.GELU, nn.Tanh, nn.Sigmoid, nn.Softmax2d, nn.Softmax)):

                # Fully Connected
                if feature_out.dim() == 2:
                    # Output Channels
                    channel_out = feature_out.shape[1]
                    # FLOPs
                    flop = channel_out
                # Convolution
                elif feature_out.dim() == 4:
                    # Output Channels, Height and Width
                    channel_out, height_out, width_out = feature_out.shape[1:]
                    # FLOPs
                    flop = channel_out * height_out * width_out
                else:
                    raise TypeError('Invalid Number of Axis of Activation')

            # Pooling 
            elif isinstance(module, (nn.MaxPool2d, nn.AvgPool2d)):
                # Output Channels, Height and Width
                channel_out, height_out, width_out = feature_out.shape[1:]
                # Kernel Size
                kernel = module.kernel_size
                # FLOPs
                flop = kernel * kernel * channel_out * height_out * width_out

            # Upsample
            elif isinstance(module, nn.Upsample):

                # Nearest Interpolation
                if module.mode == 'nearest':
                    # FLOPs
                    flop = 0
                # Bilinear Interpolation
                elif module.mode == 'bilinear':
                    # Output Channels, Height and Width
                    channel_out, height_out, width_out = feature_out.shape[1:]
                    # FLOPs
                    flop = 4 * channel_out * height_out * width_out
                else:
                    raise TypeError('Invalid Mode for Upsample')
            
            # Nearest Interpolation
            elif isinstance(module, nn.UpsamplingNearest2d):
                # FLOPs
                flop = 0

            # Bilinear Interpolation
            elif isinstance(module, nn.UpsamplingBilinear2d):
                # Output Channels, Height and Width
                channel_out, height_out, width_out = feature_out.shape[1:]
                # FLOPs
                flop = 4 * channel_out * height_out * width_out

            # Flatten
            elif isinstance(module, nn.Flatten):
                # FLOPs
                flop = 0

            # Others
            else:
                # FLOPs
                flop = 0

            # Total FLOPs
            self.flop += flop

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
            
        break_block(self.model)
                
        return

    """
    ====================================================================================================================
    Profile
    ====================================================================================================================
    """
    def profile(self, order: str | Literal['G', 'M', 'k'] = 'M') -> None:

        # Output Format
        space = "{:<25}{:<20}"

        """
        ----------------------------------------------------------------------------------------------------------------
        FLOPs and FLOPS
        ----------------------------------------------------------------------------------------------------------------
        """
        # Dummy Input
        dummy = torch.rand((1, *self.dummy))

        # Start Time
        time_start = time.time()

        # Forward Pass
        self.model(dummy)

        # End Time
        time_end = time.time()

        # FLOPs per Second
        flops = self.flop / (time_end - time_start)
        
        # Output Log
        print()
        if order == 'G':
            print(space.format(round(self.flop * 1e-9, 3), 'GFLOPs'))
            print(space.format(round(flops * 1e-9, 3), 'GFLOPS'))
        elif order == 'M':
            print(space.format(round(self.flop * 1e-6, 3), 'MFLOPs'))
            print(space.format(round(flops * 1e-6, 3), 'MFLOPS'))
        elif order == 'k':
            print(space.format(round(self.flop * 1e-3, 3), 'kFLOPs'))
            print(space.format(round(flops * 1e-3, 3), 'kFLOPS'))
        else:
            raise TypeError('Invalid Order')
        print()

        """
        ----------------------------------------------------------------------------------------------------------------
        Parameter
        ----------------------------------------------------------------------------------------------------------------
        """
        # Total Parameter & Trainable Parameter
        num_params = sum(params.numel() for params in self.model.parameters())
        num_params_train = sum(params.numel() for params in self.model.parameters() if params.requires_grad)

        # Output Log
        print()
        print(space.format(num_params, 'Total Parameters'))
        print(space.format(num_params_train, 'Trainable Parameters'))
        print()

        return 


"""
========================================================================================================================
Main Function
========================================================================================================================
"""
if __name__ == '__main__':

    # # Model
    # model = Unet()

    # # Compute Complexity
    # calculator = Complexity(model, (7, 256, 256))
    # calculator.profile()

    pass