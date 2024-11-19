"""
========================================================================================================================
Package
========================================================================================================================
"""
import os

import matplotlib.pyplot as plt

import torch 
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module 


"""
========================================================================================================================
Grad-CAM: CT Specific
========================================================================================================================
"""
class Grad_CAM():

    """
    ====================================================================================================================
    Initialization
    ====================================================================================================================
    """
    def __init__(self, model: Module, layer: Module, thres_fore: float = None, thres_back: float = None) -> None:
    
        # Model and Target Layer
        self.model = model
        self.layer = layer

        # Activation and Gradient
        self.actis = None
        self.grads = None

        # Threhold for Foreground and Background
        self.thres_fore = thres_fore or -1000
        self.thres_back = thres_back or -250

        # Capture Information from Target Layer
        self.hook_layer()

        return

    """
    ====================================================================================================================
    Capture Information from Target Layer
    ====================================================================================================================
    """
    def hook_layer(self) -> None:

        """
        ----------------------------------------------------------------------------------------------------------------
        Capture Activation from Forward Pass
        ----------------------------------------------------------------------------------------------------------------
        """
        def forward_hook(module: Module, feature_in: Tensor, feature_out: Tensor) -> None:

            self.actis = feature_out.detach()

            return 
        """
        ----------------------------------------------------------------------------------------------------------------
        Capture Gradient from Backward Pass
        ----------------------------------------------------------------------------------------------------------------
        """
        def backward_hook(module: Module, grad_in: Tensor, grad_out: tuple[Tensor]) -> None:

            self.grads = grad_out[0].detach()

            return

        """
        ----------------------------------------------------------------------------------------------------------------
        Register Hooks
        ----------------------------------------------------------------------------------------------------------------
        """
        self.layer.register_forward_hook(forward_hook)
        self.layer.register_full_backward_hook(backward_hook)

        return

    """
    ====================================================================================================================
    Grad-CAM Heatmap
    ====================================================================================================================
    """
    def grad_cam(self, real1_g: Tensor, save: bool, save_path: str, save_nums: str) -> Tensor:

        # Forward Pass
        fake2_g = self.model(real1_g)

        # Break if no Need to Save
        if not save:
            return fake2_g

        """
        ----------------------------------------------------------------------------------------------------------------
        Foreground Grad-CAM
        ----------------------------------------------------------------------------------------------------------------
        """
        # Define Threshold
        thres_fore = (self.thres_fore + 1000) / 4000 * 2 - 1
        # Comput Loss
        loss_fore = fake2_g[fake2_g > thres_fore].mean()
        # Backward Pass
        loss_fore.backward(retain_graph = True)
        # Define Importance Over Channel
        weight = self.grads.mean(dim = (2, 3), keepdim = True)
        visual_fore = (weight * self.actis).sum(dim = 1, keepdim = True)

        """
        ----------------------------------------------------------------------------------------------------------------
        Background Grad-CAM
        ----------------------------------------------------------------------------------------------------------------
        """
        # Define Threshold
        thres_back = (self.thres_back + 1000) / 4000 * 2 - 1
        # Comput Loss
        loss_back = fake2_g[fake2_g < thres_back].mean()
        # Backward Pass
        loss_back.backward(retain_graph = True)
        # Define Importance Over Channel
        weight = self.grads.mean(dim = (2, 3), keepdim = True)
        visual_back = (weight * self.actis).sum(dim = 1, keepdim = True)

        """
        ----------------------------------------------------------------------------------------------------------------
        Post-Processing
        ----------------------------------------------------------------------------------------------------------------
        """
        # Apply ReLU to Retain Positive Influence
        visual_fore = F.relu(visual_fore)
        visual_back = F.relu(visual_back)

        # Normalize to [0, 1]
        visual_fore -= visual_fore.min()
        visual_fore /= visual_fore.max()
        visual_back -= visual_back.min()
        visual_back /= visual_back.max()

        # Interpolate to Input Resolution
        visual_fore = F.interpolate(visual_fore, size = (real1_g.shape[2], real1_g.shape[3]), mode = 'bilinear')
        visual_back = F.interpolate(visual_back, size = (real1_g.shape[2], real1_g.shape[3]), mode = 'bilinear')

        # Convert Format
        visual_fore = visual_fore.detach().cpu().numpy()[0, 0]
        visual_back = visual_back.detach().cpu().numpy()[0, 0]
        real1_a = real1_g.detach().cpu().numpy()[0, 3]
        fake2_a = fake2_g.detach().cpu().numpy()[0, 0]

        """
        ----------------------------------------------------------------------------------------------------------------
        Plot
        ----------------------------------------------------------------------------------------------------------------
        """
        # Create the plot
        _, axs = plt.subplots(2, 2, figsize = (7.5, 7.5))

        # Display Input
        ax = axs[0][0]
        ax.imshow(real1_a, cmap = 'gray')

        ax.set_title('Model Input')
        ax.set_xticks([])
        ax.set_yticks([])

        # Display Output
        ax = axs[0][1]
        ax.imshow(fake2_a, cmap = 'gray')

        ax.set_title('Model Output')
        ax.set_xticks([])
        ax.set_yticks([])

        # Overlay Foregraound Grad-CAM Heatmap on Model Input
        ax = axs[1][0]
        plot_fore1 = ax.imshow(visual_fore, cmap = 'jet')
        ax.imshow(real1_a, cmap = 'gray', alpha = 0.5)
        plt.colorbar(plot_fore1, ax = ax, cax = ax.inset_axes((1, 0, 0.05, 1.0)))

        ax.set_title('Foreground')
        ax.set_xticks([])
        ax.set_yticks([])

        # Overlay Backgraound Grad-CAM Heatmap on Model Input
        ax = axs[1][1]
        plot_back1 = ax.imshow(visual_back, cmap = 'jet')
        ax.imshow(real1_a, cmap = 'gray', alpha = 0.5)
        plt.colorbar(plot_back1, ax = ax, cax = ax.inset_axes((1, 0, 0.05, 1.0)))

        ax.set_title('Background')
        ax.set_xticks([])
        ax.set_yticks([])

        # Save Figure
        plt.savefig(os.path.join(save_path, save_nums + '.png'), format = 'png', dpi = 300)

        return fake2_g


"""
========================================================================================================================
Main Function
========================================================================================================================
"""
if __name__ == '__main__':

    pass