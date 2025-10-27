# models/topo_grat.py

import torch
import torch.nn as nn
from .blocks import CTG_EPA_Block

class TopologyEnhancedSkipConnection(nn.Module):
    """
    Implements the Topology-Enhanced Skip Connection.
    This module enriches encoder features with causal information from the CGF
    before they are passed to the decoder.
    """
    def __init__(self, in_channels):
        super().__init__()
        self.conv_cgf = nn.Conv3d(2, in_channels, kernel_size=1, bias=False)
        self.delta = nn.Parameter(torch.randn(1)) # Learnable parameter delta

    def forward(self, x_encoder, cgf):
        """
        Args:
            x_encoder (torch.Tensor): Feature map from the encoder stage.
            cgf (torch.Tensor): Causal Growth Field predicted at the same stage.
        """
        # Re-weighting based on CGF magnitude
        # We use a conv to match channels, then apply sigmoid
        reweight_map = torch.sigmoid(self.conv_cgf(cgf))
        x_reweighted = x_encoder * (1 + reweight_map)

        # Boundary enhancement using CGF gradient
        # Compute gradient of CGF magnitude
        cgf_magnitude = torch.sqrt(cgf[:, 0:1]**2 + cgf[:, 1:2]**2)
        grad_x = F.conv3d(cgf_magnitude, self.sobel_x(), padding='same')
        grad_y = F.conv3d(cgf_magnitude, self.sobel_y(), padding='same')
        grad_mag = torch.sqrt(grad_x**2 + grad_y**2)
        
        boundary_term = self.delta * (x_encoder * grad_mag)
        
        return x_reweighted + boundary_term

    def sobel_x(self):
        # Simple Sobel operator for gradient calculation
        kernel = torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]], dtype=torch.float32)
        return kernel.repeat(1, 1, 1, 1, 1).to(self.delta.device)

    def sobel_y(self):
        kernel = torch.tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]], dtype=torch.float32)
        return kernel.repeat(1, 1, 1, 1, 1).to(self.delta.device)


class TopoGraT(nn.Module):
    """
    The main Topo-GraT architecture for Stage 2 segmentation.
    This is a hierarchical encoder-decoder network built with CTG-EPA blocks.
    """
    def __init__(self, in_channels=3, out_channels=2, base_c=32, num_blocks=[2, 2, 2, 2]):
        """
        Args:
            in_channels (int): Number of input channels (e.g., 3 for RGB).
            out_channels (int): Number of output classes for segmentation.
            base_c (int): The number of channels at the first stage.
            num_blocks (list): A list of integers specifying the number of CTG-EPA blocks per stage.
        """
        super().__init__()
        
        # --- Encoder Pathway ---
        self.patch_embed = nn.Conv3d(in_channels, base_c, kernel_size=7, stride=2, padding=3)
        
        self.encoder1 = nn.Sequential(*[CTG_EPA_Block(base_c) for _ in range(num_blocks[0])])
        self.down1 = nn.Conv3d(base_c, base_c*2, kernel_size=2, stride=2)
        
        self.encoder2 = nn.Sequential(*[CTG_EPA_Block(base_c*2) for _ in range(num_blocks[1])])
        self.down2 = nn.Conv3d(base_c*2, base_c*4, kernel_size=2, stride=2)
        
        self.encoder3 = nn.Sequential(*[CTG_EPA_Block(base_c*4) for _ in range(num_blocks[2])])
        self.down3 = nn.Conv3d(base_c*4, base_c*8, kernel_size=2, stride=2)
        
        self.bottleneck = nn.Sequential(*[CTG_EPA_Block(base_c*8) for _ in range(num_blocks[3])])

        # --- Decoder Pathway ---
        self.up3 = nn.ConvTranspose3d(base_c*8, base_c*4, kernel_size=2, stride=2)
        self.skip3 = TopologyEnhancedSkipConnection(base_c*4)
        self.decoder3 = nn.Sequential(*[CTG_EPA_Block(base_c*4) for _ in range(num_blocks[2])])
        
        self.up2 = nn.ConvTranspose3d(base_c*4, base_c*2, kernel_size=2, stride=2)
        self.skip2 = TopologyEnhancedSkipConnection(base_c*2)
        self.decoder2 = nn.Sequential(*[CTG_EPA_Block(base_c*2) for _ in range(num_blocks[1])])
        
        self.up1 = nn.ConvTranspose3d(base_c*2, base_c, kernel_size=2, stride=2)
        self.skip1 = TopologyEnhancedSkipConnection(base_c)
        self.decoder1 = nn.Sequential(*[CTG_EPA_Block(base_c) for _ in range(num_blocks[0])])
        
        self.final_up = nn.ConvTranspose3d(base_c, base_c, kernel_size=2, stride=2)
        self.final_conv = nn.Conv3d(base_c, out_channels, kernel_size=1)

    def forward(self, x):
        """
        Forward pass of the full Topo-GraT network.
        """
        # --- Encoder ---
        x1_embed = self.patch_embed(x)
        x1, cgf1 = self.encoder1(x1_embed)
        
        x2_down = self.down1(x1)
        x2, cgf2 = self.encoder2(x2_down)
        
        x3_down = self.down2(x2)
        x3, cgf3 = self.encoder3(x3_down)
        
        x4_down = self.down3(x3)
        x4, cgf4 = self.bottleneck(x4_down)

        # --- Decoder ---
        y3_up = self.up3(x4)
        x3_skip = self.skip3(x3, cgf3)
        y3 = self.decoder3(y3_up + x3_skip)[0] # [0] to get features, ignore CGF
        
        y2_up = self.up2(y3)
        x2_skip = self.skip2(x2, cgf2)
        y2 = self.decoder2(y2_up + x2_skip)[0]
        
        y1_up = self.up1(y2)
        x1_skip = self.skip1(x1, cgf1)
        y1 = self.decoder1(y1_up + x1_skip)[0]
        
        final_features = self.final_up(y1)
        seg_logits = self.final_conv(final_features)
        
        # We return the CGF from the bottleneck, as it has the most global context.
        # This is the CGF that will be used for the topology loss.
        return seg_logits, cgf4