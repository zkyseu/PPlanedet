import paddle.nn as nn
from paddle.vision.ops import DeformConv2D
from .utils import to_2tuple
from .norm import normal_,zeros_

class PatchEmbed(nn.Layer):
    """ Image to Patch Embedding
    """
    def __init__(self, patch_size=7, stride=4, in_chans=3, embed_dim=768,):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.proj = nn.Conv2D(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.BatchNorm2D(embed_dim)
        self.initialize_layer(self.proj)
    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return x
    @staticmethod
    def initialize_layer(layer):
        if isinstance(layer, (nn.Conv2D, nn.Linear)):
            normal_(layer.weight, mean=0., std=0.001)
            if layer.bias is not None:
                zeros_(layer.bias)



class ALAU(nn.Layer):
    """Adaptive Lane Aware Unit Module.
    Adaptive Lane Aware Unit Module is implemented based on DCN v1.
    It uses anchor shape prediction rather than feature map to
    predict offsets of deform conv layer.
    Args:
        in_channels (int): Number of channels in the input feature map.
        out_channels (int): Number of channels in the output feature map.
        kernel_size (tuple): Deformable conv kernel size.
        deform_groups (int): Deformable conv group size.
        cfg (dict or list[dict], optional): config dict.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=(3,3),
                 deform_groups=4,
                 cfg = None):
        super(ALAU, self).__init__()
        offset_channels = kernel_size[0] * kernel_size[1] * 2
        hidden_layer = (deform_groups * offset_channels) // 2
        self.conv_offset = nn.Sequential(
            PatchEmbed(patch_size=3,stride=2,in_chans=2,embed_dim=hidden_layer),
            PatchEmbed(patch_size=3,stride=2,in_chans=hidden_layer,embed_dim=deform_groups * offset_channels),
        )
        self.conv_adaption = DeformConv2D(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=[(kernel_size[0] - 1) // 2,(kernel_size[1] - 1) // 2],
            deformable_groups=deform_groups)
        self.relu = nn.ReLU()

    def forward(self, x, shape):
        offset = self.conv_offset(shape.detach())
        x = self.relu(self.conv_adaption(x, offset))
        return x,offset