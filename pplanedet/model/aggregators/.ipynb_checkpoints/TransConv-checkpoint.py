import math
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from ..builder import AGGREGATORS
from ..common_model import ConvModule

class PositionEmbeddingSine(nn.Layer):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self,
                 num_pos_feats=64,
                 temperature=10000,
                 normalize=False,
                 scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, mask):
        assert mask is not None
        not_mask = ~mask
        not_mask = not_mask.astype('float')
        y_embed = not_mask.cumsum(1)
        x_embed = not_mask.cumsum(2)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = paddle.arange(self.num_pos_feats)
        dim_t = self.temperature**(2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = paddle.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()),
            axis=4).flatten(3)
        pos_y = paddle.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()),
            axis=4).flatten(3)
        pos = paddle.concat((pos_y, pos_x), axis=3).transpose((0, 3, 1, 2))
        return pos

def build_position_encoding(hidden_dim, shape):
    mask = paddle.zeros(shape).astype('bool')
    pos_module = PositionEmbeddingSine(hidden_dim // 2)
    pos_embs = pos_module(mask)
    return pos_embs


class AttentionLayer(nn.Layer):
    """ Position attention module"""

    def __init__(self, in_dim, out_dim, ratio=4, stride=1):
        super(AttentionLayer, self).__init__()
        self.chanel_in = in_dim
        norm = True
        act = 'relu'
        self.pre_conv = ConvModule(
            in_dim,
            out_dim,
            kernel_size=3,
            stride=stride,
            norm=norm,
            act=act,)
        self.query_conv = nn.Conv2D(
            in_channels=out_dim, out_channels=out_dim // ratio, kernel_size=1)
        self.key_conv = nn.Conv2D(
            in_channels=out_dim, out_channels=out_dim // ratio, kernel_size=1)
        self.value_conv = nn.Conv2D(
            in_channels=out_dim, out_channels=out_dim, kernel_size=1)
        self.final_conv = ConvModule(
            out_dim,
            out_dim,
            kernel_size=3,
            norm=norm,
            act=act)
        self.softmax = nn.Softmax(axis=-1)
        self.gamma = self.create_parameter((1,))

    def forward(self, x, pos=None):
        """
            inputs :
                x : inpput feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        x = self.pre_conv(x)
        m_batchsize, _, height, width = x.shape
        if pos is not None:
            x += pos
        proj_query = self.query_conv(x).reshape((m_batchsize, -1,
                                             width * height)).transpose((0, 2, 1))
        proj_key = self.key_conv(x).reshape((m_batchsize, -1, width * height))

        energy = paddle.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        attention = attention.transpose((0, 2, 1))
        proj_value = self.value_conv(x).reshape((m_batchsize, -1, width * height))
        out = paddle.bmm(proj_value, attention)
        out = out.reshape((m_batchsize, -1, height, width))
        proj_value = proj_value.reshape((m_batchsize, -1, height, width))
        out_feat = self.gamma * out + x
        out_feat = self.final_conv(out_feat)
        return out_feat

@AGGREGATORS.register()
class TransConvEncoderModule(nn.Layer):
    def __init__(self, in_dim, attn_in_dims, attn_out_dims, strides, ratios, downscale=True, pos_shape=None, cfg=None):
        super(TransConvEncoderModule, self).__init__()
        if downscale:
            stride = 2
        else:
            stride = 1
        # self.first_conv = ConvModule(in_dim, 2*in_dim, kernel_size=3, stride=stride, padding=1)
        # self.final_conv = ConvModule(attn_out_dims[-1], attn_out_dims[-1], kernel_size=3, stride=1, padding=1)
        attn_layers = []
        for dim1, dim2, stride, ratio in zip(attn_in_dims, attn_out_dims, strides, ratios):
            attn_layers.append(AttentionLayer(dim1, dim2, ratio, stride))
        if pos_shape is not None:
            self.attn_layers = nn.LayerList(attn_layers)
        else:
            self.attn_layers = nn.Sequential(*attn_layers)
        self.pos_shape = pos_shape
        self.pos_embeds = []
        if pos_shape is not None:
            for dim in attn_out_dims:
                pos_embed = build_position_encoding(dim, pos_shape).cuda()
                self.pos_embeds.append(pos_embed)
    
    def forward(self, src):
        # src = self.first_conv(src)
        if self.pos_shape is None:
            src = self.attn_layers(src)
        else:
            for layer, pos in zip(self.attn_layers, self.pos_embeds):
                src = layer(src, pos)
        # src = self.final_conv(src)
        return src