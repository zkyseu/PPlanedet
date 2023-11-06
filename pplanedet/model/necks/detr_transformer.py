import copy
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from ..common_model.Transformer_module import MultiHeadAttention,_convert_attention_mask,PositionEmbedding
from ..common_model.norm import xavier_uniform_,linear_init_,conv_init_,normal_
from ..builder import NECKS

def _get_clones(module, N):
    return nn.LayerList([copy.deepcopy(module) for _ in range(N)])


class TransformerEncoderLayer(nn.Layer):
    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation="relu",
                 attn_dropout=None,
                 act_dropout=None,
                 normalize_before=False):
        super(TransformerEncoderLayer, self).__init__()
        attn_dropout = dropout if attn_dropout is None else attn_dropout
        act_dropout = dropout if act_dropout is None else act_dropout
        self.normalize_before = normalize_before

        self.self_attn = MultiHeadAttention(d_model, nhead, attn_dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(act_dropout, mode="upscale_in_train")
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout, mode="upscale_in_train")
        self.dropout2 = nn.Dropout(dropout, mode="upscale_in_train")
        self.activation = getattr(F, activation)
        self._reset_parameters()

    def _reset_parameters(self):
        linear_init_(self.linear1)
        linear_init_(self.linear2)

    @staticmethod
    def with_pos_embed(tensor, pos_embed):
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(self, src, src_mask=None, pos_embed=None):
        residual = src
        if self.normalize_before:
            src = self.norm1(src)
        q = k = self.with_pos_embed(src, pos_embed)
        src = self.self_attn(q, k, value=src, attn_mask=src_mask)

        src = residual + self.dropout1(src)
        if not self.normalize_before:
            src = self.norm1(src)

        residual = src
        if self.normalize_before:
            src = self.norm2(src)
        src = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = residual + self.dropout2(src)
        if not self.normalize_before:
            src = self.norm2(src)
        return src

class TransformerEncoder(nn.Layer):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, src_mask=None, pos_embed=None):
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=src_mask, pos_embed=pos_embed)

        if self.norm is not None:
            output = self.norm(output)

        return output
    
class TransformerDecoderLayer(nn.Layer):
    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation="relu",
                 attn_dropout=None,
                 act_dropout=None,
                 normalize_before=False):
        super(TransformerDecoderLayer, self).__init__()
        attn_dropout = dropout if attn_dropout is None else attn_dropout
        act_dropout = dropout if act_dropout is None else act_dropout
        self.normalize_before = normalize_before

        self.self_attn = MultiHeadAttention(d_model, nhead, attn_dropout)
        self.cross_attn = MultiHeadAttention(d_model, nhead, attn_dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(act_dropout, mode="upscale_in_train")
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout, mode="upscale_in_train")
        self.dropout2 = nn.Dropout(dropout, mode="upscale_in_train")
        self.dropout3 = nn.Dropout(dropout, mode="upscale_in_train")
        self.activation = getattr(F, activation)
        self._reset_parameters()

    def _reset_parameters(self):
        linear_init_(self.linear1)
        linear_init_(self.linear2)

    @staticmethod
    def with_pos_embed(tensor, pos_embed):
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(self,
                tgt,
                memory,
                tgt_mask=None,
                memory_mask=None,
                pos_embed=None,
                query_pos_embed=None):
        tgt_mask = _convert_attention_mask(tgt_mask, tgt.dtype)

        residual = tgt
        if self.normalize_before:
            tgt = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt, query_pos_embed)
        tgt = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask)
        tgt = residual + self.dropout1(tgt)
        if not self.normalize_before:
            tgt = self.norm1(tgt)

        residual = tgt
        if self.normalize_before:
            tgt = self.norm2(tgt)
        q = self.with_pos_embed(tgt, query_pos_embed)
        k = self.with_pos_embed(memory, pos_embed)
        tgt = self.cross_attn(q, k, value=memory, attn_mask=memory_mask)
        tgt = residual + self.dropout2(tgt)
        if not self.normalize_before:
            tgt = self.norm2(tgt)

        residual = tgt
        if self.normalize_before:
            tgt = self.norm3(tgt)
        tgt = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = residual + self.dropout3(tgt)
        if not self.normalize_before:
            tgt = self.norm3(tgt)
        return tgt
    

class TransformerDecoder(nn.Layer):
    def __init__(self,
                 decoder_layer,
                 num_layers,
                 norm=None,
                 return_intermediate=False):
        super(TransformerDecoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self,
                tgt,
                memory,
                tgt_mask=None,
                memory_mask=None,
                pos_embed=None,
                query_pos_embed=None):
        tgt_mask = _convert_attention_mask(tgt_mask, tgt.dtype)

        output = tgt
        intermediate = []
        for layer in self.layers:
            output = layer(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                pos_embed=pos_embed,
                query_pos_embed=query_pos_embed)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)

        if self.return_intermediate:
            return paddle.stack(intermediate)

        return output.unsqueeze(0)
        
@NECKS.register()
class DETRTransformer(nn.Layer):
    __shared__ = ['hidden_dim']

    def __init__(self,
                 num_queries=192,
                 position_embed_type='sine',
                 return_intermediate_dec=True,
                 backbone_num_channels=2048,
                 hidden_dim=256,
                 nhead=8,
                 num_encoder_layers=6,
                 num_decoder_layers=6,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation="relu",
                 attn_dropout=None,
                 act_dropout=None,
                 normalize_before=False,
                 cfg=None):
        super(DETRTransformer, self).__init__()
        assert position_embed_type in ['sine', 'learned'],\
            f'ValueError: position_embed_type not supported {position_embed_type}!'
        self.hidden_dim = hidden_dim
        self.nhead = nhead

        encoder_layer = TransformerEncoderLayer(
            hidden_dim, nhead, dim_feedforward, dropout, activation,
            attn_dropout, act_dropout, normalize_before)
        encoder_norm = nn.LayerNorm(hidden_dim) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers,
                                          encoder_norm)

        decoder_layer = TransformerDecoderLayer(
            hidden_dim, nhead, dim_feedforward, dropout, activation,
            attn_dropout, act_dropout, normalize_before)
        decoder_norm = nn.LayerNorm(hidden_dim)
        self.decoder = TransformerDecoder(
            decoder_layer,
            num_decoder_layers,
            decoder_norm,
            return_intermediate=return_intermediate_dec)

        self.input_proj = nn.Conv2D(
            backbone_num_channels, hidden_dim, kernel_size=1)
        self.query_pos_embed = nn.Embedding(num_queries, hidden_dim)
        self.position_embedding = PositionEmbedding(
            hidden_dim // 2,
            normalize=True if position_embed_type == 'sine' else False,
            embed_type=position_embed_type)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)
        conv_init_(self.input_proj)
        normal_(self.query_pos_embed.weight)

    def _convert_attention_mask(self, mask):
        return (mask - 1.0) * 1e9

    def forward(self, src, src_mask=None, *args, **kwargs):
        r"""
        Applies a Transformer model on the inputs.

        Parameters:
            src (List(Tensor)): Backbone feature maps with shape [[bs, c, h, w]].
            src_mask (Tensor, optional): A tensor used in multi-head attention
                to prevents attention to some unwanted positions, usually the
                paddings or the subsequent positions. It is a tensor with shape
                [bs, H, W]`. When the data type is bool, the unwanted positions
                have `False` values and the others have `True` values. When the
                data type is int, the unwanted positions have 0 values and the
                others have 1 values. When the data type is float, the unwanted
                positions have `-INF` values and the others have 0 values. It
                can be None when nothing wanted or needed to be prevented
                attention to. Default None.

        Returns:
            output (Tensor): [num_levels, batch_size, num_queries, hidden_dim]
            memory (Tensor): [batch_size, hidden_dim, h, w]
        """
        # use last level feature map
        src_proj = self.input_proj(src[-1])
        bs, c, h, w = paddle.shape(src_proj)
        # flatten [B, C, H, W] to [B, HxW, C]
        src_flatten = src_proj.flatten(2).transpose([0, 2, 1])
        if src_mask is not None:
            src_mask = F.interpolate(src_mask.unsqueeze(0), size=(h, w))[0]
        else:
            src_mask = paddle.ones([bs, h, w])
        pos_embed = self.position_embedding(src_mask).flatten(1, 2)

        if self.training:
            src_mask = self._convert_attention_mask(src_mask)
            src_mask = src_mask.reshape([bs, 1, 1, h * w])
        else:
            src_mask = None

        memory = self.encoder(
            src_flatten, src_mask=src_mask, pos_embed=pos_embed)

        query_pos_embed = self.query_pos_embed.weight.unsqueeze(0).tile(
            [bs, 1, 1])
        tgt = paddle.zeros_like(query_pos_embed)
        output = self.decoder(
            tgt,
            memory,
            memory_mask=src_mask,
            pos_embed=pos_embed,
            query_pos_embed=query_pos_embed)

        if self.training:
            src_mask = src_mask.reshape([bs, 1, 1, h, w])
        else:
            src_mask = None

        return (output, memory.transpose([0, 2, 1]).reshape([bs, c, h, w]),
                src_proj, src_mask)


@NECKS.register()
class Lanedetr_transormer(DETRTransformer):
    def __init__(self,
                 num_priors=192,
                 position_embed_type='sine',
                 return_intermediate_dec=True,
                 backbone_num_channels=2048,
                 hidden_dim=256,
                 nhead=8,
                 num_encoder_layers=6,
                 num_decoder_layers=6,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation="relu",
                 attn_dropout=None,
                 act_dropout=None,
                 normalize_before=False,
                 cfg=None):
        super(DETRTransformer, self).__init__()
        assert position_embed_type in ['sine', 'learned'],\
            f'ValueError: position_embed_type not supported {position_embed_type}!'
        self.hidden_dim = hidden_dim
        self.nhead = nhead
        self.num_priors=num_priors

        encoder_layer = TransformerEncoderLayer(
            hidden_dim, nhead, dim_feedforward, dropout, activation,
            attn_dropout, act_dropout, normalize_before)
        encoder_norm = nn.LayerNorm(hidden_dim) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers,
                                          encoder_norm)

        decoder_layer = TransformerDecoderLayer(
            hidden_dim, nhead, dim_feedforward, dropout, activation,
            attn_dropout, act_dropout, normalize_before)
        decoder_norm = nn.LayerNorm(hidden_dim)
        self.decoder = TransformerDecoder(
            decoder_layer,
            num_decoder_layers,
            decoder_norm,
            return_intermediate=return_intermediate_dec)

        self.input_proj = nn.Conv2D(
            backbone_num_channels, hidden_dim, kernel_size=1)
        self.position_embedding = PositionEmbedding(
            hidden_dim // 2,
            normalize=True if position_embed_type == 'sine' else False,
            embed_type=position_embed_type)

        self._reset_parameters()
        self._init_prior_embeddings()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)
        conv_init_(self.input_proj)

    def _init_prior_embeddings(self):
        self.query_pos_embed = nn.Embedding(self.num_priors, 3)
        bottom_priors_nums = self.num_priors * 3 // 4
        left_priors_nums, _ = self.num_priors // 8, self.num_priors // 8
        strip_size = 0.5 / (left_priors_nums // 2 - 1)
        bottom_strip_size = 1 / (bottom_priors_nums // 4 + 1)

        with paddle.no_grad():
            for i in range(left_priors_nums):
                self.query_pos_embed.weight[i, 0] = i // 2 * strip_size
                self.query_pos_embed.weight[i, 1] = 0.0
                self.query_pos_embed.weight[i,
                                             2] = 0.16 if i % 2 == 0 else 0.32

            for i in range(left_priors_nums,
                           left_priors_nums + bottom_priors_nums):
                self.query_pos_embed.weight[i, 0] = 0.0
                self.query_pos_embed.weight[i, 1] = (
                    (i - left_priors_nums) // 4 + 1) * bottom_strip_size
                self.query_pos_embed.weight[i, 2] = 0.2 * (i % 4 + 1)

            for i in range(left_priors_nums + bottom_priors_nums,
                           self.num_priors):
                self.query_pos_embed.weight[i, 0] = (
                    i - left_priors_nums - bottom_priors_nums) // 2 * strip_size
                self.query_pos_embed.weight[i, 1] = 1.0
                self.query_pos_embed.weight[i,
                                             2] = 0.68 if i % 2 == 0 else 0.84