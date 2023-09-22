import math
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn.initializer import Assign
from .norm import zeros_,xavier_uniform_

class Identity(nn.Layer):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


def drop_path(x, drop_prob=0.0, training=False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = paddle.to_tensor(1 - drop_prob)
    shape = (paddle.shape(x)[0], ) + (1, ) * (x.ndim - 1)
    random_tensor = keep_prob + paddle.rand(shape, dtype=x.dtype)
    random_tensor = paddle.floor(random_tensor)  # binarize
    output = x.divide(keep_prob) * random_tensor
    return output


class DropPath(nn.Layer):

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

def to_2tuple(x):
    if isinstance(x, (list, tuple)):
        return x
    return tuple([x] * 2)

def add_parameter(layer: nn.Layer, datas, name=None):
    parameter = layer.create_parameter(
        shape=(datas.shape), default_initializer=Assign(datas))
    if name:
        layer.add_parameter(name, parameter)
    return parameter

class PositionEmbedding(nn.Layer):
    def __init__(self,
                 num_pos_feats=128,
                 temperature=10000,
                 normalize=True,
                 scale=2 * math.pi,
                 embed_type='sine',
                 num_embeddings=50,
                 offset=0.,
                 eps=1e-6):
        super(PositionEmbedding, self).__init__()
        assert embed_type in ['sine', 'learned']

        self.embed_type = embed_type
        self.offset = offset
        self.eps = eps
        if self.embed_type == 'sine':
            self.num_pos_feats = num_pos_feats
            self.temperature = temperature
            self.normalize = normalize
            self.scale = scale
        elif self.embed_type == 'learned':
            self.row_embed = nn.Embedding(num_embeddings, num_pos_feats)
            self.col_embed = nn.Embedding(num_embeddings, num_pos_feats)
        else:
            raise ValueError(f"{self.embed_type} is not supported.")

    def forward(self, mask):
        """
        Args:
            mask (Tensor): [B, H, W]
        Returns:
            pos (Tensor): [B, H, W, C]
        """
        if self.embed_type == 'sine':
            y_embed = mask.cumsum(1)
            x_embed = mask.cumsum(2)
            if self.normalize:
                y_embed = (y_embed + self.offset) / (
                    y_embed[:, -1:, :] + self.eps) * self.scale
                x_embed = (x_embed + self.offset) / (
                    x_embed[:, :, -1:] + self.eps) * self.scale

            dim_t = 2 * (paddle.arange(self.num_pos_feats) //
                         2).astype('float32')
            dim_t = self.temperature**(dim_t / self.num_pos_feats)

            pos_x = x_embed.unsqueeze(-1) / dim_t
            pos_y = y_embed.unsqueeze(-1) / dim_t
            pos_x = paddle.stack(
                (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()),
                axis=4).flatten(3)
            pos_y = paddle.stack(
                (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()),
                axis=4).flatten(3)
            return paddle.concat((pos_y, pos_x), axis=3)
        elif self.embed_type == 'learned':
            h, w = mask.shape[-2:]
            i = paddle.arange(w)
            j = paddle.arange(h)
            x_emb = self.col_embed(i)
            y_emb = self.row_embed(j)
            return paddle.concat(
                [
                    x_emb.unsqueeze(0).tile([h, 1, 1]),
                    y_emb.unsqueeze(1).tile([1, w, 1]),
                ],
                axis=-1).unsqueeze(0)
        else:
            raise ValueError(f"not supported {self.embed_type}")

class MultiHeadAttention(nn.Layer):
    """
    Attention mapps queries and a set of key-value pairs to outputs, and
    Multi-Head Attention performs multiple parallel attention to jointly attending
    to information from different representation subspaces.

    Please refer to `Attention Is All You Need <https://arxiv.org/pdf/1706.03762.pdf>`_
    for more details.

    Parameters:
        embed_dim (int): The expected feature size in the input and output.
        num_heads (int): The number of heads in multi-head attention.
        dropout (float, optional): The dropout probability used on attention
            weights to drop some attention targets. 0 for no dropout. Default 0
        kdim (int, optional): The feature size in key. If None, assumed equal to
            `embed_dim`. Default None.
        vdim (int, optional): The feature size in value. If None, assumed equal to
            `embed_dim`. Default None.
        need_weights (bool, optional): Indicate whether to return the attention
            weights. Default False.

    Examples:

        .. code-block:: python

            import paddle

            # encoder input: [batch_size, sequence_length, d_model]
            query = paddle.rand((2, 4, 128))
            # self attention mask: [batch_size, num_heads, query_len, query_len]
            attn_mask = paddle.rand((2, 2, 4, 4))
            multi_head_attn = paddle.nn.MultiHeadAttention(128, 2)
            output = multi_head_attn(query, None, None, attn_mask=attn_mask)  # [2, 4, 128]
    """

    def __init__(self,
                 embed_dim,
                 num_heads,
                 dropout=0.,
                 kdim=None,
                 vdim=None,
                 need_weights=False):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.need_weights = need_weights

        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        if self._qkv_same_embed_dim:
            self.in_proj_weight = self.create_parameter(
                shape=[embed_dim, 3 * embed_dim],
                attr=None,
                dtype=self._dtype,
                is_bias=False)
            self.in_proj_bias = self.create_parameter(
                shape=[3 * embed_dim],
                attr=None,
                dtype=self._dtype,
                is_bias=True)
        else:
            self.q_proj = nn.Linear(embed_dim, embed_dim)
            self.k_proj = nn.Linear(self.kdim, embed_dim)
            self.v_proj = nn.Linear(self.vdim, embed_dim)

        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self._type_list = ('q_proj', 'k_proj', 'v_proj')

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)
            else:
                zeros_(p)

    def compute_qkv(self, tensor, index):
        if self._qkv_same_embed_dim:
            tensor = F.linear(
                x=tensor,
                weight=self.in_proj_weight[:, index * self.embed_dim:(index + 1)
                                           * self.embed_dim],
                bias=self.in_proj_bias[index * self.embed_dim:(index + 1) *
                                       self.embed_dim]
                if self.in_proj_bias is not None else None)
        else:
            tensor = getattr(self, self._type_list[index])(tensor)
        tensor = tensor.reshape(
            [0, 0, self.num_heads, self.head_dim]).transpose([0, 2, 1, 3])
        return tensor

    def forward(self, query, key=None, value=None, attn_mask=None):
        r"""
        Applies multi-head attention to map queries and a set of key-value pairs
        to outputs.

        Parameters:
            query (Tensor): The queries for multi-head attention. It is a
                tensor with shape `[batch_size, query_length, embed_dim]`. The
                data type should be float32 or float64.
            key (Tensor, optional): The keys for multi-head attention. It is
                a tensor with shape `[batch_size, key_length, kdim]`. The
                data type should be float32 or float64. If None, use `query` as
                `key`. Default None.
            value (Tensor, optional): The values for multi-head attention. It
                is a tensor with shape `[batch_size, value_length, vdim]`.
                The data type should be float32 or float64. If None, use `query` as
                `value`. Default None.
            attn_mask (Tensor, optional): A tensor used in multi-head attention
                to prevents attention to some unwanted positions, usually the
                paddings or the subsequent positions. It is a tensor with shape
                broadcasted to `[batch_size, n_head, sequence_length, sequence_length]`.
                When the data type is bool, the unwanted positions have `False`
                values and the others have `True` values. When the data type is
                int, the unwanted positions have 0 values and the others have 1
                values. When the data type is float, the unwanted positions have
                `-INF` values and the others have 0 values. It can be None when
                nothing wanted or needed to be prevented attention to. Default None.

        Returns:
            Tensor|tuple: It is a tensor that has the same shape and data type \
                as `query`, representing attention output. Or a tuple if \
                `need_weights` is True or `cache` is not None. If `need_weights` \
                is True, except for attention output, the tuple also includes \
                the attention weights tensor shaped `[batch_size, num_heads, query_length, key_length]`. \
                If `cache` is not None, the tuple then includes the new cache \
                having the same type as `cache`, and if it is `StaticCache`, it \
                is same as the input `cache`, if it is `Cache`, the new cache \
                reserves tensors concatanating raw tensors with intermediate \
                results of current query.
        """
        key = query if key is None else key
        value = query if value is None else value
        # compute q ,k ,v
        q, k, v = (self.compute_qkv(t, i)
                   for i, t in enumerate([query, key, value]))

        # scale dot product attention
        product = paddle.matmul(x=q, y=k, transpose_y=True)
        scaling = float(self.head_dim)**-0.5
        product = product * scaling

        if attn_mask is not None:
            # Support bool or int mask
            attn_mask = _convert_attention_mask(attn_mask, product.dtype)
            product = product + attn_mask
        weights = F.softmax(product)
        if self.dropout:
            weights = F.dropout(
                weights,
                self.dropout,
                training=self.training,
                mode="upscale_in_train")

        out = paddle.matmul(weights, v)

        # combine heads
        out = paddle.transpose(out, perm=[0, 2, 1, 3])
        out = paddle.reshape(x=out, shape=[0, 0, out.shape[2] * out.shape[3]])

        # project to output
        out = self.out_proj(out)

        outs = [out]
        if self.need_weights:
            outs.append(weights)
        return out if len(outs) == 1 else tuple(outs)
    
def _convert_attention_mask(attn_mask, dtype):
    """
    Convert the attention mask to the target dtype we expect.
    Parameters:
        attn_mask (Tensor, optional): A tensor used in multi-head attention
                to prevents attention to some unwanted positions, usually the
                paddings or the subsequent positions. It is a tensor with shape
                broadcasted to `[batch_size, n_head, sequence_length, sequence_length]`.
                When the data type is bool, the unwanted positions have `False` 
                values and the others have `True` values. When the data type is 
                int, the unwanted positions have 0 values and the others have 1 
                values. When the data type is float, the unwanted positions have 
                `-INF` values and the others have 0 values. It can be None when 
                nothing wanted or needed to be prevented attention to. Default None.
        dtype (VarType): The target type of `attn_mask` we expect.
    Returns:
        Tensor: A Tensor with shape same as input `attn_mask`, with data type `dtype`.
    """
    return nn.layer.transformer._convert_attention_mask(attn_mask, dtype)

def inverse_sigmoid(x, eps=1e-6):
    x = x.clip(min=0., max=1.)
    return paddle.log(x / (1 - x + eps) + eps)