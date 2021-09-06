from torch import nn, Tensor
import torch.nn.functional as F
import torch
from typing import Optional
import copy
import numpy as np


class PositionalEncoding(nn.Module):

    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''

        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()


class InteractionTransformer(nn.Module):

    def __init__(self, d_model=256, nhead=8, num_encoder_layers=1,
                 num_decoder_layers=6, num_rel_decoder_layers=6,
                 dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        # encoder of the backbone to refine feature sequence
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation)

        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, d_model)

        # entity branch
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation)
        decoder_norm = nn.LayerNorm(d_model)

        # interaction branch
        rel_decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                    dropout, activation)
        rel_decoder_norm = nn.LayerNorm(d_model)

        # branch aggregation: entity-aware attention
        interaction_layer = InteractionLayer(d_model, d_model, dropout)

        # finally Decoder for both entity and relation
        self.decoder = InteractionTransformerDecoder(
            decoder_layer,
            rel_decoder_layer,
            num_decoder_layers,
            interaction_layer,
            decoder_norm,
            rel_decoder_norm,
            return_intermediate_dec)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, rel_query_embed):
        bs, l, h = src.shape
        # generate feature sequence Batch Size x length x hidden feature dim -> bs, l, h -> l, bs, h
        src = src.permute(1, 0, 2)
        # refine the feature sequence using encoder
        memory = self.encoder(src, src_key_padding_mask=mask)
        # object query set
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        # interaction query set
        rel_query_embed = rel_query_embed.unsqueeze(1).repeat(1, bs, 1)
        # initialize the input of entity branch
        tgt = torch.zeros_like(query_embed)
        # initialize the input of interaction branch
        rel_tgt = torch.zeros_like(rel_query_embed)
        # memory shape: (W*H, bs, d_model)
        hs, rel_hs = self.decoder(tgt, rel_tgt, memory, memory_key_padding_mask=mask,
                                  pos=pos_embed, query_pos=query_embed, rel_query_pos=rel_query_embed)
        # TODO 这里return的形式可能是错的
        return hs, rel_hs, memory


class Transformer(nn.Module):

    def __init__(self, d_model=768, nhead=8, num_encoder_layers=1,
                 num_decoder_layers=6,
                 dim_feedforward=6144,
                 dropout=0.1,
                 activation="relu"):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead

        # encoder
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation)

        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, d_model)

        # entity branch
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed):
        # flatten NxCxHxW to HWxNxC
        # bs, c, h, w = src.shape

        # bs, l, h to l, bs, h
        bs, l, h = src.shape
        src = src.permute(1, 0, 2)
        # memory shape: (W*H, bs, d_model)
        memory = self.encoder(src, src_key_padding_mask=mask)
        # 直接使用bert 不再使用额外的encoder
        # memory = src

        # object query set
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        # initialize the input of entity branch
        tgt = torch.zeros_like(query_embed)
        # insatance decoder
        hs = self.decoder(tgt, memory, memory_key_padding_mask=mask, query_pos=query_embed)
        # hs = hs.flatten(start_dim=0, end_dim=1)

        return hs


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, d_model, n_position=200):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.position_enc = PositionalEncoding(d_model, n_position=n_position)

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                ):
        output = self.position_enc(src)

        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask)

        return output


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=6144, dropout=0.1,
                 activation="relu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def forward(self,
                src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                ):
        src2 = self.self_attn(src, src, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        # 位置编码
        self.position_enc = PositionalEncoding(d_model, n_position=512)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=memory,
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output


class InteractionLayer(nn.Module):
    def __init__(self, d_model, d_feature, dropout=0.1):
        super().__init__()
        self.d_feature = d_feature

        self.det_tfm = nn.Linear(d_model, d_feature)
        self.rel_tfm = nn.Linear(d_model, d_feature)
        self.det_value_tfm = nn.Linear(d_model, d_feature)

        self.rel_norm = nn.LayerNorm(d_model)

        if dropout is not None:
            self.dropout = dropout
            self.det_dropout = nn.Dropout(dropout)
            self.rel_add_dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

    def forward(self, det_in, rel_in):
        det_attn_in = self.det_tfm(det_in)
        rel_attn_in = self.rel_tfm(rel_in)
        det_value = self.det_value_tfm(det_in)
        scores = torch.matmul(det_attn_in.transpose(0, 1),
                              rel_attn_in.permute(1, 2, 0)) / math.sqrt(self.d_feature)
        det_weight = F.softmax(scores.transpose(1, 2), dim=-1)
        if self.dropout is not None:
            det_weight = self.det_dropout(det_weight)
        rel_add = torch.matmul(det_weight, det_value.transpose(0, 1))
        rel_out = self.rel_add_dropout(rel_add) + rel_in.transpose(0, 1)
        rel_out = self.rel_norm(rel_out)

        return det_in, rel_out.transpose(0, 1)


class InteractionTransformerDecoder(nn.Module):

    def __init__(self,
                 decoder_layer,
                 rel_decoder_layer,
                 num_layers,
                 interaction_layer=None,
                 norm=None,
                 rel_norm=None,
                 return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.rel_layers = _get_clones(rel_decoder_layer, num_layers)
        self.num_layers = num_layers
        if interaction_layer is not None:
            self.rel_interaction_layers = _get_clones(interaction_layer, num_layers)
        else:
            self.rel_interaction_layers = None
        self.norm = norm
        self.rel_norm = rel_norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, rel_tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                rel_query_pos: Optional[Tensor] = None):
        output = tgt
        rel_output = rel_tgt

        intermediate = []
        rel_intermediate = []

        for i in range(self.num_layers):
            # entity decoder layer
            output = self.layers[i](output, memory, tgt_mask=tgt_mask,
                                    memory_mask=memory_mask,
                                    tgt_key_padding_mask=tgt_key_padding_mask,
                                    memory_key_padding_mask=memory_key_padding_mask,
                                    query_pos=query_pos)
            # interaction decoder layer
            rel_output = self.rel_layers[i](rel_output, memory, tgt_mask=tgt_mask,
                                            memory_mask=memory_mask,
                                            tgt_key_padding_mask=tgt_key_padding_mask,
                                            memory_key_padding_mask=memory_key_padding_mask,
                                            query_pos=rel_query_pos)
            # entity-aware attention module
            if self.rel_interaction_layers is not None:
                output, rel_output = self.rel_interaction_layers[i](
                    output, rel_output
                )
            # for aux loss
            if self.return_intermediate:
                intermediate.append(self.norm(output))
                rel_intermediate.append(self.rel_norm(rel_output))
        # 为了防止self.return_intermediate=False 而  self.norm=None 时没有norm或多norm的bug
        if self.norm is not None:
            output = self.norm(output)
            rel_output = self.rel_norm(rel_output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)
                rel_intermediate.pop()
                rel_intermediate.append(rel_output)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(rel_intermediate)

        return output, rel_output


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
