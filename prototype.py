import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
from .utils import my_con_loss


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def memory_querying_responding(query, key, value, mask=None, dropout=None, topk=32):
    d_k = query.size(-1)

    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, torch.finfo(scores.dtype).min)
    selected_scores, idx = scores.topk(topk)
    # query [8, 8, 49/98, 64]) value [8, 8, 2048, 64] score [8, 8, 49/98, 2048]
    # idx 8x8x98x32
    dummy_value = value.unsqueeze(2).expand(idx.size(0), idx.size(1), idx.size(2), value.size(-2), value.size(-1)) #[8, 8, 98, 2048, 64]
    dummy_idx = idx.unsqueeze(-1).expand(idx.size(0), idx.size(1), idx.size(2), idx.size(3), value.size(-1)) # [8, 8, 98, 32, 64]

    selected_value = torch.gather(dummy_value, 3, dummy_idx) # [8, 8, 98, 32, 64]

    p_attn = F.softmax(selected_scores.float(), dim=-1)  # [8, 8, 98, 32]

    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn.unsqueeze(3), selected_value).squeeze(3), p_attn


def _prepare_feature(self, fc_feats, att_feats, att_masks, labels = None):
        att_feats, seq, att_masks, seq_mask, query_matrix, cmn_masks, _ = \
            self._prepare_feature_forward(att_feats, att_masks, labels=labels)
        memory = self.model.encode(att_feats, att_masks)

        return fc_feats[..., :1], att_feats[..., :1], memory, att_masks, labels, query_matrix, cmn_masks


def _prepare_feature_forward(self, att_feats, att_masks=None, seq=None, labels=None,):
        """
        query_matrix 和 cmn_masks 的生成与管理涉及了跨模态原型矩阵的动态使用。
        该方法根据输入标签构建查询矩阵 query_matrix，并生成与之对应的掩码 cmn_masks，
        这些矩阵和掩码为跨模态交互提供了必要的基础设施。
        """
        att_feats, att_masks = self.clip_att(att_feats, att_masks)
        att_feats = pack_wrapper(self.att_embed, att_feats, att_masks)

        if att_masks is None:
            att_masks = att_feats.new_ones(att_feats.shape[:2], dtype=torch.long)

        per_num_protype = labels.sum(-1) * self.num_protype
        max_num_protype = max(per_num_protype)
        protypes = self.dim_reduction(self.protypes).view(self.num_cluster, self.num_protype, -1)
        query_matrix = protypes.new_zeros(att_feats.size(0), max_num_protype.int(), protypes.shape[-1])
        cmn_masks = protypes.new_zeros(query_matrix.shape[0], att_feats.size(1), max_num_protype.int())

        labels_mask = labels == 1
        for i in range(att_feats.size(0)):
            query_matrix[i, :per_num_protype[i].long()] = protypes[labels_mask[i]].view(-1, protypes.shape[-1])
            cmn_masks[i, :, :per_num_protype[i].long()] = 1

        responses = self.cmn(att_feats, query_matrix, query_matrix, cmn_masks)

        # feature interaction
        att_feats = self.fuse_feature(torch.cat((att_feats, responses), dim=2))


        att_masks = att_masks.unsqueeze(-2)
        if seq is not None:
            seq = seq[:, :-1]
            seq_mask = (seq.data > 0)
            seq_mask[:, 0] += True

            seq_mask = seq_mask.unsqueeze(-2)
            seq_mask = seq_mask & subsequent_mask(seq.size(-1)).to(seq_mask)
        else:
            seq_mask = None

        return att_feats, seq, att_masks, seq_mask, query_matrix, cmn_masks[:,0,:], responses


class MultiThreadMemory(nn.Module):
    def __init__(self, h, d_model, dropout=0.1, topk=32):
        super(MultiThreadMemory, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self.topk = topk

    def forward(self, query, key, value, mask=None, layer_past=None):
        """
        这个方法处理跨模态信息的查询和响应，它使用了一个多头注意力机制来处理query, key, 和 value。
        它通过调用 memory_querying_responding 函数实现了跨模态原型的选择和相应的交互，
        这符合跨模态原型矩阵（Shared Cross-modal Prototype Matrix）的概念，
        允许模型在文本和视觉特征间进行交互和信息融合。
        """
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        if layer_past is not None and layer_past.shape[2] == key.shape[1] > 1:
            query = self.linears[0](query)
            key, value = layer_past[0], layer_past[1]
            present = torch.stack([key, value])
        else:
            query, key, value = \
                [l(x) for l, x in zip(self.linears, (query, key, value))]

        if layer_past is not None and not (layer_past.shape[2] == key.shape[1] > 1):
            past_key, past_value = layer_past[0], layer_past[1]
            key = torch.cat((past_key, key), dim=1)
            value = torch.cat((past_value, value), dim=1)
            present = torch.stack([key, value])
        
        query, key, value = \
            [x.view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for x in [query, key, value]]

        x, self.attn = memory_querying_responding(query, key, value, mask=mask, dropout=self.dropout, topk=self.topk)

        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        if layer_past is not None:
            return self.linears[-1](x), present
        else:
            return self.linears[-1](x)