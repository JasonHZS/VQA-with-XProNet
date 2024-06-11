import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def memory_querying_responding(query, key, value, mask=None, dropout=None, topk=32):
    # 计算 query 向量的维度
    d_k = query.size(-1) 
    # 根据 scaled dot-product attention 计算 query 和 key 的相似度得分
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    # 如果提供了 mask，使用 mask 更新得分
    # 在 mask 中为 0 的位置上设置得分为极小值，以在 softmax 后这些位置的权重接近 0
    if mask is not None:
       scores = scores.masked_fill(mask == 0, torch.finfo(scores.dtype).min)
    # 从 scores 中选出 topk 最高的得分和对应的索引
    selected_scores, idx = scores.topk(topk)
    # 扩展 value 张量，使其在第三维度（query维度）上重复，以便对每个查询选择相应的 value
    dummy_value = value.unsqueeze(2).expand(idx.size(0), idx.size(1), idx.size(2), value.size(-2), value.size(-1))
    # 扩展索引，使其在最后一个维度（embedding维度）上重复，以便从 dummy_value 中选取特定元素
    dummy_idx = idx.unsqueeze(-1).expand(idx.size(0), idx.size(1), idx.size(2), idx.size(3), value.size(-1))
    # 使用扩展后的索引从扩展后的 value 张量中选取元素，这些元素是由 top-k 得分确定的
    selected_value = torch.gather(dummy_value, 3, dummy_idx)
    # 对选择的得分应用 softmax，计算最终的注意力权重
    p_attn = F.softmax(selected_scores.float(), dim=-1)
    # 如果提供了 dropout 模块，则在注意力权重上应用 dropout
    if dropout is not None:
       p_attn = dropout(p_attn)
    # 使用注意力权重对选取的 value 进行加权求和，计算最终的输出
    return torch.matmul(p_attn.unsqueeze(3), selected_value).squeeze(3), p_attn


class MultiThreadMemory(nn.Module):
    def __init__(self, h, d_model, dropout=0.1, topk=32):
        super(MultiThreadMemory, self).__init__()
        assert d_model % h == 0 # 输入和输出张量的维度d_model必须能被头数head整除
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

        print(f"query: {query.shape}, key: {key.shape}, value: {value.shape}")
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
        

def _prepare_feature(self, fc_feats, att_feats, att_masks, labels = None):
       att_feats, seq, att_masks, seq_mask, query_matrix, cmn_masks, _ = _prepare_feature_forward(att_feats, att_masks, labels=labels)
       memory = self.model.encode(att_feats, att_masks)

       return fc_feats[..., :1], att_feats[..., :1], memory, att_masks, labels, query_matrix, cmn_masks


def _prepare_feature_forward(att_feats, att_masks=None, seq=None,):
        """
        query_matrix 和 cmn_masks 的生成与管理涉及了跨模态原型矩阵的动态使用。
        该方法根据输入标签构建查询矩阵 query_matrix，并生成与之对应的掩码 cmn_masks，
        这些矩阵和掩码为跨模态交互提供了必要的基础设施。
        
        参数:
        att_feats (Tensor): 输入特征张量。(全部拼接后的向量)
        att_masks (Tensor, 可选): 对应于输入特征的掩码。
        seq (Tensor, 可选): 输入序列，通常用于序列处理任务。

        返回:
        Tuple: 包括处理后的特征、序列、特征掩码、序列掩码、查询矩阵、共同掩码和响应。
        """
        
        # protypes = nn.Parameter(torch.FloatTensor(num_protype * num_cluster, d_txt_ebd + d_img_ebd))
        init_protypes = torch.load(init_protypes_path).float()
        protypes = nn.Parameter(init_protypes)
        
        dim_reduction = nn.Linear(d_txt_ebd + d_img_ebd, cmm_dim)
        cmn = MultiThreadMemory(num_heads=16, d_model=2048, topk=5)
        fuse_feature = nn.Linear(d_model*2, d_model)
        
        protypes = dim_reduction(protypes).view(num_cluster, num_protype, -1)
        
        # 初始化一个默认查询矩阵和掩码
        # 保证函数即使在没有标签输入的情况下也能运行
        query_matrix = att_feats.new_zeros(att_feats.size(0), att_feats.size(1), att_feats.size(2))

        # 使用查询矩阵和掩码进行跨模态交互，并获取响应
        responses = cmn(att_feats, query_matrix, query_matrix)

        # 特征融合
        att_feats = fuse_feature(torch.cat((att_feats, responses), dim=2))

        return att_feats, query_matrix, responses


