import numpy as np
import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

class ImageFeatureExtractor(nn.Module):
    """
    暂时不用
    """
    def __init__(self, pretrained=True):
        super(ImageFeatureExtractor, self).__init__()
        # 加载预训练的 ResNet
        resnet = models.resnet50(pretrained=pretrained)
        
        # 移除最后的全连接层
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        
        # 特征维度
        # 对于 resnet50，它是 2048
        self.out_features = resnet.fc.in_features

    def forward(self, x):
        # 通过 ResNet 提取特征
        x = self.features(x)
        
        # 将特征图展平
        x = x.view(x.size(0), -1)
        return x
    

class VisualExtractor(nn.Module):
    """
    input:
    输入的 images 通过一个 ResNet50 模型（没有最后两层）来提取特征。
    通常，ResNet50 接收形状为 (batch_size, 3, 224, 224) 的输入，
    并输出形状为 (batch_size, 2048, 7, 7) 的特征图（这是在去掉完全连接层和最后一个池化层后的结果）。

    output:
    patch_feats: (batch_size, 49, 2048)
        这里 49 是 7x7 网格展平后的结果。
        这种重塑操作意味着每一个 49 个元素代表一个 "patch"（或区块），
        每个 patch 是一个 2048 维的向量，表示从原始图像的特定区域中提取的特征。
    avg_feats: (batch_size, 2048)

    TODO：提取特征后可以先保存到文件中
    """
    def __init__(self, pretrained=True):
        super(VisualExtractor, self).__init__()
        # 加载预训练的 ResNet
        model = models.resnet50(pretrained=pretrained)
        modules = list(model.children())[:-2]
        self.model = nn.Sequential(*modules)
        self.avg_fnt = torch.nn.AvgPool2d(kernel_size=7, stride=1, padding=0)

    def forward(self, images):
        patch_feats = self.model(images)
        avg_feats = self.avg_fnt(patch_feats).squeeze().reshape(-1, patch_feats.size(1))
        batch_size, feat_size, _, _ = patch_feats.shape
        patch_feats = patch_feats.reshape(batch_size, feat_size, -1).permute(0, 2, 1)
        return patch_feats, avg_feats


class QuesEmbedding(nn.Module):
    """
    input:
        output_size:指单层输出的维度，但是最后 return 的维度是 output_size*2
    输入为问题的词嵌入（不是原始文本），输出为提取的特征。
    """
    def __init__(self, input_size=500, output_size=1024, num_layers=1, batch_first=True):
        super(QuesEmbedding, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=output_size, batch_first=True)

    def forward(self, ques):
        _, hx = self.lstm(ques)
        # h, _ = hx
        # ques_embedding = h[0]
        lstm_embedding = torch.cat([hx[0], hx[1]], dim=2) # 拼接后的维度为output_size*2
        ques_embedding = lstm_embedding[0]
        return ques_embedding
    
class MutanFusion(nn.Module):
    def __init__(self, input_dim, out_dim, num_layers):
        super(MutanFusion, self).__init__()
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.num_layers = num_layers

        hv = []
        for i in range(self.num_layers):
            do = nn.Dropout(p=0.5)
            lin = nn.Linear(input_dim, out_dim)
            hv.append(nn.Sequential(do, lin, nn.Tanh()))
        #
        self.image_transformation_layers = nn.ModuleList(hv)
        #
        hq = []
        for i in range(self.num_layers):
            do = nn.Dropout(p=0.5)
            lin = nn.Linear(input_dim, out_dim)
            hq.append(nn.Sequential(do, lin, nn.Tanh()))
        #
        self.ques_transformation_layers = nn.ModuleList(hq)

    def forward(self, ques_emb, img_emb):
        batch_size = img_emb.size()[0]
        x_mm = []
        for i in range(self.num_layers):
            x_hv = img_emb
            x_hv = self.image_transformation_layers[i](x_hv)

            x_hq = ques_emb
            x_hq = self.ques_transformation_layers[i](x_hq)

            x_mm.append(torch.mul(x_hq, x_hv))
        
        x_mm = torch.stack(x_mm, dim=1)
        x_mm = x_mm.sum(1).view(batch_size, self.out_dim)
        x_mm = F.tanh(x_mm)
        return x_mm


class Attention(nn.Module):
    def __init__(self, d=1024, k=512, dropout=True):
        super(Attention, self).__init__()
        self.ff_image = nn.Linear(d, k)
        self.ff_ques = nn.Linear(d, k)
        if dropout:
            self.dropout = nn.Dropout(p=0.5)
        self.ff_attention = nn.Linear(k, 1)

    def forward(self, vi, vq):
        # N * 196 * 1024 -> N * 196 * 512
        hi = self.ff_image(vi)
        # N * 1024 -> N * 512 -> N * 1 * 512
        hq = self.ff_ques(vq).unsqueeze(dim=1)
        # N * 196 * 512
        ha = F.tanh(hi + hq)
        if getattr(self, 'dropout'):
            ha = self.dropout(ha)
        # N * 196 * 512 -> N * 196 * 1 -> N * 196
        ha = self.ff_attention(ha).squeeze(dim=2)
        pi = F.softmax(ha)
        # (N * 196 * 1, N * 196 * 1024) -> N * 1024
        vi_attended = (pi.unsqueeze(dim=2) * vi).sum(dim=1)
        u = vi_attended + vq
        return u


class SANModel(nn.Module):
    def __init__(self, vocab_size, word_emb_size=500, emb_size=1024, att_ff_size=512, output_size=1000,
                 num_att_layers=1, num_mlp_layers=1, mode='train', extract_img_features=True, features_dir=None):
        super(SANModel, self).__init__()
        self.mode = mode
        self.features_dir = features_dir
        self.image_channel = VisualExtractor(output_size=emb_size, mode=mode, extract_img_features=extract_img_features,
                                            features_dir=features_dir)

        self.word_emb_size = word_emb_size
        self.word_embeddings = nn.Embedding(vocab_size, word_emb_size)
        self.ques_channel = QuesEmbedding(
            word_emb_size, output_size=emb_size, num_layers=1, batch_first=False)

        self.san = nn.ModuleList(
            [Attention(d=emb_size, k=att_ff_size)] * num_att_layers)

        self.mlp = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(emb_size, output_size))

    def forward(self, images, questions, image_ids):
        image_embeddings = self.image_channel(images, image_ids)
        embeds = self.word_embeddings(questions)
        # nbatch = embeds.size()[0]
        # nwords = embeds.size()[1]

        # ques_embeddings = self.ques_channel(embeds.view(nwords, nbatch, self.word_emb_size))
        ques_embeddings = self.ques_channel(embeds)
        vi = image_embeddings
        u = ques_embeddings
        for att_layer in self.san:
            u = att_layer(vi, u)
        output = self.mlp(u)
        return output


class VQAModel(nn.Module):
    """
    vocab_size与output_size保持一致，因为VQAModel最后输出的维度取决于词汇表的大小，
    所以模型最后输入的是最大概率的那个维度的索引
    """

    def __init__(self, vocab_size=1000, word_emb_size=300, emb_size=2048, output_size=1000, ques_channel_type='lstm', 
                use_mutan=True, extract_img_features=True, features_dir=None):
        super(VQAModel, self).__init__()
        # self.mode = mode # 'train' or 'eval'
        self.word_emb_size = word_emb_size
        self.visual_extractor = VisualExtractor(pretrained=True)

        # NOTE the padding_idx below.
        self.word_embeddings = nn.Embedding(vocab_size, word_emb_size)

        if ques_channel_type.lower() == 'lstm':
            self.ques_embedding = QuesEmbedding(
                input_size=word_emb_size, output_size=int(emb_size/2), num_layers=1, batch_first=False)
        elif ques_channel_type.lower() == 'deeplstm':
            self.ques_embedding = QuesEmbedding(
                input_size=word_emb_size, output_size=emb_size, num_layers=2, batch_first=False)
        else:
            msg = 'ques channel type not specified. please choose one of -  lstm or deeplstm'
            print(msg)
            raise Exception(msg)
        
        if use_mutan:
            self.mutan = MutanFusion(emb_size, emb_size, 5)
            self.mlp = nn.Sequential(nn.Linear(emb_size, output_size))
        else:
            self.mlp = nn.Sequential(
                nn.Linear(emb_size, 1000),
                nn.Dropout(p=0.5),
                nn.Tanh(),
                nn.Linear(1000, output_size))

    def forward(self, images, questions):
        patch_feats, avg_feats = self.visual_extractor(images)
        # print('视觉特征：', avg_feats.shape)
        embeds = self.word_embeddings(questions)
        ques_embeddings = self.ques_embedding(embeds)
        # print('文本特征：', ques_embeddings.shape)
        combined = self.mutan(ques_embeddings, avg_feats)
        output = self.mlp(combined)
        return output