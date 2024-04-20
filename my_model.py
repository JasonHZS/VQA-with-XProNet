import numpy as np
import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

class ImageFeatureExtractor(nn.Module):
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
    通常，ResNet50 在使用默认的 ImageNet 预训练模型时，
    接收形状为 (batch_size, 3, 224, 224) 的输入并输出形状为 (batch_size, 2048, 7, 7) 的特征图（这是在去掉完全连接层和最后一个池化层后的结果）。
    output:
    patch_feats: (batch_size, 49, 2048)
    avg_feats: (batch_size, 2048)
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
    def __init__(self, input_size=500, output_size=1024, num_layers=1, batch_first=True):
        super(QuesEmbedding, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=output_size, batch_first=batch_first)

    def forward(self, ques):
        # seq_len * N * 500 -> (1 * N * 1024, 1 * N * 1024)
        _, hx = self.lstm(ques)
        # (1 * N * 1024, 1 * N * 1024) -> 1 * N * 1024
        h, _ = hx
        ques_embedding = h[0]
        return ques_embedding


class XProNet(nn.Module):
    """
    代替 vqa 任务中的 fusion 网络？？？
    """
    def __init__(self, args, tokenizer, mode = 'train'):
        super(XProNet, self).__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.visual_extractor = ImageFeatureExtractor(args)
        self.encoder_decoder = QuesEmbedding(args, tokenizer, mode = mode)

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)

    def forward(self, images, targets=None, labels=None, mode='train', update_opts={}):
        att_feats, fc_feats = self.visual_extractor(images)
        if mode == 'train':
            output = self.encoder_decoder(fc_feats, att_feats, targets, labels=labels, mode='forward')
            return output
        elif mode == 'sample':
            output, output_probs = self.encoder_decoder(fc_feats, att_feats, labels=labels, mode='sample', update_opts=update_opts)
            return output, output_probs
        else:
            raise ValueError
    

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
        # Pdb().set_trace()
        batch_size = img_emb.size()[0]
        x_mm = []
        for i in range(self.num_layers):
            x_hv = img_emb
            x_hv = self.image_transformation_layers[i](x_hv)

            x_hq = ques_emb
            x_hq = self.ques_transformation_layers[i](x_hq)
            x_mm.append(torch.mul(x_hq, x_hv))
        #
        x_mm = torch.stack(x_mm, dim=1)
        x_mm = x_mm.sum(1).view(batch_size, self.out_dim)
        x_mm = F.tanh(x_mm)
        return x_mm


# class VQAModel(nn.Module):

#     def __init__(self, vocab_size=10000, word_emb_size=300, emb_size=1024, output_size=1000, image_channel_type='I', ques_channel_type='lstm', use_mutan=True, mode='train', extract_img_features=True, features_dir=None):
#         super(VQAModel, self).__init__()
#         self.mode = mode
#         self.word_emb_size = word_emb_size
#         self.image_channel = ImageEmbedding(image_channel_type, output_size=emb_size, mode=mode,
#                                             extract_features=extract_img_features, features_dir=features_dir)

#         # NOTE the padding_idx below.
#         self.word_embeddings = nn.Embedding(vocab_size, word_emb_size)
#         if ques_channel_type.lower() == 'lstm':
#             self.ques_channel = QuesEmbedding(
#                 input_size=word_emb_size, output_size=emb_size, num_layers=1, batch_first=False)
#         elif ques_channel_type.lower() == 'deeplstm':
#             self.ques_channel = QuesEmbedding(
#                 input_size=word_emb_size, output_size=emb_size, num_layers=2, batch_first=False)
#         else:
#             msg = 'ques channel type not specified. please choose one of -  lstm or deeplstm'
#             print(msg)
#             raise Exception(msg)
#         if use_mutan:
#             self.mutan = MutanFusion(emb_size, emb_size, 5)
#             self.mlp = nn.Sequential(nn.Linear(emb_size, output_size))
#         else:
#             self.mlp = nn.Sequential(
#                 nn.Linear(emb_size, 1000),
#                 nn.Dropout(p=0.5),
#                 nn.Tanh(),
#                 nn.Linear(1000, output_size))

#     def forward(self, images, questions, image_ids):
#         image_embeddings = self.image_channel(images, image_ids)
#         embeds = self.word_embeddings(questions)
#         ques_embeddings = self.ques_channel(embeds)
#         if hasattr(self, 'mutan'):
#             combined = self.mutan(ques_embeddings, image_embeddings)
#         else:
#             combined = image_embeddings * ques_embeddings
#         output = self.mlp(combined)
#         return output