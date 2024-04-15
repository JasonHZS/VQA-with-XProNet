import torch
import torch.nn as nn
from torchvision import models

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
    def __init__(self, args, tokenizer, mode = 'train'):
        super(XProNet, self).__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.visual_extractor = ImageFeatureExtractor(args)
        self.encoder_decoder = QuesEmbedding(args, tokenizer, mode = mode)
        #if args.dataset_name == 'iu_xray':
        #    self.forward = self.forward_iu_xray
        #else:
        #    self.forward = self.forward_mimic_cxr

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)

    def forward(self, images, targets=None, labels=None, mode='train', update_opts={}):
        if self.args.dataset_name=='iu_xray':
            att_feats_0, fc_feats_0 = self.visual_extractor(images[:, 0])
            att_feats_1, fc_feats_1 = self.visual_extractor(images[:, 1])

            fc_feats = torch.cat((fc_feats_0, fc_feats_1), dim=1)
            att_feats = torch.cat((att_feats_0, att_feats_1), dim=1)
            if mode == 'train':
                output = self.encoder_decoder(fc_feats, att_feats, targets, labels = labels, mode='forward')
                return output
            elif mode == 'sample':
                output, output_probs = self.encoder_decoder(fc_feats, att_feats, labels = labels, mode='sample', update_opts=update_opts)
                return output, output_probs
            else:
                raise ValueError

        else:
            att_feats, fc_feats = self.visual_extractor(images)
            if mode == 'train':
                output = self.encoder_decoder(fc_feats, att_feats, targets, labels=labels, mode='forward')
                return output
            elif mode == 'sample':
                output, output_probs = self.encoder_decoder(fc_feats, att_feats, labels=labels, mode='sample', update_opts=update_opts)
                return output, output_probs
            else:
                raise ValueError