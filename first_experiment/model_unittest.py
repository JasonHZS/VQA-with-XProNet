import unittest
import torch
import torch.nn as nn
from torchvision import transforms, models
from my_vqadata import VQADataset
from my_model import ImageFeatureExtractor, QuesEmbedding, VisualExtractor, MutanFusion, VQAModel
from torch.utils.data import DataLoader
from modules.utils import parse_agrs
from modules.encoder_decoder import EncoderDecoder
from modules.tokenizers import Tokenizer
from unittest.mock import MagicMock


# class TestImageFeatureExtractor(unittest.TestCase):
#     def test_feature_shape(self):
#         # 实例化特征提取器
#         feature_extractor = ImageFeatureExtractor()

#         # 创建一个模拟的图像批次（比如 2 个 3x224x224 的图像）
#         # mock_images = torch.randn(2, 3, 224, 224)

#         # 创建数据集实例
#         dataset = VQADataset()
#         # 使用 DataLoader 来加载数据
#         data_loader = DataLoader(dataset, batch_size=1, shuffle=True)
#         mock_images = next(iter(data_loader))['image']

#         # 提取特征
#         features = feature_extractor(mock_images)

#         # 检查特征形状是否正确（对于 resnet50，特征形状应为 (2, 2048)）
#         self.assertEqual(features.shape, (1, 2048))


class TestVisualExtractor(unittest.TestCase):
    def setUp(self):
        self.model = VisualExtractor(pretrained=False)
        self.images = torch.randn(4, 3, 224, 224)  # 4 images, 3 color channels, 224x224 pixels

    def test_forward(self):
        # Testing forward pass
        patch_feats, avg_feats = self.model(self.images)
        self.assertEqual(patch_feats.shape, (4, 49, 2048))  # Expected shape based on ResNet output and processing
        self.assertEqual(avg_feats.shape, (4, 2048))       # Averaged features should be (batch_size, feat_size)

    def test_output_types(self):
        # Testing output types
        patch_feats, avg_feats = self.model(self.images)
        self.assertIsInstance(patch_feats, torch.Tensor)
        self.assertIsInstance(avg_feats, torch.Tensor)

    def test_no_error(self):
        # Testing that no error is raised
        try:
            patch_feats, avg_feats = self.model(self.images)
        except Exception as e:
            self.fail(f"Forward pass raised an exception {e}")


class TestVQADataset(unittest.TestCase):
    def setUp(self):
        # Setup code to run before each test
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        self.dataset = VQADataset(transform=self.transform)

    def test_initialization(self):
        # Test dataset initialization
        self.assertIsInstance(self.dataset.data, dict)
        self.assertEqual(len(self.dataset), len(self.dataset.data.keys()))

    def test_length(self):
        # Test the length of the dataset
        self.assertEqual(len(self.dataset), len(self.dataset.keys))

    def test_get_item(self):
        # Test retrieving an item
        sample = self.dataset[0]  # Get the first item
        self.assertIsInstance(sample, dict)
        self.assertIn('image', sample)
        self.assertIn('answer', sample)
        self.assertIn('question', sample)
        self.assertIsInstance(sample['image'], torch.Tensor)
        self.assertEqual(sample['image'].shape, torch.Size([3, 224, 224]))


class TestQuesEmbedding(unittest.TestCase):
    def setUp(self):
        # Initialize the QuesEmbedding instance
        self.input_size = 500
        self.output_size = 1024
        self.batch_size = 2
        self.seq_len = 10
        self.num_layers = 1
        self.ques_embedding = QuesEmbedding(input_size=self.input_size, output_size=self.output_size, num_layers=self.num_layers)

    def test_initialization(self):
        # Check LSTM initialization
        self.assertIsInstance(self.ques_embedding.lstm, nn.LSTM)
        self.assertEqual(self.ques_embedding.lstm.input_size, self.input_size)
        self.assertEqual(self.ques_embedding.lstm.hidden_size, self.output_size)

    def test_forward(self):
        # Test the forward pass
        test_input = torch.randn(self.batch_size, self.seq_len, self.input_size)
        output = self.ques_embedding(test_input)
        expected_shape = (self.batch_size, self.output_size*2)
        
        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(output.shape, expected_shape)


class TestMutanFusionIntegration(unittest.TestCase):
    def setUp(self):
        # 初始化视觉提取器
        self.visual_extractor = VisualExtractor(pretrained=True)
        # 初始化问题嵌入模块
        self.ques_embedding = QuesEmbedding(input_size=500, output_size=1024, num_layers=1, batch_first=True)
        # 初始化 MutanFusion 模块
        # 问题与视觉嵌入模块的输入维度为 2048（先适应resnet 的输入维度）
        self.mutan_fusion = MutanFusion(input_dim=2048, out_dim=2048, num_layers=3)

        # 创建一些虚拟的数据输入
        self.dummy_images = torch.rand(10, 3, 224, 224)  # 假设有10个图像，每个图像为224x224大小，3通道
        self.dummy_questions = torch.rand(10, 10, 500)  # 假设有10个问题，每个问题是长度为10的序列，每个序列元素500维

    def test_integration(self):
        # 通过视觉提取器处理图像
        patch_feats, avg_feats = self.visual_extractor(self.dummy_images)
        # print(avg_feats.shape)
        # 通过问题嵌入模块处理问题
        ques_embed = self.ques_embedding(self.dummy_questions)
        # print(ques_embed.shape)
        # 通过 MutanFusion 模块处理融合
        fusion_output = self.mutan_fusion(ques_embed, avg_feats)

        # 检查融合输出的形状是否正确
        self.assertEqual(fusion_output.shape, (10, 2048))

        # 进一步测试输出是否为非空
        self.assertFalse(torch.isnan(fusion_output).any())
        

class TestVQAModel(unittest.TestCase):
    def setUp(self):
        # 初始化模型
        self.model = VQAModel()
        self.vocab_size = 10000
        self.model.eval()  # 设置为评估模式，以禁用例如Dropout等训练特定的层

    def test_model_initialization(self):
        """ 测试模型是否成功初始化 """
        self.assertIsInstance(self.model, VQAModel)

    def test_forward_pass(self):
        """ 测试模型的前向传播 """
        # 创建一个简单的输入样本，这里假设输入的图像尺寸为224x224，问题长度为10
        images = torch.rand(10, 3, 224, 224)  # 假设有10个图像，每个图像为224x224大小，3通道 
        # questions = torch.rand(10, 10, 500)  # 假设有10个问题，每个问题是长度为10的序列，每个序列元素500维
        questions = torch.randint(0, self.vocab_size, (10, 10))
        # 执行前向传播
        output = self.model(images, questions)

        # 检查输出形状是否正确
        self.assertEqual(output.shape, (10, 1000))  # 假设最终输出维度是1000


if __name__ == '__main__':
    unittest.main()
