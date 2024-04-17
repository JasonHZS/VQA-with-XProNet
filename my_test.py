import unittest
import torch
import torch.nn as nn
from torchvision import transforms, models
from my_vqadata import VQADataset
from my_model import ImageFeatureExtractor, QuesEmbedding, VisualExtractor
from torch.utils.data import DataLoader
from modules.utils import parse_agrs
from modules.encoder_decoder import EncoderDecoder
from modules.tokenizers import Tokenizer
from unittest.mock import MagicMock


class TestImageFeatureExtractor(unittest.TestCase):
    def test_feature_shape(self):
        # 实例化特征提取器
        feature_extractor = ImageFeatureExtractor()

        # 创建一个模拟的图像批次（比如 2 个 3x224x224 的图像）
        # mock_images = torch.randn(2, 3, 224, 224)

        # 创建数据集实例
        dataset = VQADataset()
        # 使用 DataLoader 来加载数据
        data_loader = DataLoader(dataset, batch_size=1, shuffle=True)
        mock_images = next(iter(data_loader))['image']

        # 提取特征
        features = feature_extractor(mock_images)

        # 检查特征形状是否正确（对于 resnet50，特征形状应为 (2, 2048)）
        self.assertEqual(features.shape, (1, 2048))

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
        expected_shape = (self.batch_size, self.output_size)
        
        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(output.shape, expected_shape)


if __name__ == '__main__':
    unittest.main()
