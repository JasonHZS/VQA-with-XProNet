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
        

class TestPrepareFeature(unittest.TestCase):

    def setUp(self):
        # 初始化模型和参数
        args = parse_agrs()
        self.num_features = 2048
        self.seq_length = 10
        self.tokenizer = Tokenizer(args)
        self.model = EncoderDecoder(args, self.tokenizer, mode = None)  # 假设args包含了所需的所有配置

        # 模拟图像特征
        self.fc_feats = torch.randn(args.batch_size, self.num_features, 7*7)
        # 模拟文本特征
        self.att_feats = torch.randn(args.batch_size, self.seq_length, self.num_features)
        # 模拟注意力掩码
        self.att_masks = torch.ones(args.batch_size, self.seq_length)
        self.labels = torch.randint(0, 1, (args.batch_size, self.seq_length))

    def test_prepare_feature(self):
        # 调用 _prepare_feature 方法
        memory, _, _, _, _, _ = self.model._prepare_feature(self.fc_feats, self.att_feats, self.att_masks, self.labels)

        # 检查编码器输出的特征和注意力掩码
        # 这里我们假设编码器输出的特征形状为 (batch_size, seq_length, hidden_size)
        # 并且注意力掩码的形状与输入的文本特征形状一致
        self.assertEqual(memory.shape, (self.batch_size, self.seq_length, self.model.d_model))
        self.assertEqual(self.att_masks.shape, (self.batch_size, self.seq_length))


if __name__ == '__main__':
    unittest.main()
