import unittest
import torch
from model import ImageFeatureExtractor

class TestImageFeatureExtractor(unittest.TestCase):
    def test_feature_shape(self):
        # 实例化特征提取器
        feature_extractor = ImageFeatureExtractor()

        # 创建一个模拟的图像批次（比如 2 个 3x224x224 的图像）
        mock_images = torch.randn(2, 3, 224, 224)

        # 提取特征
        features = feature_extractor(mock_images)

        # 检查特征形状是否正确（对于 resnet50，特征形状应为 (2, 2048)）
        self.assertEqual(features.shape, (2, 2048))

if __name__ == '__main__':
    unittest.main()
