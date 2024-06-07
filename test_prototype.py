import unittest
import torch
import math
from torch.nn import functional as F
from prototype import memory_querying_responding

class TestMemoryQueryingResponding(unittest.TestCase):
    def test_basic_functionality(self):
        torch.manual_seed(42)  # for reproducibility
        query = torch.randn(8, 8, 98, 64)  # [batch, heads, query_dim, key_dim]
        key = torch.randn(8, 8, 2048, 64)  # [batch, heads, value_dim, key_dim]
        value = torch.randn(8, 8, 2048, 64)  # [batch, heads, value_dim, embedding_dim]

        # Running the function without mask or dropout
        result, attention = memory_querying_responding(query, key, value)
        self.assertEqual(result.size(), torch.Size([8, 8, 98, 64]))
        self.assertEqual(attention.size(), torch.Size([8, 8, 98, 32]))

    def test_with_mask(self):
        torch.manual_seed(42)
        query = torch.randn(8, 8, 49, 64)
        key = torch.randn(8, 8, 2048, 64)
        value = torch.randn(8, 8, 2048, 64)
        mask = torch.ones(8, 8, 49, 2048)
        mask[:, :, :10, :100] = 0  # Adding some masked regions

        result, attention = memory_querying_responding(query, key, value, mask=mask)
        self.assertEqual(result.size(), torch.Size([8, 8, 49, 64]))
        self.assertEqual(attention.size(), torch.Size([8, 8, 49, 32]))

    def test_with_dropout(self):
        torch.manual_seed(42)
        query = torch.randn(8, 8, 49, 64)
        key = torch.randn(8, 8, 2048, 64)
        value = torch.randn(8, 8, 2048, 64)
        dropout = torch.nn.Dropout(p=0.1)

        result, attention = memory_querying_responding(query, key, value, dropout=dropout)
        self.assertEqual(result.size(), torch.Size([8, 8, 49, 64]))
        self.assertEqual(attention.size(), torch.Size([8, 8, 49, 32]))

# Running the tests
if __name__ == '__main__':
    unittest.main()
