import torch
from torch.utils.data import Dataset
import json
from PIL import Image
import os
from torchvision import transforms
from torch.utils.data import DataLoader

class VQADataset(Dataset):
    def __init__(self, json_file, img_dir, transform=None):
        with open(json_file, 'r') as f:
            self.data = json.load(f)
        self.keys = list(self.data.keys())
        self.img_dir = img_dir
        self.transform = transform

        if self.transform is None:
            # 如果没有指定转换，则使用默认的转换
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        key = self.keys[index]
        item = self.data[key]
        img_path = os.path.join(self.img_dir, item['img_file'])
        image = Image.open(img_path).convert('RGB')  # 确保图像是 RGB 格式

        # 应用转换
        if self.transform:
            image = self.transform(image)

       #  fact_surface = item['fact_surface']
        answer = item['answer']
        question = item['question']

        return {"image": image, "answer": answer, "question": question}



if __name__ == '__main__':
       # 定义 JSON 文件和图像目录路径
       json_file = "/root/vqa/VQA-with-XProNet/data/new_dataset_release/train_qs_data.json"
       img_dir = "/root/vqa/VQA-with-XProNet/data/new_dataset_release/images"

       # 创建数据集实例
       dataset = VQADataset(json_file, img_dir)

       # 使用 DataLoader 来加载数据
       data_loader = DataLoader(dataset, batch_size=1, shuffle=True)

       # 迭代 DataLoader 进行测试
       for batch in data_loader:
              print(batch['image'].shape, batch['question'], batch['answer'])
              break  # 只打印第一批数据来检查
