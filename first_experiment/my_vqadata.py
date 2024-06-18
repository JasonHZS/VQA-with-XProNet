import torch
from torch.utils.data import Dataset
import json
from PIL import Image
import os
from collections import Counter
from torchvision import transforms
from torch.utils.data import DataLoader

class VQADataset(Dataset):
    def __init__(self, transform=None, json_file=None, img_dir=None, vocab_file=None):
        self.json_file = json_file
        self.img_dir = img_dir
        self.vocab_file = vocab_file  # 添加一个词汇表文件的参数
        with open(self.json_file, 'r') as f:
            self.data = json.load(f)
        self.keys = list(self.data.keys())
        self.transform = transform

        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])
        
        # 检查是否存在词汇表文件，如果不存在则创建词汇表
        if os.path.exists(self.vocab_file):
            with open(self.vocab_file, 'r') as f:
                self.vocab = json.load(f)
        else:
            self.vocab = self.create_vocab(self.data)
            with open(self.vocab_file, 'w') as f:
                json.dump(self.vocab, f)
            # with open(self.vocab_file+'_test', 'w') as f:
            #     json.dump(self.vocab, f)

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        key = self.keys[index]
        item = self.data[key]
        img_path = os.path.join(self.img_dir, item['img_file'])
        image = Image.open(img_path).convert('RGB')
        if self.transform: 
            image = self.transform(image)

        answer = item['answer']
        question = item['question']
        return {"image": image, "answer": answer, "question": question}

    def create_vocab(self, data):
        counter = Counter()
        for key, item in data.items():
            question_words = item['question'].split()
            answer_words = item['answer'].split()
            counter.update(question_words)
            counter.update(answer_words)
        
        # 将特殊标记添加到词汇表开始
        vocab = {word: idx + 4 for idx, word in enumerate(counter)}
        vocab['<pad>'] = 0
        vocab['<unk>'] = 1
        vocab['<start>'] = 2
        vocab['<end>'] = 3
        
        return vocab


# if __name__ == '__main__':
#     train_json_path = "/Users/oasis/Documents/GitHub Project/VQA-with-XProNet/data/new_dataset_release/train_qs_data.json"
#     img_dir = "/Users/oasis/Documents/GitHub Project/VQA-with-XProNet/data/new_dataset_release/images"
#     vocab_file = "/Users/oasis/Documents/GitHub Project/VQA-with-XProNet/data/new_dataset_release/vocab.json"
#     dataset = VQADataset(json_file=train_json_path, img_dir=img_dir, vocab_file=vocab_file)
#     # print(dataset.vocab)  # 打印词汇表查看

#     # 使用 DataLoader 来加载数据
#     data_loader = DataLoader(dataset, batch_size=2, shuffle=True)

#     # 迭代 DataLoader 进行测试
#     for batch in data_loader:
#             print(batch['image'].shape, batch['question'], batch['answer'])
#             break  # 只打印第一批数据来检查

    # print(torch.__version__)
    # print(torch.cuda.is_available())

