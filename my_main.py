import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from my_vqadata import VQADataset 
from my_train import train_model, validate
from my_model import VQAModel


startEpoch = 0
project_root = os.getcwd()
print("project_root:", project_root)
train_json_path = project_root+"/data/new_dataset_release/train_qs_data.json"
test_json_path = project_root+"/data/new_dataset_release/test_qs_data.json"
img_dir = project_root+"/data/new_dataset_release/images"
train_vocab_file = project_root+"/data/new_dataset_release/train_vocab.json"
test_vocab_file = project_root+"/data/new_dataset_release/test_vocab.json"

# 创建数据集实例
train_dataset = VQADataset(json_file=train_json_path, img_dir=img_dir, vocab_file=train_vocab_file)
test_dataset = VQADataset(json_file=test_json_path, img_dir=img_dir, vocab_file=test_vocab_file)
# 合并训练集和测试集的词汇表
train_dataset.vocab.update(test_dataset.vocab)
vocab_size = len(train_dataset.vocab)
print("vocab_size:", vocab_size)

train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True, drop_last=True)
val_loader = DataLoader(test_dataset, batch_size=10, shuffle=False, drop_last=True)
data_loaders = {'train': train_loader, 'val': val_loader}

model = VQAModel(vocab_size=vocab_size, output_size=vocab_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                     lr=0.01, weight_decay=0.0005)

# 判断并选择设备
def select_device():
       if torch.cuda.is_available():
              device = torch.device("cuda")  # 优先使用CUDA（NVIDIA GPU）
              print("Using CUDA (GPU)")
       elif torch.backends.mps.is_available():
              device = torch.device("mps")  # 如果CUDA不可用但MPS可用，使用MPS（Apple Silicon）
              print("Using MPS (Apple Silicon)")
       else:
              device = torch.device("cpu")  # 如果都不可用，使用CPU
              print("Using CPU")
       return device

device = select_device()

model.to(device)
best_acc = 0
save_dir = "/Users/oasis/Documents/GitHub Project/VQA-with-XProNet/my_ckp"
model = train_model(model, 
              data_loaders, 
              train_dataset.vocab,
              device,
              criterion, 
              optimizer, 
              scheduler=None,
              save_dir=save_dir,
              num_epochs=1, 
              best_accuracy=best_acc, 
              start_epoch=startEpoch
       )

# Train Loss: 3.9562 Acc: 0.193 (9/4660)
# Epoch Train Time: 2m 37s
# Validation Loss: 7.8746 Acc: 0.172 (2/1160)
# Epoch Validation Time: 0m 11s
# Training complete in 2m 49s
# Best val Acc: 0.171527