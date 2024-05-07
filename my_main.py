import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset

from my_vqadata import VQADataset 
from my_train import train_model, validate
from my_model import VQAModel


startEpoch = 0
project_root = os.getcwd()
print("project_root:", project_root)

train_data_dir = project_root+'/data/KG_VQA/fvqa/exp_data/train_data'
sub_folders = ['train0', 'train1', 'train2', 'train3', 'train4']
img_dir = project_root+"/data/KG_VQA/fvqa/exp_data/images/images"
vocab_file = project_root+'/data/KG_VQA/fvqa/exp_data/common_data/vocab_train_500.json'
# 初始化词汇表和数据集列表
vocab = {}
# 遍历每个子文件夹，加载数据集
datasets = []
for folder in sub_folders:
       json_file = os.path.join(train_data_dir, folder, 'all_qs_dict_release_train_500.json')
       dataset = VQADataset(json_file=json_file, img_dir=img_dir, vocab_file=vocab_file)
       datasets.append(dataset)
       # 更新词汇表
       vocab.update(dataset.vocab)
# 合并所有数据集
train_dataset = ConcatDataset(datasets)
# 打印数据集大小
print("train_dataset size:", len(train_dataset))
# 词汇表大小
vocab_size = len(vocab)
print("train vocab_size:", vocab_size)
# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True, drop_last=True)
data_loaders = {'train': train_loader}


test_data_dir = project_root+'/data/KG_VQA/fvqa/exp_data/test_data'
sub_folders = ['test0', 'test1', 'test2', 'test3', 'test4']
vocab_file = project_root+'/data/KG_VQA/fvqa/exp_data/common_data/vocab_test_500.json'
datasets = []
for folder in sub_folders:
       json_file = os.path.join(test_data_dir, folder, 'all_qs_dict_release_test_500.json')
       dataset = VQADataset(json_file=json_file, img_dir=img_dir, vocab_file=vocab_file)
       datasets.append(dataset)
       # 更新词汇表
       vocab.update(dataset.vocab)

# 合并所有数据集
test_dataset = ConcatDataset(datasets)
# 打印数据集大小
print("test_dataset size:", len(test_dataset))
# 词汇表大小
vocab_size = len(vocab)
print("all vocab_size:", vocab_size)
val_loader = DataLoader(test_dataset, batch_size=10, shuffle=False, drop_last=True)
data_loaders['val'] = val_loader

# model = VQAModel(vocab_size=vocab_size, output_size=vocab_size)
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
#                      lr=0.01, weight_decay=0.0005)

# # 判断并选择设备
# def select_device():
#        if torch.cuda.is_available():
#               device = torch.device("cuda")  # 优先使用CUDA（NVIDIA GPU）
#               print("Using CUDA (GPU)")
#        elif torch.backends.mps.is_available():
#               device = torch.device("mps")  # 如果CUDA不可用但MPS可用，使用MPS（Apple Silicon）
#               print("Using MPS (Apple Silicon)")
#        else:
#               device = torch.device("cpu")  # 如果都不可用，使用CPU
#               print("Using CPU")
#        return device

# device = select_device()

# model.to(device)
# best_acc = 0
# save_dir = "/Users/oasis/Documents/GitHub Project/VQA-with-XProNet/my_ckp"
# model = train_model(model, 
#               data_loaders, 
#               train_dataset.vocab,
#               device,
#               criterion, 
#               optimizer, 
#               scheduler=None,
#               save_dir=save_dir,
#               num_epochs=1, 
#               best_accuracy=best_acc, 
#               start_epoch=startEpoch
#        )
