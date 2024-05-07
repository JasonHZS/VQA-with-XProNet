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

def load_datasets(data_dir, sub_folders, img_dir, vocab_file):
    datasets = []
    vocab = {}
    for folder in sub_folders:
        json_file = os.path.join(data_dir, folder, 'all_qs_dict_release_train_500.json' if 'train' in data_dir else 'all_qs_dict_release_test_500.json')
        dataset = VQADataset(json_file=json_file, img_dir=img_dir, vocab_file=vocab_file)
        datasets.append(dataset)
        vocab.update(dataset.vocab)
    dataset = ConcatDataset(datasets)
    print(f"{data_dir.split('/')[-1]} dataset size:", len(dataset))
    return dataset, vocab

def setup_data_loader(datasets, batch_size=10, shuffle=True, drop_last=True):
    loaders = {}
    for key, dataset in datasets.items():
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
        loaders[key] = loader
    return loaders

train_data_dir = project_root+'/data/KG_VQA/fvqa/exp_data/train_data'
test_data_dir = project_root+'/data/KG_VQA/fvqa/exp_data/test_data'
img_dir = project_root+"/data/KG_VQA/fvqa/exp_data/images/images"
train_vocab_file = project_root+'/data/KG_VQA/fvqa/exp_data/common_data/vocab_train_500.json'
test_vocab_file = project_root+'/data/KG_VQA/fvqa/exp_data/common_data/vocab_test_500.json'

sub_folders_train = ['train0', 'train1', 'train2', 'train3', 'train4']
sub_folders_test = ['test0', 'test1', 'test2', 'test3', 'test4']

train_dataset, train_vocab = load_datasets(train_data_dir, sub_folders_train, img_dir, train_vocab_file)
test_dataset, test_vocab = load_datasets(test_data_dir, sub_folders_test, img_dir, test_vocab_file)

# 合并词汇表
full_vocab = train_vocab.copy()
full_vocab.update(test_vocab)
vocab_size = len(full_vocab)
print("Total vocab size:", vocab_size)

data_loaders = setup_data_loader({'train': train_dataset, 'val': test_dataset})

model = VQAModel(vocab_size=vocab_size, output_size=vocab_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.01, weight_decay=0.0005)

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
              full_vocab,
              device,
              criterion, 
              optimizer, 
              scheduler=None,
              save_dir=save_dir,
              num_epochs=10, 
              best_accuracy=best_acc, 
              start_epoch=startEpoch
       )

# validate(model, data_loaders['val'], full_vocab, device, criterion)
# Train Loss: 0.5901 Acc: 2.322 (320/13780)
# Epoch Train Time: 2m 54s
# Validation Loss: 1.2014 Acc: 0.080 (11/13670)
# Epoch Validation Time: 1m 44s
# Training complete in 46m 31s
# Best val Acc: 4.760512