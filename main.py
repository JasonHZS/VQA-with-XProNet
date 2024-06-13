import os
import json
import torch
import evaluate
import open_clip
import collections
import numpy as np
import torch.nn as nn
from PIL import Image
from tqdm.auto import tqdm
import torch.nn.functional as F
from collections import OrderedDict
from prototype import MultiThreadMemory
from datasets import Dataset, load_metric
from transformers import default_data_collator
from open_clip import tokenizer as clip_tokenizer
from torch.utils.data.dataloader import default_collate
# from transformers import BertTokenizerFast, BertForQuestionAnswering, TrainingArguments, Trainer
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, TrainingArguments, Trainer

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




if __name__ == '__main__':
       device = select_device()
       max_length = 120 # The maximum length of a feature (question and context)
       doc_stride = 16 # The authorized overlap between two part of the context when splitting it is needed.
       
       tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
       qamodel = AutoModelForQuestionAnswering.from_pretrained("distilbert/distilbert-base-uncased")
       qamodel = qamodel.to(device)

