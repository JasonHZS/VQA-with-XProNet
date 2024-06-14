import torch
import numpy as np
import torch.nn as nn
import open_clip
from PIL import Image
from prototype import MultiThreadMemory
from datasets import load_from_disk
from prototype import VqaPrototypeModel
from open_clip import tokenizer as clip_tokenizer
from transformers import default_data_collator
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

def get_clip_dim():
       clip_model, _, _ = open_clip.create_model_and_transforms('RN101', pretrained='openai', device=device)
              
       return clip_model.visual.output_dim, clip_model.token_embedding.weight.shape[1]


if __name__ == '__main__':
       device = select_device()
       max_length = 120 # The maximum length of a feature (question and context)
       doc_stride = 16 # The authorized overlap between two part of the context when splitting it is needed.
       
       tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
       qamodel = AutoModelForQuestionAnswering.from_pretrained("distilbert/distilbert-base-uncased")
       qamodel = qamodel.to(device)
       
       tokenized_combine_train_dataset = load_from_disk('/root/autodl-tmp/vqa/VQA-with-XProNet/saved_data/train')
       tokenized_combine_val_dataset = load_from_disk('/root/autodl-tmp/vqa/VQA-with-XProNet/saved_data/val')
       prototype_vectors = torch.load('/root/autodl-tmp/vqa/VQA-with-XProNet/saved_data/init_prototype/prototype_vectors.pt')
       
       # 设置训练参数
       model_name = "vqa-bert"
       batch_size = 4
       image_features = get_clip_dim()
       vqa_model = VqaPrototypeModel(prototype_vectors, 
                                     qamodel=qamodel,
                                     device=device).to(device) 

       args = TrainingArguments(
       output_dir=f"{model_name}-finetuned-squad",
       evaluation_strategy="epoch",
       learning_rate=0.01,
       per_device_train_batch_size=batch_size,
       per_device_eval_batch_size=batch_size,
       num_train_epochs=1,
       weight_decay=0.01,
       logging_steps=5,
       gradient_accumulation_steps=4,
       )

       trainer = Trainer(
       model=vqa_model,
       args=args,
       train_dataset=tokenized_combine_train_dataset,
       eval_dataset=tokenized_combine_val_dataset,
       data_collator=default_data_collator,  
       tokenizer=tokenizer
       )

       torch.cuda.empty_cache()
       trainer.train()

