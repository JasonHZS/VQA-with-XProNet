import torch
import numpy as np
import torch.nn as nn
import open_clip
from loguru import logger
from torch.utils.data import Subset
from prototype import MultiThreadMemory
from datasets import load_from_disk
from prototype import VqaPrototypeModel
from transformers import PretrainedConfig
from transformers import default_data_collator
from torch.utils.data.dataloader import default_collate
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, TrainingArguments, Trainer
from evaluation import eval

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
       clip_model, _, _ = open_clip.create_model_and_transforms('RN101', 
                                                                pretrained='openai', 
                                                                device=device)
       return clip_model.visual.output_dim, clip_model.token_embedding.weight.shape[1]

class VqaPrototypeConfig(PretrainedConfig):
    def __init__(self, prototype_size, qamodel_config, image_features, **kwargs):
        super().__init__(**kwargs)
        self.prototype_size = prototype_size
        self.qamodel_config = qamodel_config
        self.image_features = image_features
        
if __name__ == '__main__':
       device = select_device()
       max_length = 120 # The maximum length of a feature (question and context)
       doc_stride = 16 # The authorized overlap between two part of the context when splitting it is needed.
       
       logger.info("Loading tokenizer and model...")
       tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
       qamodel = AutoModelForQuestionAnswering.from_pretrained("distilbert/distilbert-base-uncased")
       qamodel = qamodel.to(device)
       
       logger.info("Loading datasets...")
       tokenized_combine_train_dataset = load_from_disk('/root/autodl-tmp/vqa/VQA-with-XProNet/saved_data/train')
       tokenized_combine_val_dataset = load_from_disk('/root/autodl-tmp/vqa/VQA-with-XProNet/saved_data/val')
       prototype_vectors = torch.load('/root/autodl-tmp/vqa/VQA-with-XProNet/saved_data/init_prototype/prototype_vectors.pt')
       # logger.info(tokenized_combine_train_dataset.column_names)
       
       
       indices = np.random.permutation(len(tokenized_combine_val_dataset))[:20]
       small_val_dataset = tokenized_combine_val_dataset.select(indices)
       indices = np.random.permutation(len(tokenized_combine_train_dataset))[:200]
       small_train_dataset = tokenized_combine_train_dataset.select(indices)

       
       # 设置训练参数
       model_name = "vqa-bert"
       batch_size = 10
       image_features, text_features = get_clip_dim()
       vqa_model = VqaPrototypeModel(prototype_vectors, 
                                     qamodel=qamodel,
                                     device=device,
                                     image_features=image_features).to(device) 

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
              train_dataset=small_train_dataset,
              eval_dataset=small_val_dataset,
              data_collator=default_data_collator,  
              tokenizer=tokenizer
              )

       torch.cuda.empty_cache()
       logger.info("Training...")
       trainer.train()

       logger.info("Evaluation...")
       # TODO:修改评估函数，接受tokenized_combine_val_dataset输入
       eval(tokenizer, vqa_model, small_val_dataset)
       
       # trainer.save_model("test-squad-trained")
       # logger.info("Model saved!")
       
       # # 创建配置对象并保存
       # vqa_config = VqaPrototypeConfig(
       #        prototype_size=100,
       #        qamodel_config=qamodel.config, 
       #        image_features=image_features
       #        )
       # vqa_config.save_pretrained("test-squad-trained")
       # logger.info("Config saved!")
       
