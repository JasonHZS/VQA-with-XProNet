import os
import json
import torch 
import open_clip
from PIL import Image
import numpy as np  
from loguru import logger
from datasets import Dataset
from open_clip import tokenizer as clip_tokenizer
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

class VQADataset:
    def __init__(self, key, questions, contexts, answers, img_names):
        self.questions = questions
        self.contexts = contexts
        self.answers = answers
        self.img_names = img_names
        self.key = key

    def __getitem__(self, idx):
        return {
            'key': self.key[idx],
            'question': self.questions[idx],
            'context': self.contexts[idx],
            'answers': self.answers[idx],
            'image_name': self.img_names[idx]
        }

    def __len__(self):
        return len(self.questions)
 
def load_datasets(data_dir, sub_folders, img_dir):
    questions = []
    contexts = []
    answers = []
    img_names = []
    keys = []
    for folder in sub_folders:
        json_file_name = 'all_qs_dict_release_train_500.json' if 'train' in folder else 'all_qs_dict_release_test_500.json'
        json_file_path = os.path.join(data_dir, folder, json_file_name)
        with open(json_file_path) as f:
            data = json.load(f)
            for key in data.keys():
                keys.append(key)
                questions.append(data[key]['question'])
                contexts.append(data[key]['fact_surface'].replace("[[", "").replace("]]", ""))
                # 处理答案格式
                answer_text = data[key]['answer']
                answer_start = contexts[-1].find(answer_text)  # 通过查找答案在上下文中的位置
                answers.append({
                    'answer_start': [answer_start if answer_start != -1 else 0],
                    'text': [answer_text]
                })
                img_names.append(os.path.join(img_dir, data[key]['img_file']))
    
    # 创建Hugging Face datasets对象
    dataset = Dataset.from_dict({
        'key': keys,
        'question': questions,
        'context': contexts,
        'answers': answers,
        'image_name': img_names
    })
    
    return dataset

def get_train_val_dataset():
       project_root = os.getcwd()
       train_data_dir = os.path.join(project_root, 'data/KG_VQA/fvqa/exp_data/train_seen_data')
       test_data_dir = os.path.join(project_root, 'data/KG_VQA/fvqa/exp_data/test_unseen_data')
       img_dir = os.path.join(project_root, "data/KG_VQA/fvqa/exp_data/images/images")
       sub_folders_train = ['train0', 'train1', 'train2', 'train3', 'train4']
       sub_folders_test = ['test0', 'test1', 'test2', 'test3', 'test4']

       train_dataset = load_datasets(train_data_dir, sub_folders_train, img_dir)
       validation_dataset = load_datasets(test_data_dir, sub_folders_test, img_dir) 

       print(train_dataset[0])
       print('训练集大小：', len(train_dataset))
       print('验证集大小：', len(validation_dataset))
       return train_dataset, validation_dataset

def prepare_train_features(examples):
    # Some of the questions have lots of whitespace on the left, which is not useful and will make the
    # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
    # left whitespace
    examples["question"] = [q.lstrip() for q in examples["question"]]

    # Tokenize our examples with truncation and padding, but keep the overflows using a stride. This results
    # in one example possible giving several features when a context is long, each of those features having a
    # context that overlaps a bit the context of the previous feature.
    tokenized_examples = tokenizer(
        examples["question" if pad_on_right else "context"],
        examples["context" if pad_on_right else "question"],
        truncation="only_second" if pad_on_right else "only_first",
        max_length=120,
       #  stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    # Since one example might give us several features if it has a long context, we need a map from a feature to
    # its corresponding example. This key gives us just that.
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    # The offset mappings will give us a map from token to character position in the original context. This will
    # help us compute the start_positions and end_positions.
    offset_mapping = tokenized_examples.pop("offset_mapping")

    # Let's label those examples!
    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []

    for i, offsets in enumerate(offset_mapping):
        # We will label impossible answers with the index of the CLS token.
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        answers = examples["answers"][sample_index]
        # If no answers are given, set the cls_index as answer.
        if len(answers["answer_start"]) == 0:
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
        else:
            # Start/end character index of the answer in the text.
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])

            # Start token index of the current span in the text.
            token_start_index = 0
            while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                token_start_index += 1

            # End token index of the current span in the text.
            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                token_end_index -= 1

            # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
            if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                # Note: we could go after the last offset if the answer is the last word (edge case).
                while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                    token_start_index += 1
                tokenized_examples["start_positions"].append(token_start_index - 1)
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                tokenized_examples["end_positions"].append(token_end_index + 1)
    
    return tokenized_examples

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
   
def get_clip_dim(train_dataset, clip_preprocess, clip_tokenizer):
    image = Image.open(train_dataset[0]['image_name']).convert("RGB")
    image_input = clip_preprocess(image).unsqueeze(0).to(device)  # Unsqueeze 添加一个批次维度
    text_tokens = clip_tokenizer.tokenize(train_dataset[0]['context']).to(device)

    with torch.no_grad():
        image_features = clip_model.encode_image(image_input).float()
        text_features = clip_model.encode_text(text_tokens).float()
        
    return image_features.shape, text_features.shape

def process_combine_data(batch):
    # 处理一批图像数据
    images = [Image.open(path).convert("RGB") for path in batch['image_name']]
    image_inputs = torch.stack([clip_preprocess(image) for image in images]).to(device)

    # 处理一批文本数据
    text_tokens = clip_tokenizer.tokenize([context for context in batch['context']]).to(device)

    with torch.no_grad():
        image_features = clip_model.encode_image(image_inputs).float()
        # print(image_features.shape)
        text_features = clip_model.encode_text(text_tokens).float()
        # print(text_features.shape)

    # 准备输入数据
    # input_ids = tokenizer.encode(batch['question'], batch['context'], add_special_tokens=False, return_tensors="pt")

    # 处理一批问题和上下文数据
    # input_ids = [tokenizer.encode(q, c, add_special_tokens=True, return_tensors="pt") for q, c in zip(batch['question'], batch['context'])]
    # input_ids = [tokenizer.encode(q, c, add_special_tokens=True, return_tensors="pt", max_length=max_length, padding='max_length', truncation=True) for q, c in zip(batch['question'], batch['context'])]
    # input_ids = torch.cat(input_ids, dim=0).to(device)  # 合并批次数据
    input_ids = []
    for q, c in zip(batch['question'], batch['context']):
        input_id = tokenizer.encode(q, c, add_special_tokens=True, return_tensors="pt", max_length=max_length, padding='max_length', truncation=True)
        input_ids.append(input_id)
    input_ids = torch.cat(input_ids, dim=0).to(device, non_blocking=True)  # 合并批次数据

    with torch.no_grad():
        outputs = qamodel(input_ids, output_hidden_states=True)
        last_hidden_states = outputs.hidden_states[-1]
        # print(last_hidden_states.shape)

    # 扩展图像特征和文本特征
    image_features_expanded = image_features.unsqueeze(1).repeat(1, last_hidden_states.shape[1], 1).to(device)
    # print(image_features_expanded.shape)
    text_features_expanded = text_features.unsqueeze(1).repeat(1, last_hidden_states.shape[1], 1).to(device)
    # print(text_features_expanded.shape)

    # 拼接特征
    combined_features = torch.cat([last_hidden_states.to(device), image_features_expanded, text_features_expanded], dim=2)
    
    # 释放中间张量占用的内存
    del last_hidden_states, image_features_expanded, text_features_expanded

    return {"combined_features": combined_features}

def remove_columns(example):
    example.pop("input_ids", None)
    example.pop("attention_mask", None)
    return example

   
if __name__ == '__main__':
    device = select_device()
    tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
    qamodel = AutoModelForQuestionAnswering.from_pretrained("distilbert/distilbert-base-uncased")
    qamodel = qamodel.to(device)
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms('RN101', pretrained='openai', device=device)
    pad_on_right = tokenizer.padding_side == "right"
    max_length = 120 # The maximum length of a feature (question and context)
    doc_stride = 16 # The authorized overlap between two part of the context when splitting it is needed.   
    
    project_root = os.getcwd()
    train_data_dir = os.path.join(project_root, 'data/KG_VQA/fvqa/exp_data/train_seen_data')
    test_data_dir = os.path.join(project_root, 'data/KG_VQA/fvqa/exp_data/test_unseen_data')
    img_dir = os.path.join(project_root, "data/KG_VQA/fvqa/exp_data/images/images")
    sub_folders_train = ['train0', 'train1', 'train2', 'train3', 'train4']
    sub_folders_test = ['test0', 'test1', 'test2', 'test3', 'test4']

    # --------------------------------- train dataset ---------------------------------

    logger.info('开始加载训练数据集')
    train_dataset = load_datasets(train_data_dir, sub_folders_train, img_dir)
    
    indices = np.random.permutation(len(train_dataset))[:6000]
    # 使用选定的索引切割数据集
    small_train_dataset = train_dataset.select(indices)

    logger.info(f"sample: {train_dataset[0]}")
    logger.info(f'训练集大小：{len(train_dataset)}')
  
    logger.info('开始使用CLIP提取图像与文本特征')
    train_combine_dataset = small_train_dataset.map(process_combine_data, batched=True, batch_size=32)
    logger.info('图像与文本特征提取完成')
    logger.info(f"数据集 info: {train_combine_dataset}")
    
    logger.info('开始预处理训练数据集')
    tokenized_train_combine_dataset = train_combine_dataset.map(prepare_train_features, batched=True, remove_columns=train_dataset.column_names)    
    tokenized_train_combine_dataset = tokenized_train_combine_dataset.map(remove_columns, batched=True)
    
    logger.info('训练数据集预处理完成')
    logger.info(f"训练数据集 info: {tokenized_train_combine_dataset}")
    
    tokenized_train_combine_dataset.save_to_disk('/root/autodl-tmp/vqa/VQA-with-XProNet/saved_data/train')
    logger.info('训练数据集保存完毕')
    
    del train_dataset, train_combine_dataset, tokenized_train_combine_dataset

    # --------------------------------- val dataset ---------------------------------
    
    logger.info('开始加载测试数据集')
    validation_dataset = load_datasets(test_data_dir, sub_folders_test, img_dir) 
    logger.info(f'测试集大小：{len(validation_dataset)}')
    
    indices = np.random.permutation(len(validation_dataset))[:2000]
    small_val_dataset = validation_dataset.select(indices)

    logger.info('开始使用CLIP提取图像与文本特征')
    val_combine_dataset = small_val_dataset.map(process_combine_data, batched=True, batch_size=32)
    logger.info('图像与文本特征提取完成')
    logger.info(f"测试数据集 info: {val_combine_dataset}")
    
    logger.info('开始预处理测试数据集')
    tokenized_val_combine_dataset = val_combine_dataset.map(prepare_train_features, batched=True, remove_columns=validation_dataset.column_names)
    tokenized_val_combine_dataset = tokenized_val_combine_dataset.map(remove_columns, batched=True)
    
    logger.info('测试数据集预处理完成')
    logger.info(f"测试数据集 info: {tokenized_val_combine_dataset}")
    
    tokenized_val_combine_dataset.save_to_disk('/root/autodl-tmp/vqa/VQA-with-XProNet/saved_data/val')
    logger.info('测试数据集保存完毕')

