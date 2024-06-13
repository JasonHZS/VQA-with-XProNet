import os
import json
import torch
from datasets import Dataset

class VQADataset:
    def __init__(self, key, questions, contexts, answers, img_names):
        questions = questions
        contexts = contexts
        answers = answers
        img_names = img_names
        key = key

    def __getitem__(self, idx):
        return {
            'key': key[idx],
            'question': questions[idx],
            'context': contexts[idx],
            'answers': answers[idx],
            'image_name': img_names[idx]
        }

    def __len__(self):
        return len(questions)
 
 
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