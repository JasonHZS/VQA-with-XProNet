import torch 
from loguru import logger
from datasets import load_from_disk
from torch.utils.data import DataLoader
from transformers import DefaultDataCollator
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

# 自定义 hit@k 评估函数
def hit_at_k(predictions, references, k):
    hits = 0
    for pred, ref in zip(predictions, references):
        if ref in pred[:k]:
            hits += 1
    return hits / len(references)

def eval(tokenizer, trained_model, validation_dataset, device):
       # 提取预测和参考答案
       predictions = []
       references = []

       for i in range(len(validation_dataset)):
              sample = validation_dataset[i]
              combined_features = torch.tensor(sample['combined_features'], dtype=torch.float32).to(device)
              question = sample['question']
              context = sample['context']
              answers = sample['answers']
              attention_mask = torch.tensor(sample['attention_mask'], dtype=torch.float32).to(device)
              
              with torch.no_grad():
                     inputs = tokenizer(question, context, return_tensors="pt", padding=True, truncation=True)
                     outputs = trained_model(combined_features=combined_features.unsqueeze(0), 
                                             attention_mask=attention_mask.unsqueeze(0))
                     
              start_logits = outputs[1]
              end_logits = outputs[2]
              answer_start_index = start_logits.argmax(dim=1)
              answer_end_index = end_logits.argmax(dim=1) 
              # logger.info(f"answer_start_index: {answer_start_index}")
              # logger.info(f"answer_end_index: {answer_end_index}")
              
              if answer_end_index < answer_start_index:
                     answer_end_index = answer_start_index
              predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
              predict_answer = tokenizer.decode(predict_answer_tokens, skip_special_tokens=True)
              # predict_answer = tokenizer.convert_tokens_to_string(predict_answer_tokens[answer_start_index:answer_end_index])
              # predict_answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(predict_answer_tokens))

              predictions.append(predict_answer)
              references.append(' '.join(answers['text']))

       # 计算 hit@5 和 hit@10
       logger.info(f"predictions: {predictions}")
       logger.info(f"references: {references}")
       hit1 = hit_at_k(predictions, references, 1)
       hit5 = hit_at_k(predictions, references, 5)
       hit10 = hit_at_k(predictions, references, 10)

       print(f"hit@1: {hit1:.2%}")
       print(f"hit@5: {hit5:.2%}")
       print(f"hit@10: {hit10:.2%}")


# if __name__ == '__main__':
#        # 重新加载模型
#        trained_model = AutoModelForQuestionAnswering.from_pretrained("test-squad-trained")
#        # 重新加载分词器
#        tokenizer = AutoTokenizer.from_pretrained("test-squad-trained")

#        # 加载验证集
#        validation_dataset = load_from_disk('/root/autodl-tmp/vqa/VQA-with-XProNet/saved_data/val')
#        eval(tokenizer, trained_model, validation_dataset)

