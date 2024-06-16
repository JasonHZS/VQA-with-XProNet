import torch 
from datasets import load_from_disk
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

# 自定义 hit@k 评估函数
def hit_at_k(predictions, references, k):
    hits = 0
    for pred, ref in zip(predictions, references):
        if ref in pred[:k]:
            hits += 1
    return hits / len(references)

def eval(tokenizer, trained_model, validation_dataset):
       # 提取预测和参考答案
       predictions = []
       references = []

       for i in range(len(validation_dataset)):
              sample = validation_dataset[i]
              combined_features = sample['combined_features']
              question = sample['question']
              context = sample['context']
              answers = sample['answers']
              attention_mask = sample['attention_mask']
              start_positions = sample['start_positions']
              end_positions = sample['end_positions']
              # 预测答案
              inputs = tokenizer(question, context, return_tensors="pt")
              with torch.no_grad():
                     outputs = trained_model(combined_features=combined_features,
                                             attention_mask=attention_mask,
                                             start_positions=start_positions, 
                                             end_positions=end_positions)
                     
              answer_start_index = outputs.start_logits.argmax()
              answer_end_index = outputs.end_logits.argmax() 
              
              predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
              predict_answer = tokenizer.decode(predict_answer_tokens, skip_special_tokens=True)
              
              predictions.append(predict_answer)
              references.append(answers)

       # 计算 hit@5 和 hit@10
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

