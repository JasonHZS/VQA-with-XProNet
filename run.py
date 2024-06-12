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

pad_on_right = tokenizer.padding_side == "right"

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
        max_length=256,
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

tokenizer = BertTokenizerFast.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
qa_bert = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
qa_bert = qa_bert.to(device)
clip_model, _, clip_preprocess = open_clip.create_model_and_transforms('ViT-B-16', pretrained='openai', device=device)

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
    input_ids = [tokenizer.encode(q, c, add_special_tokens=True, return_tensors="pt", max_length=max_length, padding='max_length', truncation=True) for q, c in zip(batch['question'], batch['context'])]
    input_ids = torch.cat(input_ids, dim=0).to(device)  # 合并批次数据

    with torch.no_grad():
        outputs = qa_bert(input_ids, output_hidden_states=True)
        last_hidden_states = outputs.hidden_states[-1]
        # print(last_hidden_states.shape)

    # 扩展图像特征和文本特征
    image_features_expanded = image_features.unsqueeze(1).repeat(1, last_hidden_states.shape[1], 1).to(device)
    # print(image_features_expanded.shape)
    text_features_expanded = text_features.unsqueeze(1).repeat(1, last_hidden_states.shape[1], 1).to(device)
    # print(text_features_expanded.shape)

    # 拼接特征
    combined_features = torch.cat([last_hidden_states.to(device), image_features_expanded, text_features_expanded], dim=2)

    return {"combined_features": combined_features}
    # return {"combined_features": combined_features.cpu().numpy()}

# 随机生成100个唯一的索引
indices = np.random.permutation(len(train_dataset))[:100]
# 使用选定的索引切割数据集
small_train_dataset = train_dataset.select(indices)

indices = np.random.permutation(len(validation_dataset))[:20]
small_val_dataset = validation_dataset.select(indices)

processed_train_dataset = small_train_dataset.map(process_combine_data, batched=True, batch_size=32)
processed_val_dataset = small_val_dataset.map(process_combine_data, batched=True, batch_size=32)

# 处理数据集
tokenized_combine_train_dataset = processed_train_dataset.map(prepare_train_features, batched=True, remove_columns=['key', 'question', 'context', 'answers', 'image_name'])
tokenized_combine_val_dataset = processed_val_dataset.map(prepare_train_features, batched=True, remove_columns=['key', 'question', 'context', 'answers', 'image_name'])

# 假设 processed_train_dataset 是已经加载的数据集
features = processed_train_dataset['combined_features']

# 将特征列表转换为 NumPy 数组，以便用于 K-Means 算法
features_matrix = np.array(features)
print(f"features_matrix shape: {features_matrix.shape}")

# features_matrix = features_matrix.reshape(features_matrix.shape[1], -1)
features_matrix = features_matrix.mean(axis=1)
print(f"features_matrix shape: {features_matrix.shape}")

# 设置 K-Means 算法，聚类数为 10
kmeans = KMeans(n_clusters=10, random_state=0).fit(features_matrix)

# 获取每个簇的中心点，这些中心点即为原型向量
prototype_vectors = kmeans.cluster_centers_
print(prototype_vectors.shape)
# 打印原型向量，查看结果
print("Prototype vectors:\n", prototype_vectors)
prototype_vectors = torch.tensor(prototype_vectors, dtype=torch.float32)
prototype_vectors = prototype_vectors.unsqueeze(0).to(device)

class VqaPrototypeModel(nn.Module):
    def __init__(self, prototype_vectors, device, batch_size=16, seq_length=38):
        super(VqaPrototypeModel, self).__init__()
        self.device = device
        self.bert = qa_bert.to(self.device)
        self.prototype = MultiThreadMemory(h=16, d_model=2048, topk=3, dropout=0.1, device=self.device).to(self.device)
        self.prototype_vectors = prototype_vectors.repeat(batch_size, seq_length, 1).to(self.device)
        # print(f"prototype_vectors shape: {prototype_vectors.shape}")
        self.fc = nn.Linear(4096, 1024).to(self.device)  # 注意，如果原始特征和响应被拼接，这里的输入维度应为 2048 + feature_dim

    def forward(self, combined_features, start_positions, end_positions):
        combined_features = torch.tensor(combined_features).to(self.device)
        # print(f"combined_features shape: {combined_features.shape}")

        response = self.prototype(combined_features, self.prototype_vectors, self.prototype_vectors, device=self.device)
        # print("Shape of Response:", response.shape)
        
        # 拼接原型响应和 BERT 输出
        final_combined_features = torch.cat([combined_features, response], dim=2)
        # print("Shape of final_combined_features:", final_combined_features.shape)
        
        reduced_features = self.fc(final_combined_features)
        # print("Shape of reduced_features:", reduced_features.shape)
        
        outputs = self.bert(inputs_embeds=reduced_features, start_positions=start_positions, end_positions=end_positions)
        
        return outputs.loss, outputs.start_logits, outputs.end_logits


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def memory_querying_responding(query, key, value, mask=None, dropout=None, topk=32):
    # 计算 query 向量的维度
    d_k = query.size(-1) 
    # 根据 scaled dot-product attention 计算 query 和 key 的相似度得分
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    # 如果提供了 mask，使用 mask 更新得分
    # 在 mask 中为 0 的位置上设置得分为极小值，以在 softmax 后这些位置的权重接近 0
    if mask is not None:
       scores = scores.masked_fill(mask == 0, torch.finfo(scores.dtype).min)
    # 从 scores 中选出 topk 最高的得分和对应的索引
    selected_scores, idx = scores.topk(topk)
    # 扩展 value 张量，使其在第三维度（query维度）上重复，以便对每个查询选择相应的 value
    dummy_value = value.unsqueeze(2).expand(idx.size(0), idx.size(1), idx.size(2), value.size(-2), value.size(-1))
    # 扩展索引，使其在最后一个维度（embedding维度）上重复，以便从 dummy_value 中选取特定元素
    dummy_idx = idx.unsqueeze(-1).expand(idx.size(0), idx.size(1), idx.size(2), idx.size(3), value.size(-1))
    # 使用扩展后的索引从扩展后的 value 张量中选取元素，这些元素是由 top-k 得分确定的
    selected_value = torch.gather(dummy_value, 3, dummy_idx)
    # 对选择的得分应用 softmax，计算最终的注意力权重
    p_attn = F.softmax(selected_scores.float(), dim=-1)
    # 如果提供了 dropout 模块，则在注意力权重上应用 dropout
    if dropout is not None:
       p_attn = dropout(p_attn)
    # 使用注意力权重对选取的 value 进行加权求和，计算最终的输出
    return torch.matmul(p_attn.unsqueeze(3), selected_value).squeeze(3), p_attn


class MultiThreadMemory(nn.Module):
    def __init__(self, h, d_model, dropout=0.1, topk=32, device='cuda'):
        super(MultiThreadMemory, self).__init__()
        assert d_model % h == 0 # 输入和输出张量的维度d_model必须能被头数head整除
        self.d_k = d_model // h
        self.h = h
        self.device = device
        self.linears = clones(nn.Linear(d_model, d_model), 4).to(self.device)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout).to(self.device)
        self.topk = topk

    def forward(self, query, key, value, device, mask=None, layer_past=None):
        """
        这个方法处理跨模态信息的查询和响应，它使用了一个多头注意力机制来处理query, key, 和 value。
        它通过调用 memory_querying_responding 函数实现了跨模态原型的选择和相应的交互，
        这符合跨模态原型矩阵（Shared Cross-modal Prototype Matrix）的概念，
        允许模型在文本和视觉特征间进行交互和信息融合。
        """

        query = query.to(device)
        key = key.to(device)
        value = value.to(device)
        
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        if layer_past is not None and layer_past.shape[2] == key.shape[1] > 1:
            query = self.linears[0](query)
            key, value = layer_past[0], layer_past[1]
            present = torch.stack([key, value])
        else:
            query, key, value = \
                [l(x) for l, x in zip(self.linears, (query, key, value))]

        if layer_past is not None and not (layer_past.shape[2] == key.shape[1] > 1):
            past_key, past_value = layer_past[0], layer_past[1]
            key = torch.cat((past_key, key), dim=1)
            value = torch.cat((past_value, value), dim=1)
            present = torch.stack([key, value])
        
        query, key, value = \
            [x.view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for x in [query, key, value]]

        x, self.attn = memory_querying_responding(query, key, value, mask=mask, dropout=self.dropout, topk=self.topk)

        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        if layer_past is not None:
            return self.linears[-1](x), present
        else:
            return self.linears[-1](x)

torch.cuda.empty_cache()
vqa_model = VqaPrototypeModel(prototype_vectors, device=device).to(device) 

# 设置训练参数
model_name = "vqa-bert"
batch_size = 16

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

trainer.train()