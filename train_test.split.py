import json
import pandas as pd
import numpy as np
import os 

data_path = '/root/vqa/VQA-with-XProNet/data/new_dataset_release/all_qs_dict_release.json'

# 读取 json 文件
with open(data_path, 'r') as f:
       data = json.load(f)

print("数据集大小：", len(data))

for k in data.keys():
    print(k)
    print(data[k])
    break

img_dir = '/root/vqa/VQA-with-XProNet/data/new_dataset_release/images'
img_list = os.listdir(img_dir)
print("图片数量：", len(img_list))

print('ILSVRC2012_test_00050748.JPEG' in img_list)

# 统计data['img_file']中的图片名称为 val，train，test的数量
val = 0
train = 0
test = 0
for k in data.keys():
    img_name = data[k]['img_file']
    if 'val' in img_name:
        train += 1
    elif 'test' in img_name:
        test += 1
    else:
        print(img_name)

print("训练集数据数量：",  train)
print("测试集数据数量：",  test)

# 判断在data['img_file']中有多少图片在img_list中
count = 0
for k in data.keys():
    img_name = data[k]['img_file']
    if img_name in img_list:
        count += 1

print("img_file在img_list中的图片数量：", count)

# 将 json data 随机划分成训练集与测试集

data_list = list(data.items())
np.random.shuffle(data_list)
train_data = data_list[:int(len(data_list)*0.8)]
test_data = data_list[int(len(data_list)*0.8):]
# 保存
train_data_path = '/root/vqa/VQA-with-XProNet/data/new_dataset_release/train_qs_data.json'
test_data_path = '/root/vqa/VQA-with-XProNet/data/new_dataset_release/test_qs_data.json'


# 把 list 形态的 test_data转成 json 格式
test_data = dict(test_data)
train_data = dict(train_data)


# with open(train_data_path, 'w') as json_file:
#     json.dump(train_data, json_file, indent=4)

# with open(test_data_path, 'w') as json_file:
#     json.dump(test_data, json_file, indent=4)
