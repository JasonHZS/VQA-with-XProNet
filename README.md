# 说明
- first_experiment 文件夹是第一期实验的 vqa 代码，即没有加入原型的 vqa 实验；
- test_qamodel.ipynb 可供参考，主要都是实现的思路，可以从这个 notebook 看到每一步的尝试流程；
- 有些代码用到了他人仓库的代码，有些是没用的，没删除干净，可以忽略；

# vqa+prototype实验

## 1. 数据集

- 数据集路径：`/root/autodl-tmp/vqa/VQA-with-XProNet/data/KG_VQA/fvqa/exp_data`；
- 数据集预处理即保存：data_preprocess.py；
- 事先处理好数据并保存，训练时在加载，如果换了 qamodel 或者 clip 模型，那必须重新处理数据；

## 2. 模型

- qamodel 采用来自 huggingface 的 distilbert 模型，这个比较小，可以快速训练，换用其他模型也可以，但是需要修改代码适配模型要求的输入维度等；
- clip 模型采用了 openai 的 clip 模型，clip 有很多可以选择的模型，根据需要实验不同的模型，同样注意clip 编码后的向量维度；

## 3. 原型

- 原型初始化py脚本：prototype_init.py；
       - 输入clip 提取的图像与context（fact）特征向量，在拼接 qamodel 编码的问题+context（fact） 向量；
       - 使用 kmeans 对输入的拼接向量进行聚类，得到 n 个原型，保存；
- 原型查询与响应：
       - 复用原本 Xpronet 的代码：见 prototype.py 的 `memory_querying_responding` 与 `MultiThreadMemory`；

## 4. 训练与评估

- 训练和评估都写在 main.py；
- 训练时，将 clip + qamodel 拼接的向量 combine_feature 与原型响应的 respone 向量 拼接，输入到 qamodel 的 inputs_embeds

## 5. 注意

有一些参数是因为一些对应关系，所以写死在代码里的，比如`VqaPrototypeModel`的初始化的`batch_size=16` 和 `seq_length=38`，修改时请注意；
总流程这样下来算是跑通了，但是之前说过效果不一定好，可以尝试调整一些参数，比如训练的参数（这些没空调优了，你可以自己调一下），原型的数量，原型的维度，换用 clip 或者 qamodel 等；