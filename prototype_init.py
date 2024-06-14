import torch
import numpy as np
from loguru import logger
from sklearn.cluster import KMeans
from datasets import load_from_disk

def calculate_prototype_vectors(processed_train_dataset, n_clusters=20):
       # 假设 processed_train_dataset 是已经加载的数据集
       features = processed_train_dataset['combined_features']

       # 将特征列表转换为 NumPy 数组，以便用于 K-Means 算法
       features_matrix = np.array(features)
       logger.info(f"features_matrix shape: {features_matrix.shape}")

       # features_matrix = features_matrix.reshape(features_matrix.shape[1], -1)
       features_matrix = features_matrix.mean(axis=1)
       logger.info(f"features_matrix shape: {features_matrix.shape}")

       # 设置 K-Means 算法
       kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(features_matrix)

       # 获取每个簇的中心点，这些中心点即为原型向量
       prototype_vectors = kmeans.cluster_centers_
       logger.info(prototype_vectors.shape)
       prototype_vectors = torch.tensor(prototype_vectors, dtype=torch.float32)
       prototype_vectors = prototype_vectors.unsqueeze(0)
       
       return prototype_vectors

if __name__ == '__main__':
       processed_train_dataset = load_from_disk('/root/autodl-tmp/vqa/VQA-with-XProNet/saved_data/train')
       # 调用函数
       prototype_vectors = calculate_prototype_vectors(processed_train_dataset)
       
       # 保存原型向量
       torch.save(prototype_vectors, '/root/autodl-tmp/vqa/VQA-with-XProNet/saved_data/init_prototype/prototype_vectors.pt')
