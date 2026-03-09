import os
import torch
import json
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


class Word2VecTester:
    """Word2Vec模型测试和可视化工具类"""

    def __init__(self, model_dir, device="cuda"):
        """
        初始化测试器

        参数:
            model_dir: 模型保存目录（包含 model.pt 和 vocab.json）
            device: 设备
        """
        self.model_dir = model_dir
        self.device = device
        self.embed_v = None
        self.embed_u = None
        self.token_to_idx = None
        self.idx_to_token = None
        self.training_info = None

        self._load_model()

    def _load_model(self):
        """从本地加载模型"""
        # 加载词典
        vocab_path = os.path.join(self.model_dir, "vocab.json")
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab_data = json.load(f)

        self.token_to_idx = vocab_data["token_to_idx"]
        self.idx_to_token = vocab_data["idx_to_token"]

        # 加载训练信息
        info_path = os.path.join(self.model_dir, "training_info.json")
        if os.path.exists(info_path):
            with open(info_path, "r") as f:
                self.training_info = json.load(f)

        # 加载模型权重
        model_path = os.path.join(self.model_dir, "model.pt")
        checkpoint = torch.load(model_path, map_location=self.device)

        # 重新创建模型
        vocab_size = checkpoint["vocab_size"]
        embed_size = checkpoint["embed_size"]

        self.embed_v = nn.Embedding(vocab_size, embed_size)
        self.embed_u = nn.Embedding(vocab_size, embed_size)

        self.embed_v.load_state_dict(checkpoint["embed_v_state_dict"])
        self.embed_u.load_state_dict(checkpoint["embed_u_state_dict"])

        self.embed_v.to(self.device)
        self.embed_u.to(self.device)

    def get_similar_tokens(self, query_token, k=10):
        """
        获取与查询词最相似的k个词，并可视化

        参数:
            query_token: 查询词
            k: 返回的相似词数量
            plot: 是否绘制可视化图表
            figsize: 图表大小

        返回:
            list: 相似词列表，每个元素包含词和相似度
        """
        if query_token not in self.token_to_idx:
            print(f"错误: 词 '{query_token}' 不在词典中")
            return None

        self.embed_v.to(self.device)
        W = self.embed_v.weight.data

        x = W[self.token_to_idx[query_token]]
        cos = torch.matmul(W, x) / (torch.norm(W, dim=1) * torch.norm(x) + 1e-9)

        topk = torch.topk(cos, k=k + 1).indices.tolist()

        results = []
        similarities = []
        words = []

        for i in topk[1:]:  # 跳过查询词本身
            word = self.idx_to_token[i]
            similarity = cos[i].item()
            results.append({"word": word, "similarity": similarity})
            words.append(word)
            similarities.append(similarity)

        return results

    def get_embed_vectors(self, words):
        """
        获取指定词语的嵌入向量

        参数:
            words: 词语列表

        返回:
            dict: 词语到向量的映射
        """
        self.embed_v.to(self.device)
        W = self.embed_v.weight.data

        vectors = {}
        for word in words:
            if word in self.token_to_idx:
                vectors[word] = W[self.token_to_idx[word]].cpu().numpy()
            else:
                print(f"Error: {word} not in the vector list")

        return vectors

    def visualize_embeddings(
        self,
        words=None,
        method="tsne",
        perplexity=30,
        n_iter=1000,
        figsize=(12, 10),
        output_dir=None,
    ):
        """
        使用 t-SNE 或 PCA 可视化词向量

        参数:
            words: 要可视化的词语列表，None则随机选择100个词
            method: 降维方法 ('tsne' 或 'pca')
            perplexity: t-SNE的perplexity参数
            n_iter: t-SNE的迭代次数
            figsize: 图表大小
            output_dir: 输出目录，None则使用模型目录
        """

        self.embed_v.to(self.device)
        W = self.embed_v.weight.data.cpu().numpy()

        # 选择要可视化的词
        if words is None:
            # 随机选择100个词
            indices = np.random.choice(
                len(self.idx_to_token), min(100, len(self.idx_to_token)), replace=False
            )
            words = [self.idx_to_token[i] for i in indices]

        # 获取词向量
        word_indices = [self.token_to_idx[w] for w in words if w in self.token_to_idx]
        vectors = W[word_indices]

        # 降维
        if method.lower() == "tsne":
            reducer = TSNE(
                n_components=2, perplexity=perplexity, max_iter=n_iter, random_state=42
            )
        else:  # PCA
            reducer = PCA(n_components=2, random_state=42)

        vectors_2d = reducer.fit_transform(vectors)

        # 可视化
        plt.figure(figsize=figsize)

        # 绘制散点图
        scatter = plt.scatter(
            vectors_2d[:, 0],
            vectors_2d[:, 1],
            c=range(len(vectors_2d)),
            cmap="viridis",
            s=100,
            alpha=0.6,
            edgecolors="black",
            linewidth=0.5,
        )

        # 添加标签
        for i, word in enumerate(words):
            if word in self.token_to_idx:
                idx = word_indices.index(self.token_to_idx[word])
                plt.annotate(
                    word,
                    (vectors_2d[idx, 0], vectors_2d[idx, 1]),
                    fontsize=9,
                    alpha=0.8,
                    xytext=(5, 5),
                    textcoords="offset points",
                )

        plt.colorbar(scatter, label="Word Index")
        plt.xlabel(f"{method.upper()} Component 1", fontsize=12)
        plt.ylabel(f"{method.upper()} Component 2", fontsize=12)
        plt.title(
            f"Word Embeddings Visualization ({method.upper()})",
            fontsize=14,
            fontweight="bold",
        )
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # 确定输出目录
        if output_dir is None:
            output_dir = self.model_dir
        else:
            os.makedirs(output_dir, exist_ok=True)

        # 创建可视化子目录
        viz_dir = os.path.join(output_dir, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)

        # 保存图片，文件名包含方法标识
        filename = os.path.join(viz_dir, f"{method}_embedding_visualize.pdf")
        plt.savefig(filename)
        plt.close()

    def word_analogy(self, word1, word2, word3, k=5):
        """
        词类比任务: word1 - word2 + word3 = ?

        参数:
            word1, word2, word3: 类比词语
            k: 返回的结果数量

        返回:
            list: 类比结果列表
        """
        self.embed_v.to(self.device)
        W = self.embed_v.weight.data

        # 检查词语是否存在
        for word in [word1, word2, word3]:
            if word not in self.token_to_idx:
                print(f"错误: 词 '{word}' 不在词典中")
                return None

        # 计算类比向量
        vec1 = W[self.token_to_idx[word1]]
        vec2 = W[self.token_to_idx[word2]]
        vec3 = W[self.token_to_idx[word3]]

        target_vec = vec1 - vec2 + vec3

        # 计算与所有词的相似度
        cos = torch.matmul(W, target_vec) / (
            torch.norm(W, dim=1) * torch.norm(target_vec) + 1e-9
        )

        # 排除输入词
        exclude_indices = {self.token_to_idx[w] for w in [word1, word2, word3]}

        # 找出最相似的词
        sorted_indices = torch.argsort(cos, descending=True)

        results = []
        count = 0
        for idx in sorted_indices:
            if idx.item() not in exclude_indices:
                word = self.idx_to_token[idx.item()]
                similarity = cos[idx].item()
                results.append({"word": word, "similarity": similarity})
                count += 1
                if count >= k:
                    break

        return results

    def compute_similarity(self, word1, word2):
        """
        计算两个词之间的余弦相似度

        参数:
            word1, word2: 两个词

        返回:
            float: 余弦相似度
        """
        self.embed_v.to(self.device)
        W = self.embed_v.weight.data

        if word1 not in self.token_to_idx or word2 not in self.token_to_idx:
            print(f"错误: 其中一个词不在词典中")
            return None

        vec1 = W[self.token_to_idx[word1]]
        vec2 = W[self.token_to_idx[word2]]

        similarity = torch.matmul(vec1, vec2) / (
            torch.norm(vec1) * torch.norm(vec2) + 1e-9
        )

        return similarity.item()
