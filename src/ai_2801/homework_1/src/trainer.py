import os
import json
import math
import time
import torch
import torch.nn as nn
import torch.optim as optim
import random
import collections
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
from torch.utils.data import DataLoader


class Word2VecTrainer:
    """Word2Vec训练器，支持批量调参和评估"""

    def __init__(
        self,
        data_path="data/ptb.train.txt",
        min_freq=5,
        max_window_size=5,
        K=5,
        batch_size=512,
    ):
        self.data_path = data_path
        self.min_freq = min_freq
        self.max_window_size = max_window_size
        self.K = K
        self.batch_size = batch_size

        # 数据相关
        self.raw_dataset = None
        self.counter = None
        self.idx_to_token = None
        self.token_to_idx = None
        self.dataset = None
        self.subsampled_dataset = None
        self.all_centers = None
        self.all_contexts = None
        self.all_negatives = None
        self.data_iter = None
        self.num_tokens = 0
        
        self.load_and_preprocess_data()

    def load_and_preprocess_data(self):
        """加载和预处理数据"""
        print("Loading and preprocessing data...")

        # 1. 读取数据
        with open(self.data_path, "r") as f:
            lines = f.readlines()
            self.raw_dataset = [st.split() for st in lines]

        # 2. 构建词典，过滤低频词
        self.counter = collections.Counter([tk for st in self.raw_dataset for tk in st])
        self.counter = dict(
            filter(lambda x: x[1] >= self.min_freq, self.counter.items())
        )

        # 3. 词到索引的映射
        self.idx_to_token = [tk for tk, _ in self.counter.items()]
        self.token_to_idx = {tk: idx for idx, tk in enumerate(self.idx_to_token)}
        self.dataset = [
            [self.token_to_idx[tk] for tk in st if tk in self.token_to_idx]
            for st in self.raw_dataset
        ]
        self.num_tokens = sum([len(st) for st in self.dataset])

        print(f"Vocabulary size: {len(self.idx_to_token)}")
        print(f"Total tokens: {self.num_tokens}")

        # 4. 二次采样
        self.subsampled_dataset = self._subsample()

        # 5. 提取中心词和上下文词
        self.all_centers, self.all_contexts = self._get_centers_and_contexts()
        print(f"Total center-context pairs: {len(self.all_centers)}")

        # 6. 负采样
        self.all_negatives = self._get_negatives()

        # 7. 构建数据加载器
        self._build_data_loader()

        print("Data preprocessing completed!\n")

    def _subsample(self):
        """二次采样"""

        def discard(idx):
            return random.uniform(0, 1) < 1 - math.sqrt(
                1e-4 / self.counter[self.idx_to_token[idx]] * self.num_tokens
            )

        return [[tk for tk in st if not discard(tk)] for st in self.dataset]

    def _get_centers_and_contexts(self):
        """提取中心词和上下文词"""
        centers, contexts = [], []
        for st in self.subsampled_dataset:
            if len(st) < 2:
                continue
            centers += st
            for center_i in range(len(st)):
                window_size = random.randint(1, self.max_window_size)
                indices = list(
                    range(
                        max(0, center_i - window_size),
                        min(len(st), center_i + 1 + window_size),
                    )
                )
                indices.remove(center_i)
                contexts.append([st[idx] for idx in indices])
        return centers, contexts

    def _get_negatives(self):
        """负采样"""
        sampling_weights = [self.counter[w] ** 0.75 for w in self.idx_to_token]
        all_negatives, neg_candidates, i = [], [], 0
        population = list(range(len(sampling_weights)))

        for contexts in self.all_contexts:
            negatives = []
            while len(negatives) < len(contexts) * self.K:
                if i == len(neg_candidates):
                    neg_candidates = random.choices(
                        population, weights=sampling_weights, k=1000
                    )
                    i = 0
                neg = neg_candidates[i]
                i += 1
                if neg not in contexts:
                    negatives.append(neg)
            all_negatives.append(negatives)
        return all_negatives

    def _batchify(self, data):
        """批次整理"""
        max_len = max(len(c) + len(n) for _, c, n in data)

        centers = []
        contexts_negatives = []
        masks = []
        labels = []

        for center, context, negative in data:
            centers.append(center)
            contexts_negatives.append(
                context + negative + [0] * (max_len - len(context) - len(negative))
            )
            masks.append(
                [1] * (len(context) + len(negative))
                + [0] * (max_len - len(context) - len(negative))
            )
            labels.append(
                [1] * len(context)
                + [0] * (len(negative) + max_len - len(context) - len(negative))
            )

        return (
            torch.tensor(centers).view(-1, 1),
            torch.tensor(contexts_negatives),
            torch.tensor(masks, dtype=torch.float),
            torch.tensor(labels, dtype=torch.float),
        )

    def _build_data_loader(self):
        """构建数据加载器"""
        dataset = list(zip(self.all_centers, self.all_contexts, self.all_negatives))
        self.data_iter = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True, collate_fn=self._batchify
        )

    def skip_gram(self, center, contexts_and_negatives, embed_v, embed_u):
        """Skip-gram模型"""
        v = embed_v(center)
        u = embed_u(contexts_and_negatives)
        pred = torch.bmm(v, u.permute(0, 2, 1))
        return pred

    def sigmoid_binary_cross_entropy(self, pred, label, mask):
        """BCE损失函数"""
        pred = pred.squeeze(1)
        p = torch.sigmoid(pred)
        loss = -(label * torch.log(p + 1e-10) + (1 - label) * torch.log(1 - p + 1e-10))
        loss = (loss * mask).sum() / mask.sum()
        return loss

    def train(self, embed_size, lr, num_epochs, device="cuda", save_freq=None):
        """
        训练模型

        参数:
            embed_size: 词向量维度
            lr: 学习率
            num_epochs: 训练轮数
            device: 设备
            save_freq: 保存checkpoint的频率（epoch数），None表示只在最后保存

        返回:
            dict: 包含训练历史和模型信息的字典
        """
        print(
            f"Training with embed_size={embed_size}, lr={lr}, num_epochs={num_epochs}"
        )
        if save_freq:
            print(f"Checkpoints will be saved every {save_freq} epochs")

        # 初始化模型
        embed_v = nn.Embedding(len(self.idx_to_token), embed_size)
        embed_u = nn.Embedding(len(self.idx_to_token), embed_size)

        embed_v.to(device)
        embed_u.to(device)
        optimizer = optim.Adam(
            list(embed_v.parameters()) + list(embed_u.parameters()), lr=lr
        )

        # 训练历史
        loss_history = []
        time_history = []

        for epoch in trange(num_epochs):
            l_sum, n = 0.0, 0
            start = time.time()

            for batch in self.data_iter:
                centers, contexts_negatives, mask, labels = [
                    x.to(device) for x in batch
                ]
                labels = labels.float()

                # 前向传播
                pred = self.skip_gram(centers, contexts_negatives, embed_v, embed_u)

                # 计算损失
                l = self.sigmoid_binary_cross_entropy(pred, labels, mask)

                # 反向传播
                optimizer.zero_grad()
                l.backward()
                optimizer.step()

                l_sum += l.item() * mask.size(0)
                n += mask.size(0)

            # 记录训练信息
            avg_loss = l_sum / n
            epoch_time = time.time() - start
            loss_history.append(avg_loss)
            time_history.append(epoch_time)

            print(
                f"Epoch {epoch+1}/{num_epochs}, loss={avg_loss:.4f}, time={epoch_time:.2f}s"
            )

            # 定期保存checkpoint（每次保存为独立的实验文件夹）
            if save_freq and (epoch + 1) % save_freq == 0:
                self._save_model(
                    embed_v,
                    embed_u,
                    embed_size,
                    lr,
                    epoch + 1,
                    loss_history,
                    time_history,
                )

        print(f"Training completed! Final loss: {loss_history[-1]:.4f}\n")

        # 保存最终模型
        save_dir = self._save_model(
            embed_v, embed_u, embed_size, lr, num_epochs, loss_history, time_history
        )

        return {
            "embed_v": embed_v,
            "embed_u": embed_u,
            "loss_history": loss_history,
            "time_history": time_history,
            "final_loss": loss_history[-1],
            "save_dir": save_dir,
            "params": {"embed_size": embed_size, "lr": lr, "num_epochs": num_epochs},
        }

    def _save_model(
        self,
        embed_v,
        embed_u,
        embed_size,
        lr,
        num_epochs,
        loss_history,
        time_history,
        base_dir="checkpoints",
    ):
        """
        保存模型权重和相关信息

        参数:
            embed_v, embed_u: 嵌入层
            embed_size, lr, num_epochs: 超参数
            loss_history, time_history: 训练历史
            base_dir: 基础保存目录

        返回:
            save_dir: 保存的目录路径
        """
        # 创建保存目录：base_dir/embed_size_lr_num_epochs
        save_dir = os.path.join(base_dir, f"embed{embed_size}_lr{lr}_ep{num_epochs}")
        os.makedirs(save_dir, exist_ok=True)

        # 保存模型权重
        torch.save(
            {
                "embed_v_state_dict": embed_v.state_dict(),
                "embed_u_state_dict": embed_u.state_dict(),
                "vocab_size": len(self.idx_to_token),
                "embed_size": embed_size,
            },
            os.path.join(save_dir, "model.pt"),
        )

        # 保存训练信息
        training_info = {
            "embed_size": embed_size,
            "lr": lr,
            "num_epochs": num_epochs,
            "loss_history": loss_history,
            "time_history": time_history,
            "avg_time_per_epoch": np.mean(time_history) if time_history else None,
            "final_loss": loss_history[-1] if loss_history else None,
        }

        with open(os.path.join(save_dir, "training_info.json"), "w") as f:
            json.dump(training_info, f, indent=2)

        # 保存词典
        vocab_data = {
            "token_to_idx": self.token_to_idx,
            "idx_to_token": self.idx_to_token,
        }

        with open(os.path.join(save_dir, "vocab.json"), "w", encoding="utf-8") as f:
            json.dump(vocab_data, f, ensure_ascii=False, indent=2)

        # 绘制并保存训练曲线
        self.plot_training_curves(save_dir, loss_history, time_history)

        print(f"Model saved to {save_dir}/")

        return save_dir

    def plot_training_curves(self, save_dir, loss_history, time_history):
        """
        绘制训练 loss 曲线并保存为PDF

        参数:
            save_dir: 保存目录
            loss_history: loss历史列表
            time_history: 时间历史列表
        """
        pdf_path = os.path.join(save_dir, "training_curves.pdf")

        # 创建单页图表
        fig, ax = plt.subplots(figsize=(10, 6))

        # 绘制 loss 曲线
        ax.plot(
            range(1, len(loss_history) + 1),
            loss_history,
            "b-o",
            linewidth=2,
            markersize=6,
            label="Training Loss"
        )

        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Loss", fontsize=12)
        ax.set_title("Training Loss Curve", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.set_xticks(range(1, len(loss_history) + 1))

        # 添加数值标签
        for i, loss in enumerate(loss_history):
            ax.text(
                i + 1,
                loss,
                f"{loss:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
                alpha=0.7
            )

        plt.tight_layout()

        # 保存为 PDF
        fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
        plt.close()

        print(f"Training curves saved to {pdf_path}")

