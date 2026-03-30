"""
自行实现的 fastText 模型训练代码
使用 PyTorch 实现 fastText 的核心功能
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import re
import os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from collections import Counter
import multiprocessing
from torch.utils.tensorboard import SummaryWriter
import time


# ==================== FastText 模型核心 ====================


class _FastTextModel(nn.Module):
    """
    FastText模型的核心神经网络结构。

    这个类定义了EmbeddingBag层用于计算词和n-gram的平均向量，
    以及两个线性层用于分类。
    """

    def __init__(
        self,
        vocab_size: int,
        num_buckets: int,
        hidden_size: int,
        embedding_dim: int,
        num_classes: int,
    ):
        """
        初始化FastText模型。

        Args:
            vocab_size (int): 词汇表的大小 (不包括n-gram)。
            num_buckets (int): 用于哈希n-gram的桶的数量。
            hidden_size (int): 隐藏层的大小 (fastText原版中没有，此处为扩展)。
            embedding_dim (int): 词向量的维度。
            num_classes (int): 分类的类别总数。
        """
        super(_FastTextModel, self).__init__()
        # 总的嵌入表大小等于词汇表大小加上哈希桶的数量
        total_embeddings = vocab_size + num_buckets
        self.embeddings = nn.EmbeddingBag(
            num_embeddings=total_embeddings,
            embedding_dim=embedding_dim,
            mode="mean",  # 使用平均值来聚合词和n-gram的向量
        )
        # 扩展的结构，原版fasttext直接从embedding连接到输出层
        self.A = nn.Linear(embedding_dim, hidden_size)
        self.B = nn.Linear(hidden_size, num_classes)

    def forward(self, indices, offsets):
        """
        定义模型的前向传播逻辑。

        Args:
            indices (torch.Tensor): 包含一个批次中所有文本的词和n-gram索引的扁平化张量。
            offsets (torch.Tensor): 一个张量，指示`indices`中每个序列的起始位置。

        Returns:
            torch.Tensor: 每个输入文本的分类logits。
        """
        # 使用EmbeddingBag获取平均嵌入向量
        embedded = self.embeddings(indices, offsets)
        hidden = self.A(embedded)
        # * use relu activation functions
        hidden = torch.relu(hidden)
        logits = self.B(hidden)
        return logits


# ==================== 数据集处理 ====================


class _PreprocessedDataset(Dataset):
    """
    一个简单的PyTorch数据集类，用于加载预处理好的数据。
    """

    def __init__(self, file_path):
        """
        初始化数据集。

        Args:
            file_path (str): .pt格式的预处理数据文件路径。
        """
        self.data = torch.load(file_path)

    def __len__(self):
        """返回数据集中的样本数量。"""
        return len(self.data)

    def __getitem__(self, idx):
        """根据索引获取一个样本。"""
        return self.data[idx]


# ==================== 模型封装和数据处理 ====================


class _ModelWrapper:
    """
    一个封装了模型、数据处理、训练和预测逻辑的辅助类。
    """

    def __init__(self, config, device):
        """
        初始化模型包装器。

        Args:
            config (dict): 包含所有超参数和设置的字典。
            device (torch.device): 运行模型的设备 (CPU或GPU)。
        """
        self.config = config
        self.device = device
        self.vocab = None
        self.word_to_ix = None
        self.label_to_ix, self.ix_to_label = {}, {}

        # 如果是训练模式，则构建标签映射
        if "input" in config:
            self.label_to_ix, self.ix_to_label = self._get_label_info(config["input"])
            self.config["num_classes"] = len(self.label_to_ix)

        # 模型将在词汇表构建后被初始化
        self.model = None

    def _get_label_info(self, file_path):
        """
        从文件中扫描并提取所有唯一的标签，并创建映射。

        Args:
            file_path (str): 训练数据文件路径。

        Returns:
            tuple: (label_to_ix, ix_to_label) 两个字典，用于标签和索引的相互转换。
        """
        labels = set()
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                match = re.search(r"__label__\S+", line)
                if match:
                    labels.add(match.group(0))
        sorted_labels = sorted(list(labels))
        label_to_ix = {label: i for i, label in enumerate(sorted_labels)}
        ix_to_label = {i: label for label, i in label_to_ix.items()}
        return label_to_ix, ix_to_label

    def _build_vocab(self, file_path):
        """
        根据minCount参数扫描文件以构建词汇表。

        Args:
            file_path (str): 训练数据文件路径。
        """
        word_counts = Counter()
        print(f"Building vocabulary from {file_path}...")
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                text = re.sub(r"__label__\S+\s", "", line)
                word_counts.update(text.lower().split())

        # 根据minCount过滤词汇
        self.vocab = [
            word
            for word, count in word_counts.items()
            if count >= self.config["minCount"]
        ]
        # 为未知词添加一个特殊标记
        self.word_to_ix = {word: i for i, word in enumerate(self.vocab)}
        self.word_to_ix["<UNK>"] = len(self.vocab)
        self.num_classes = len(self.label_to_ix)
        print(
            f"Vocabulary size: {len(self.vocab)} (words with count >= {self.config['minCount']})"
        )

    def _generate_ngrams(self, word: str) -> list:
        """
        为一个单词生成字符级别的n-gram。

        Args:
            word (str): 输入的单词。

        Returns:
            list: 生成的n-gram列表。
        """
        minn = self.config["minn"]
        maxn = self.config["maxn"]
        if minn == 0 or maxn == 0:
            return []

        extended_word = "<" + word + ">"
        ngrams = []
        for n in range(minn, maxn + 1):
            if len(extended_word) >= n:
                ngrams.extend(
                    [
                        extended_word[i : i + n]
                        for i in range(len(extended_word) - n + 1)
                    ]
                )
        return ngrams

    def _get_indices_for_text(self, text: str):
        """
        将一行文本转换为其对应的索引列表 (包括词索引和n-gram哈希索引)。

        Args:
            text (str): 输入的文本行。

        Returns:
            list: 包含所有词和n-gram索引的列表。
        """
        indices = []
        tokens = text.lower().split()

        # 词索引
        vocab_size = len(self.vocab)
        unk_idx = self.word_to_ix["<UNK>"]
        word_indices = [self.word_to_ix.get(token, unk_idx) for token in tokens]
        indices.extend(word_indices)

        # 字符n-gram索引 (通过哈希)
        # N-grams被哈希到从 vocab_size 开始的桶中
        for token in tokens:
            ngrams = self._generate_ngrams(token)
            for ngram in ngrams:
                # 使用字符串哈希函数将n-gram映射到桶中
                # 使用 abs() 确保哈希值始终为正数，避免负数索引越界
                hash_value = abs(hash(ngram)) % self.config["bucket"]
                # 将哈希值偏移vocab_size，确保不与词索引冲突
                indices.append(vocab_size + hash_value)
        return indices

    def _preprocess_file(self, raw_path, processed_path):
        """
        预处理原始文本文件，并将其保存为torch张量格式，以加快训练速度。

        Args:
            raw_path (str): 原始文本文件路径。
            processed_path (str): .pt格式的输出文件路径。
        """
        processed_data = []
        with open(raw_path, "r", encoding="utf-8") as f:
            # 为tqdm获取总行数
            num_lines = sum(1 for _ in f)
            f.seek(0)  # 将文件指针重置到开头

            # 直接迭代文件对象以节省内存
            for line in tqdm(
                f, total=num_lines, desc=f"Processing {os.path.basename(raw_path)}"
            ):
                line = line.strip()
                if not line:
                    continue
                label_match = re.search(r"__label__\S+", line)
                if not label_match:
                    continue
                label = label_match.group(0)
                text = re.sub(r"__label__\S+\s", "", line)

                indices = self._get_indices_for_text(text)
                processed_data.append(
                    (torch.LongTensor(indices), self.label_to_ix[label])
                )

        torch.save(processed_data, processed_path)
        print(f"Pre-processed data saved to {processed_path}")

    def _create_preprocessed_collate_fn(self):
        """
        创建一个用于DataLoader的collate_fn。

        该函数负责将一批可变长度的样本打包成一个批次，
        生成用于EmbeddingBag的indices和offsets。

        Returns:
            function: a collate_fn function.
        """

        def collate_fn(batch):
            indices_list, labels_list = zip(*batch)
            labels_tensor = torch.LongTensor(labels_list)
            offsets = [0] + [len(i) for i in indices_list]
            offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
            indices_tensor = torch.cat(indices_list)
            return indices_tensor, offsets, labels_tensor

        return collate_fn

    def save_model(self, path):
        """
        将模型、配置和词汇表保存到文件。

        Args:
            path (str): 模型保存路径。
        """
        torch.save(
            {
                "config": self.config,
                "label_to_ix": self.label_to_ix,
                "vocab": self.vocab,
                "word_to_ix": self.word_to_ix,
                "model_state_dict": self.model.state_dict(),
            },
            path,
        )

    def _validate(self, val_dataloader, device):
        """
        在验证集上评估模型。

        Args:
            val_dataloader: 验证数据的 DataLoader。
            device: 运行设备。

        Returns:
            tuple: (平均损失, 准确率)。
        """
        self.model.eval()
        val_loss = 0.0
        all_true, all_preds = [], []
        loss_function = nn.CrossEntropyLoss()

        with torch.no_grad():
            for indices, offsets, targets in val_dataloader:
                indices, offsets, targets = (
                    indices.to(device),
                    offsets.to(device),
                    targets.to(device),
                )
                outputs = self.model(indices, offsets)
                loss = loss_function(outputs, targets)
                val_loss += loss.item()

                predictions = torch.argmax(outputs, dim=1)
                all_true.extend(targets.cpu().numpy())
                all_preds.extend(predictions.cpu().numpy())

        self.model.train()

        avg_val_loss = val_loss / len(val_dataloader)
        n_examples = len(all_true)
        correct = sum(1 for t, p in zip(all_true, all_preds) if t == p)
        accuracy = correct / n_examples if n_examples > 0 else 0.0

        return avg_val_loss, accuracy

    def test(self, path):
        """
        在给定的测试文件上评估模型。

        Args:
            path (str): 测试文件路径。

        Returns:
            tuple: (样本总数, 准确率, 准确率) - 返回两次准确率以模仿fasttext库的P@1和R@1。
        """
        processed_path = path + ".pt"
        if not os.path.exists(processed_path):
            print(f"Pre-processing test file: {path}")
            self._preprocess_file(path, processed_path)
        test_dataset = _PreprocessedDataset(processed_path)
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=64,
            collate_fn=self._create_preprocessed_collate_fn(),
            num_workers=self.config["thread"],
        )
        self.model.eval()
        all_true, all_preds = [], []
        with torch.no_grad():
            for indices, offsets, targets in test_dataloader:
                indices, offsets, targets = (
                    indices.to(self.device),
                    offsets.to(self.device),
                    targets.to(self.device),
                )
                outputs = self.model(indices, offsets)
                predictions = torch.argmax(outputs, dim=1)
                all_true.extend(targets.cpu().numpy())
                all_preds.extend(predictions.cpu().numpy())

        n_examples = len(all_true)
        correct = sum(1 for t, p in zip(all_true, all_preds) if t == p)
        precision = correct / n_examples if n_examples > 0 else 0.0
        return n_examples, precision, precision

    def predict(self, texts, k=1):
        """
        对给定的文本进行预测。

        Args:
            texts (str or list): 一个或多个待预测的文本。
            k (int): 返回top-k个最可能的标签。

        Returns:
            tuple: (labels, probabilities) 预测的标签和对应的概率。
        """
        if isinstance(texts, str):
            texts = [texts]
        self.model.eval()

        indices_list = [
            torch.LongTensor(self._get_indices_for_text(text)) for text in texts
        ]

        dummy_labels = [0] * len(indices_list)
        collate_fn = self._create_preprocessed_collate_fn()
        indices_tensor, offsets, _ = collate_fn(list(zip(indices_list, dummy_labels)))
        indices_tensor, offsets = indices_tensor.to(self.device), offsets.to(
            self.device
        )

        with torch.no_grad():
            outputs = self.model(indices_tensor, offsets)
            probs = torch.softmax(outputs, dim=1)
            topk_probs, topk_indices = torch.topk(probs, k, dim=1)

        labels = [[self.ix_to_label[idx.item()] for idx in row] for row in topk_indices]
        return labels, topk_probs.cpu().numpy()


# ==================== 画图函数 ====================


def _plot_loss(loss_history, save_path="training_loss.png"):
    """
    绘制并保存训练损失曲线图。

    Args:
        loss_history (list): 包含每个训练步骤损失值的列表。
        save_path (str): 图像保存路径。
    """
    plt.figure(figsize=(12, 6))
    plt.plot(loss_history, label="Training Loss")
    plt.title("Training Loss Curve")
    plt.xlabel("Training Steps")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.show()
    plt.savefig(save_path)
    print(f"Training loss curve saved to '{save_path}'")


# ==================== 主接口类 ====================


class fasttext:
    """
    一个模拟官方fasttext库API的主接口类。
    所有方法都定义为静态方法，以便于直接调用。
    """

    @staticmethod
    def train_supervised(
        input,
        lr=0.1,
        dim=100,
        ws=5,
        epoch=5,
        minCount=1,
        minCountLabel=0,
        minn=0,
        maxn=0,
        neg=5,
        wordNgrams=1,
        loss="softmax",
        bucket=2000000,
        thread=1,
        lrUpdateRate=100,
        t=0.0001,
        label="__label__",
        verbose=2,
        pretrainedVectors="",
        batch_size=32,
        validation_split=0.1,
        tensorboard_log_dir="runs/fasttext_experiment",
    ):
        """
        训练一个有监督的分类模型。

        Args:
            input (str): 训练文件路径 (必需)。
            lr (float): 学习率。
            dim (int): 词向量维度。
            ws (int): 上下文窗口大小 (在此脚本中未使用，为保持API一致性)。
            epoch (int): 训练轮次。
            minCount (int): 词汇的最低词频。
            minCountLabel (int): 标签的最低词频 (未使用)。
            minn (int): 最小字符n-gram长度。0表示禁用。
            maxn (int): 最大字符n-gram长度。0表示禁用。
            neg (int): 负采样数量 (未使用)。
            wordNgrams (int): 词n-gram最大长度 (未使用)。
            loss (str): 损失函数。目前只支持 'softmax'。
            bucket (int): n-gram哈希桶的数量。
            thread (int): 线程数。默认为CPU核心数。
            lrUpdateRate (int): 学习率更新频率 (未使用)。
            t (float): 采样阈值 (未使用)。
            label (str): 标签前缀。
            verbose (int): 日志详细程度 (未使用)。
            pretrainedVectors (str): 预训练词向量文件路径 (未使用)。
            batch_size (int): 训练时的批处理大小。
            validation_split (float): 验证集比例，默认0.1（10%）。
            tensorboard_log_dir (str): TensorBoard日志目录。

        Returns:
            _ModelWrapper: 一个训练好的模型包装器实例。
        """
        if thread is None:
            thread = max(1, os.cpu_count())

        if loss != "softmax":
            raise NotImplementedError(
                f"Loss function '{loss}' is not implemented. "
                f"Only 'softmax' is supported in this script."
            )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        args_dict = locals()
        model_wrapper = _ModelWrapper(args_dict, device)

        model_wrapper._build_vocab(input)

        # 初始化模型
        model_wrapper.model = _FastTextModel(
            vocab_size=len(model_wrapper.word_to_ix),
            num_buckets=bucket,
            embedding_dim=dim,
            hidden_size=4 * dim,
            num_classes=model_wrapper.num_classes,
        ).to(device)

        # 预处理数据
        processed_path = input + ".pt"
        if os.path.exists(processed_path):
            os.remove(processed_path)  # 如果参数更改，强制重新处理
        model_wrapper._preprocess_file(input, processed_path)

        # 加载数据集并分割为训练集和验证集
        full_dataset = _PreprocessedDataset(processed_path)
        total_size = len(full_dataset)
        val_size = int(total_size * validation_split)
        train_size = total_size - val_size

        train_dataset, val_dataset = random_split(
            full_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42),
        )

        print(f"Dataset split: Train={train_size}, Val={val_size}")

        # 创建 DataLoader
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=model_wrapper._create_preprocessed_collate_fn(),
            num_workers=thread,
            pin_memory=True if device.type == "cuda" else False,
        )

        val_dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=model_wrapper._create_preprocessed_collate_fn(),
            num_workers=thread,
            pin_memory=True if device.type == "cuda" else False,
        )

        loss_history = []

        # 定义损失函数和优化器
        loss_function = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model_wrapper.model.parameters(), lr=lr)
        total_steps = epoch * len(train_dataloader)
        lr_lambda = lambda step: 1.0 - step / total_steps
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)  # 线性学习率

        # 初始化 TensorBoard
        log_dir = os.path.join(
            tensorboard_log_dir, time.strftime("%Y%m%d-%H%M%S")
        )
        writer = SummaryWriter(log_dir)
        print(f"TensorBoard logs will be saved to: {log_dir}")
        print(f"View logs with: tensorboard --logdir={tensorboard_log_dir}")

        # 开始训练
        model_wrapper.model.train()
        print(f"Starting training on {device}...")
        global_step = 0
        for epoch_idx in range(epoch):
            progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch_idx+1}/{epoch}")
            epoch_loss = 0.0
            num_batches = 0

            for indices, offsets, targets in progress_bar:
                indices, offsets, targets = (
                    indices.to(device),
                    offsets.to(device),
                    targets.to(device),
                )

                # 训练代码
                optimizer.zero_grad()
                # 前向传播
                outputs = model_wrapper.model(indices, offsets)
                # 计算损失
                loss_val = loss_function(outputs, targets)
                # 反向传播
                loss_val.backward()
                optimizer.step()
                scheduler.step()

                loss_history.append(loss_val.item())
                epoch_loss += loss_val.item()
                num_batches += 1

                current_lr = optimizer.param_groups[0]["lr"]
                progress_bar.set_postfix(
                    {"loss": f"{loss_val.item():.4f}", "lr": f"{current_lr:.6f}"}
                )

                # 记录到 TensorBoard
                writer.add_scalar("Loss/train_step", loss_val.item(), global_step)
                writer.add_scalar("Learning_rate", current_lr, global_step)
                global_step += 1

            # 每个 epoch 结束后在验证集上评估
            avg_train_loss = epoch_loss / num_batches
            val_loss, val_acc = model_wrapper._validate(val_dataloader, device)

            # 记录 epoch 级别的指标
            writer.add_scalar("Loss/train_epoch", avg_train_loss, epoch_idx)
            writer.add_scalar("Loss/val_epoch", val_loss, epoch_idx)
            writer.add_scalar("Accuracy/val", val_acc, epoch_idx)

            print(
                f"Epoch {epoch_idx+1}/{epoch} - "
                f"Train Loss: {avg_train_loss:.4f}, "
                f"Val Loss: {val_loss:.4f}, "
                f"Val Acc: {val_acc:.4f}"
            )

        writer.close()

        _plot_loss(loss_history)
        print("Training complete.")
        return model_wrapper

    @staticmethod
    def load_model(path):
        """
        从文件加载一个预训练好的模型。

        Args:
            path (str): 模型文件路径。

        Returns:
            _ModelWrapper: 一个加载好的模型包装器实例。
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(path, map_location=device)
        config = checkpoint["config"]

        model_wrapper = _ModelWrapper(config, device)
        model_wrapper.label_to_ix = checkpoint["label_to_ix"]
        model_wrapper.ix_to_label = {i: l for l, i in model_wrapper.label_to_ix.items()}
        model_wrapper.vocab = checkpoint["vocab"]
        model_wrapper.word_to_ix = checkpoint["word_to_ix"]

        model_wrapper.config["num_classes"] = len(model_wrapper.label_to_ix)
        model_wrapper.model = _FastTextModel(
            vocab_size=len(model_wrapper.word_to_ix),
            num_buckets=config["bucket"],
            embedding_dim=config["dim"],
            hidden_size=4 * config["dim"],  # 保持与训练时一致
            num_classes=config["num_classes"],
        ).to(device)
        model_wrapper.model.load_state_dict(checkpoint["model_state_dict"])
        model_wrapper.model.eval()

        print(f"Model loaded from {path}")
        return model_wrapper
