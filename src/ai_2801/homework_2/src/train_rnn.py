"""
RNN 语言模型训练脚本
从 HW2.ipynb 中提取的 RNN 模型和数据处理类
支持 tensorboard 日志记录
"""

import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter
from torch.utils.tensorboard import SummaryWriter
import os
import time


class Dict:
    """用于返回多个值的辅助类"""
    def __init__(self, logits, hidden):
        self.logits = logits
        self.hidden = hidden


class PTBDataset:
    """
    PTB 数据集处理类
    负责构建词表、数据预处理和批次生成
    """

    def __init__(self, file_path, vocab=None):
        """
        初始化数据集

        Args:
            file_path: 数据文件路径
            vocab: 可选的预定义词表，如果为 None 则从文件中构建
        """
        self.file_path = file_path
        if vocab is None:
            self.vocab, self.words = self._build_vocab()
        else:
            self.vocab = vocab
            with open(file_path, "r") as f:
                self.words = f.read().replace("\n", "<eos>").split()

        # 创建反向词表用于解码
        self.vocab_reverse = {i: w for w, i in self.vocab.items()}

    def _build_vocab(self):
        """从文件构建词表"""
        with open(self.file_path, "r") as f:
            words = f.read().replace("\n", "<eos>").split()
        counter = Counter(words)
        vocab = {word: i for i, (word, _) in enumerate(counter.most_common())}
        return vocab, words

    def file_to_ids(self):
        """将文本转换为词表索引"""
        return [self.vocab[w] for w in self.words if w in self.vocab]

    def ids_to_words(self, ids):
        """将词表索引转换回单词"""
        return [self.vocab_reverse[i] for i in ids]

    def batchify(self, data, batch_size, device):
        """
        将一维数据切成 batch_size 份

        Args:
            data: 词表索引列表
            batch_size: 批次大小
            device: 设备

        Returns:
            形状为 (seq_len, batch_size) 的张量
        """
        nbatch = len(data) // batch_size
        data = data[:nbatch * batch_size]
        data = torch.tensor(data, dtype=torch.long, device=device)
        return data.view(batch_size, -1).t().contiguous()


class RNNLM(nn.Module):
    """
    两层 RNN 语言模型

    架构:
        - Embedding 层
        - 第一层 RNN
        - 第二层 RNN
        - 输出层
    """

    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        """
        初始化 RNN 模型

        Args:
            vocab_size: 词表大小
            embedding_dim: 词向量维度
            hidden_dim: 隐藏层维度
        """
        super(RNNLM, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # 第一层 RNN 参数
        self.W_ih1 = nn.Parameter(torch.randn(embedding_dim, hidden_dim) * 0.1)
        self.W_hh1 = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.1)
        self.b1 = nn.Parameter(torch.zeros(hidden_dim))

        # 第二层 RNN 参数
        self.W_ih2 = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.1)
        self.W_hh2 = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.1)
        self.b2 = nn.Parameter(torch.zeros(hidden_dim))

        # 输出层
        self.decoder = nn.Linear(hidden_dim, vocab_size)

        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size

    def init_hidden(self, batch_size):
        """初始化隐藏状态"""
        return torch.zeros(2, batch_size, self.hidden_dim)

    def forward(self, x, hidden):
        """
        前向传播

        Args:
            x: 输入张量，形状 (seq_len, batch)
            hidden: 隐藏状态，形状 (num_layers, batch, hidden_size)

        Returns:
            Dict 对象，包含 logits 和 hidden
        """
        seq_len, batch_size = x.size()
        x_emb = self.embedding(x)  # (seq_len, batch, embedding_dim)

        h1, h2 = hidden[0], hidden[1]
        outputs = []

        # 对每个时间步进行迭代
        for t in range(seq_len):
            # 第一层 RNN
            ih1 = torch.matmul(x_emb[t], self.W_ih1)
            hh1 = torch.matmul(h1, self.W_hh1)
            h1_new = torch.tanh(ih1 + hh1 + self.b1)

            # 第二层 RNN
            ih2 = torch.matmul(h1_new, self.W_ih2)
            hh2 = torch.matmul(h2, self.W_hh2)
            h2_new = torch.tanh(ih2 + hh2 + self.b2)

            # 更新隐藏状态
            h1 = h1_new
            h2 = h2_new

            # 输出层解码
            output_t = self.decoder(h2_new)
            outputs.append(output_t)

        # 将所有时刻的输出堆叠起来
        logits = torch.stack(outputs, dim=0)  # (seq_len, batch, vocab_size)
        hidden = torch.stack([h1, h2], dim=0)  # (2, batch, hidden_dim)

        return Dict(logits, hidden)


def get_batch(source, i, bptt):
    """
    从源数据中获取一个批次

    Args:
        source: 源数据张量
        i: 起始位置
        bptt: 序列长度

    Returns:
        data: 输入数据
        target: 目标数据
    """
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].reshape(-1)
    return data, target


class RNNTrainer:
    """
    RNN 模型训练器
    负责训练循环、日志记录和模型保存
    """

    def __init__(
        self,
        model,
        train_data,
        val_data=None,
        log_dir="./logs",
        model_dir="./checkpoints",
        batch_size=20,
        bptt=50,
        lr=1.0,
        clip_grad=0.25
    ):
        """
        初始化训练器

        Args:
            model: RNN 模型
            train_data: 训练数据
            val_data: 验证数据（可选）
            log_dir: tensorboard 日志目录
            model_dir: 模型保存目录
            batch_size: 批次大小
            bptt: BPTT 截断长度
            lr: 学习率
            clip_grad: 梯度裁剪阈值
        """
        self.model = model
        self.train_data = train_data
        self.val_data = val_data
        self.batch_size = batch_size
        self.bptt = bptt
        self.clip_grad = clip_grad
        self.model_dir = model_dir

        self.device = train_data.device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=lr)

        # 创建目录
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)

        # Tensorboard writer
        self.writer = SummaryWriter(log_dir)

    def train_epoch(self, epoch):
        """
        训练一个 epoch

        Args:
            epoch: 当前 epoch 编号

        Returns:
            avg_loss: 平均损失
        """
        self.model.train()
        total_loss = []
        start_time = time.time()

        hidden = self.model.init_hidden(self.batch_size).to(self.device)

        for batch, i in enumerate(range(0, self.train_data.size(0) - 1, self.bptt)):
            data, targets = get_batch(self.train_data, i, self.bptt)
            data, targets = data.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data, hidden)
            logits, hidden = output.logits, output.hidden

            loss = self.criterion(
                logits.view(-1, logits.size(-1)),
                targets
            )
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
            self.optimizer.step()

            hidden = hidden.detach()
            total_loss.append(loss.item())

            # 记录到 tensorboard
            global_step = (epoch - 1) * len(range(0, self.train_data.size(0) - 1, self.bptt)) + batch
            self.writer.add_scalar('Loss/train_batch', loss.item(), global_step)

            if batch % 100 == 0 and batch > 0:
                cur_loss = sum(total_loss[-100:]) / 100
                elapsed = time.time() - start_time
                print(f'| epoch {epoch:3d} | {batch:5d} batches | '
                      f'loss {cur_loss:5.2f} | time {elapsed:.2f}s')

        avg_loss = sum(total_loss) / len(total_loss)
        return avg_loss

    def evaluate(self, data):
        """
        在给定数据上评估模型

        Args:
            data: 评估数据

        Returns:
            avg_loss: 平均损失
        """
        self.model.eval()
        total_loss = []
        hidden = self.model.init_hidden(self.batch_size).to(self.device)

        with torch.no_grad():
            for i in range(0, data.size(0) - 1, self.bptt):
                data_batch, targets = get_batch(data, i, self.bptt)
                data_batch, targets = data_batch.to(self.device), targets.to(self.device)

                output = self.model(data_batch, hidden)
                logits, hidden = output.logits, output.hidden

                loss = self.criterion(
                    logits.view(-1, logits.size(-1)),
                    targets
                )
                total_loss.append(loss.item())

        avg_loss = sum(total_loss) / len(total_loss)
        return avg_loss

    def train(self, epochs, save_every=5):
        """
        完整训练流程

        Args:
            epochs: 训练轮数
            save_every: 每隔多少 epoch 保存模型
        """
        best_val_loss = None

        for epoch in range(1, epochs + 1):
            epoch_start_time = time.time()

            # 训练
            train_loss = self.train_epoch(epoch)

            # 记录到 tensorboard
            self.writer.add_scalar('Loss/train_epoch', train_loss, epoch)

            # 验证
            if self.val_data is not None:
                val_loss = self.evaluate(self.val_data)
                self.writer.add_scalar('Loss/val', val_loss, epoch)
                print('-' * 89)
                print(f'| end of epoch {epoch:3d} | time: {time.time() - epoch_start_time:.2f}s | '
                      f'train loss {train_loss:5.2f} | val loss {val_loss:5.2f}')
                print('-' * 89)

                # 保存最佳模型
                if best_val_loss is None or val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_model('best_model.pt')
            else:
                print('-' * 89)
                print(f'| end of epoch {epoch:3d} | time: {time.time() - epoch_start_time:.2f}s | '
                      f'train loss {train_loss:5.2f}')
                print('-' * 89)

            # 定期保存模型
            if epoch % save_every == 0:
                self.save_model(f'checkpoint_epoch_{epoch}.pt')

        self.writer.close()

    def save_model(self, filename):
        """保存模型"""
        model_path = os.path.join(self.model_dir, filename)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, model_path)
        print(f'Model saved to {model_path}')

    def load_model(self, filename):
        """加载模型"""
        model_path = os.path.join("./checkpoints", filename)
        checkpoint = torch.load(model_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f'Model loaded from {model_path}')


def main():
    """主训练函数"""
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    # 数据路径（使用绝对路径）
    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_file = os.path.join(script_dir, "../ptb.train.txt")

    # 训练参数
    epochs = 50
    batch_size = 20
    bptt = 50
    embedding_dim = 32
    hidden_dim = 64
    lr = 0.001

    print("Loading data...")
    # 加载训练集并构建词表
    train_dataset = PTBDataset(train_file)
    vocab = train_dataset.vocab
    vocab_size = len(vocab)
    print(f'Vocab size: {vocab_size}')

    # 从训练集中切分出 10% 作为验证集
    train_ids = train_dataset.file_to_ids()
    split_idx = int(len(train_ids) * 0.9)  # 90% 用于训练，10% 用于验证
    train_ids_split = train_ids[:split_idx]
    val_ids_split = train_ids[split_idx:]

    print(f'Train set size: {len(train_ids_split)} tokens')
    print(f'Validation set size: {len(val_ids_split)} tokens (split from training data)')

    # 准备训练数据
    train_data = train_dataset.batchify(train_ids_split, batch_size, device)

    # 准备验证数据
    val_data = train_dataset.batchify(val_ids_split, batch_size, device)

    # 创建模型
    print("Creating model...")
    model = RNNLM(vocab_size, embedding_dim, hidden_dim).to(device)

    # 创建训练器
    trainer = RNNTrainer(
        model=model,
        train_data=train_data,
        val_data=val_data,
        log_dir="src/ai_2801/homework_2/logs",
        model_dir="src/ai_2801/homework_2/checkpoints",
        batch_size=batch_size,
        bptt=bptt,
        lr=lr
    )

    # 开始训练
    print("Starting training...")
    trainer.train(epochs, save_every=10)

    print("Training completed!")


if __name__ == "__main__":
    main()
