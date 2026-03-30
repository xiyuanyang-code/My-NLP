"""
BERT模型数据清洗脚本
从 HW3.ipynb 中提取的 BERT 模型部分，用于 FineWeb 数据清洗
"""

import re
import os
import time
import json
from datetime import datetime
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch


def reformat_text(text: str) -> str:
    """
    预处理数据

    Args:
        text: 原始文本

    Returns:
        预处理后的文本
    """
    # 将每条数据里的换行符去掉
    text = text.replace('\n', ' ').replace('\r', ' ')
    # 将所有字母转化为小写
    text = text.lower()
    # 将标点符号.\!?,'/()前后添加空格
    for punct in ['.', '\\', '!', '?', ',', "'", '/', '(', ')']:
        text = text.replace(punct, ' ' + punct + ' ')
    # 将多个连续空格替换为单个空格，并去除首尾空格
    text = ' '.join(text.split())
    return text


class BertFineWebClassifier:
    """
    一个封装了 Hugging Face fineweb-edu-classifier 模型的分类器。
    """
    def __init__(self, model_name):
        print(f"正在从 '{model_name}' 加载模型...")

        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            print(f"Found {torch.cuda.device_count()} GPUs. Using DataParallel.")
            self.device = "cuda"  # 主设备设为cuda

            # 正常加载模型（先加载到CPU或主GPU）
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_name)

            # 使用 nn.DataParallel 包装模型
            self.model = torch.nn.DataParallel(model)

            # 将包装后的模型移动到GPU设备
            self.model.to(self.device)
            self.model.eval()

            self.max_length = self.model.module.config.max_position_embeddings
            print(f"Model loaded and wrapped with DataParallel. Max sequence length: {self.max_length}")
        else:
            self.device = 'cuda' if torch.cuda.is_available() else "cpu"
            print(f"模型将运行在: {self.device}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()

            self.max_length = self.model.config.max_position_embeddings

    def predict(self, text_batch):
        """
        对文本批次进行预测

        Args:
            text_batch: 文本列表

        Returns:
            预测分数列表
        """
        with torch.no_grad():
            inputs = self.tokenizer(
                text_batch,
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=self.max_length
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model(**inputs)
            scores = outputs.logits.squeeze(-1).float().cpu().detach().numpy()
            return scores.tolist()


class FineWebDataCleaner:
    """
    使用一个 BERT 分类器以流式方式清洗 FineWeb 数据集。
    """
    def __init__(self, config, classifier):
        self.config = config
        self.classifier = classifier
        print("\n正在初始化 FineWeb 数据清洗器...")

    def _collate_fn(self, batch):
        """为 DataLoader 准备的函数，用于从每个样本中提取文本。"""
        return [reformat_text(item['text']) for item in batch]

    def clean_data(self):
        """以流式处理数据集，按分数过滤并保存结果。"""
        print("以流式模式加载 FineWeb 数据集...")
        fw_stream = load_dataset(
            self.config['streaming_dataset_name'],
            name=self.config['streaming_dataset_config'],
            split="train",
            streaming=False
        )

        data_loader = DataLoader(
            fw_stream,
            batch_size=self.config['batch_size'],
            num_workers=self.config['num_workers'],
            collate_fn=self._collate_fn,
            pin_memory=True
        )

        nums_saved = 0
        total_processed = 0
        print(f"开始清洗数据。高质量数据将被保存到 {self.config['cleaned_output_file']}")

        with open(self.config['cleaned_output_file'], "w", encoding='utf-8') as f:
            for text_batch in tqdm(data_loader, desc="清洗 FineWeb 数据"):
                total_processed += len(text_batch)

                # 预测分数
                pred_scores = self.classifier.predict(text_batch)

                # 过滤并保存高质量数据
                for text, score in zip(text_batch, pred_scores):
                    if score >= self.config['quality_threshold']:
                        # 删除换行符并以换行符区分不同数据
                        cleaned_text = text.replace('\n', ' ').replace('\r', ' ')
                        f.write(cleaned_text + '\n')
                        nums_saved += 1

        print(f"\n清洗完成！共处理了 {total_processed} 条文档，保存了 {nums_saved} 条高质量文档。")
        return total_processed, nums_saved


def main():
    """主函数"""
    # 生成时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 创建输出目录
    output_dir = f"src/ai_2801/homework_3/output/bert_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    # BERT模型配置
    MODEL_NAME = "/data/xiyuanyang/My-NLP/models/fineweb_edu_classifier"

    # 数据清洗配置
    config = {
        'streaming_dataset_name': 'src/ai_2801/homework_3/fineweb/sample/10BT',
        'streaming_dataset_config': None,
        'batch_size': 512,
        'num_workers': 1,
        'quality_threshold': 3,  # 只保存分数 >= 3 的文本
        'cleaned_output_file': os.path.join(output_dir, 'cleaned_dataset.txt'),
    }

    # 统计信息
    statistics = {
        'model_name': MODEL_NAME,
        'batch_size': config['batch_size'],
        'quality_threshold': config['quality_threshold'],
        'timestamp': timestamp,
        'start_time': None,
        'end_time': None,
        'duration_seconds': None,
        'total_processed': 0,
        'total_saved': 0,
        'save_rate': 0.0,
    }

    try:
        # 记录开始时间
        start_time = time.time()
        statistics['start_time'] = datetime.now().isoformat()

        # 初始化BERT分类器
        classifier = BertFineWebClassifier(model_name=MODEL_NAME)

        # 初始化数据清洗器
        cleaner = FineWebDataCleaner(config, classifier)

        # 开始执行清洗任务
        print(f"\n开始数据清洗任务...")
        total_processed, total_saved = cleaner.clean_data()

        # 记录结束时间
        end_time = time.time()
        statistics['end_time'] = datetime.now().isoformat()
        statistics['duration_seconds'] = end_time - start_time
        statistics['total_processed'] = total_processed
        statistics['total_saved'] = total_saved
        statistics['save_rate'] = total_saved / total_processed if total_processed > 0 else 0.0

        # 保存统计信息
        statistics_path = os.path.join(output_dir, 'statistics.json')
        with open(statistics_path, 'w', encoding='utf-8') as f:
            json.dump(statistics, f, indent=2, ensure_ascii=False)

        print(f"\n统计信息已保存到: {statistics_path}")
        print(f"清洗耗时: {statistics['duration_seconds']:.2f} 秒")
        print(f"保存率: {statistics['save_rate']:.2%}")

    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
