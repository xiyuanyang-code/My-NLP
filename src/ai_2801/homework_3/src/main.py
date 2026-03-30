"""
FastText 训练脚本 - 支持官方 fasttext 和自定义实现
"""

import argparse
import os
import sys
import time
import json
from datetime import datetime
from tqdm.auto import tqdm
import torch


def reformat_text(text: str) -> str:
    """预处理数据"""
    text = text.replace('\n', ' ').replace('\r', ' ')
    text = text.lower()
    for punct in ['.', '\\', '!', '?', ',', "'", '/', '(', ')']:
        text = text.replace(punct, ' ' + punct + ' ')
    text = ' '.join(text.split())
    return text


def prepare_data(processor_config):
    """准备训练数据"""
    from datasets import load_dataset
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    print("正在初始化数据处理器...")

    print(f"正在加载数据集 '{processor_config['dataset_name']}'...")
    dataset = load_dataset(processor_config['dataset_name'], split="train")

    # 分割数据集
    print("正在分割数据集...")
    train_dataset = dataset.select(range(450000))
    test_dataset = dataset.select(range(450000, len(dataset)))
    print(f"数据集大小 -> 训练集: {len(train_dataset)}, 测试集: {len(test_dataset)}")

    df = train_dataset.to_pandas()
    # 绘制分数分布图
    plt.figure(figsize=(10, 6))
    sns.countplot(x='score', data=df, palette='viridis')
    plt.title('Distribution of Scores in the Training Dataset')
    plt.xlabel('Score')
    plt.ylabel('Count')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

    # 保存数据
    print(f"正在向 {processor_config['train_file']} 保存训练数据...")
    with open(processor_config['train_file'], "w", encoding="utf-8") as f:
        for item in tqdm(train_dataset, desc="处理训练数据"):
            score = int(item['score'])
            text = reformat_text(item['text'])
            f.write(f"__label__{score} {text}\n")

    print(f"正在向 {processor_config['test_file']} 保存测试数据...")
    with open(processor_config['test_file'], "w", encoding="utf-8") as f:
        for item in tqdm(test_dataset, desc="处理测试数据"):
            score = int(item['score'])
            text = reformat_text(item['text'])
            f.write(f"__label__{score} {text}\n")

    # 节约内存
    del dataset
    del train_dataset
    del test_dataset
    print("数据准备完成。")


def train_with_official_fasttext(model_config, output_dir):
    """使用官方 fasttext 训练"""
    import fasttext
    import re
    import pandas as pd

    print("\n========== 使用官方 FastText  ==========")

    # 训练模型
    print(f"开始训练模型... (将保存至 {model_config['model_path']})")
    model = fasttext.train_supervised(**model_config["model_params"])
    model.save_model(model_config['model_path'])
    print("训练完成。")

    # 评估模型
    print(f"\n在 {model_config['test_file']} 上进行评估...")
    result = model.test(model_config['test_file'])
    print(f"总体结果 -> 样本数: {result[0]}, Precision@1: {result[1]:.4f}, Recall@1: {result[2]:.4f}")

    # 详细分析和混淆矩阵
    with open(model_config['test_file'], "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    texts = [re.sub(r'__label__\d+\s', '', line) for line in lines]
    true_labels = [int(re.match(r'__label__(\d+)', line).group(1)) for line in lines]

    pred_labels_raw, _ = model.predict(texts)
    predicted_labels = [int(p[0].replace('__label__', '')) for p in pred_labels_raw]

    # 混淆矩阵
    confusion_matrix = pd.crosstab(
        pd.Series(true_labels, name='True'),
        pd.Series(predicted_labels, name='Predicted')
    )
    print("\n--- 混淆矩阵 ---")
    print(confusion_matrix)

    # 保存混淆矩阵
    confusion_matrix_path = os.path.join(output_dir, 'confusion_matrix.csv')
    confusion_matrix.to_csv(confusion_matrix_path)
    print(f"混淆矩阵已保存到: {confusion_matrix_path}")

    return {
        'num_samples': result[0],
        'precision': result[1],
        'recall': result[2],
    }


def train_with_custom_fasttext(model_config, output_dir):
    """使用自定义 fasttext 训练"""
    # 导入自定义实现
    sys.path.insert(0, os.path.dirname(__file__))
    from my_fasttext_trainer import fasttext as custom_fasttext

    print("\n========== 使用自定义 FastText ==========")
    print(f"TensorBoard 日志目录: {model_config.get('tensorboard_log_dir', 'runs/fasttext_experiment')}")

    # 训练模型
    model = custom_fasttext.train_supervised(**model_config["model_params"])

    # 评估模型
    print(f"\n在 {model_config['test_file']} 上进行评估...")
    result = model.test(model_config['test_file'])
    print(f"总体结果 -> 样本数: {result[0]}, Precision@1: {result[1]:.4f}, Recall@1: {result[2]:.4f}")

    # 保存模型
    model_path = os.path.join(output_dir, 'custom_model.pt')
    model.save_model(model_path)
    print(f"模型已保存到: {model_path}")

    return {
        'num_samples': result[0],
        'precision': result[1],
        'recall': result[2],
    }


def clean_finetune_data(model_config, cleaner_config, use_official):
    """使用训练好的模型清洗 FineWeb 数据集"""
    from datasets import load_dataset
    from torch.utils.data import DataLoader
    import re

    print("\n========== 开始数据清洗 ==========")

    # 加载模型
    if use_official:
        import fasttext
        print(f"从 {model_config['model_path']} 加载官方 fasttext 模型...")
        model = fasttext.load_model(model_config['model_path'])

        def predict_fn(texts):
            labels, _ = model.predict(texts)
            return [int(label[0].replace('__label__', '')) for label in labels]
    else:
        sys.path.insert(0, os.path.dirname(__file__))
        from my_fasttext_trainer import fasttext as custom_fasttext
        # 使用 model_config 中的模型路径
        model_path = model_config['model_path']
        print(f"从 {model_path} 加载自定义 fasttext 模型...")
        model = custom_fasttext.load_model(model_path)

        def predict_fn(texts):
            labels, _ = model.predict(texts, k=1)
            return [int(label[0].replace('__label__', '')) for label in labels]

    # 加载数据集
    print("以流式模式加载 FineWeb 数据集...")
    fw_stream = load_dataset(
        cleaner_config['streaming_dataset_name'],
        name=cleaner_config['streaming_dataset_config'],
        split="train",
        streaming=False
    )

    data_loader = DataLoader(
        fw_stream,
        batch_size=cleaner_config['batch_size'],
        num_workers=cleaner_config['num_workers'],
        collate_fn=lambda batch: [reformat_text(item['text']) for item in batch],
        pin_memory=True
    )

    nums_saved = 0
    total_processed = 0
    print(f"开始清洗数据。高质量数据将被保存到 {cleaner_config['cleaned_output_file']}")

    with open(cleaner_config['cleaned_output_file'], "w", encoding='utf-8') as f:
        for text_batch in tqdm(data_loader, desc="清洗 FineWeb 数据"):
            total_processed += len(text_batch)

            # 预测分数
            pred_scores = predict_fn(text_batch)

            # 过滤并保存高质量数据
            for text, score in zip(text_batch, pred_scores):
                if score >= cleaner_config['quality_threshold']:
                    cleaned_text = text.replace('\n', ' ').replace('\r', ' ')
                    f.write(cleaned_text + '\n')
                    nums_saved += 1

    print(f"\n清洗完成！共处理了 {total_processed} 条文档，保存了 {nums_saved} 条高质量文档。")

    return total_processed, nums_saved


# ==================== 超参数配置 ====================

HYPERPARAMETERS = {
    # 数据集路径配置
    'dataset_path': 'src/ai_2801/homework_3/fineweb-edu-llama3-annotations',
    'fineweb_path': 'src/ai_2801/homework_3/fineweb',
    'train_file': 'src/ai_2801/homework_3/dataset/train_dataset.txt',
    'test_file': 'src/ai_2801/homework_3/dataset/test_dataset.txt',

    # 训练超参数
    'epoch': 5,
    'lr': 0.1,
    'dim': 100,
    'wordNgrams': 1,
    'bucket': 2000000,
    'minn': 3,
    'maxn': 6,
    'batch_size': 32,
    'thread': 4,

    # 数据清洗参数
    'quality_threshold': 3,  # 只保存分数 >= 3 的文本
    'cleaning_batch_size': 500,
    'num_workers': 10,

    # 自定义实现专用参数
    'validation_split': 0.1,  # 验证集比例
    'tensorboard_log_dir': 'src/ai_2801/homework_3/runs/fasttext_experiment',
}


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='FastText 训练脚本')
    parser.add_argument(
        '--use_official',
        action='store_true',
        help='使用官方 fasttext 库训练（默认使用自定义实现）'
    )
    parser.add_argument(
        '--no_training',
        action='store_true',
        help='跳过训练，直接加载已有模型进行数据清洗'
    )
    parser.add_argument(
        '--model_path',
        type=str,
        default=None,
        help='指定要加载的模型路径（用于 --no_training 模式）'
    )

    args = parser.parse_args()

    # 验证参数
    if args.no_training and args.model_path is None:
        print("错误: 使用 --no_training 时必须指定 --model_path")
        sys.exit(1)

    if args.no_training and not os.path.exists(args.model_path):
        print(f"错误: 指定的模型文件不存在: {args.model_path}")
        sys.exit(1)

    # 生成时间戳和输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    implementation = "official" if args.use_official else "custom"

    if args.no_training:
        # 使用已有模型的目录
        output_dir = os.path.dirname(args.model_path)
        model_filename = os.path.basename(args.model_path)
        print(f"\n{'='*60}")
        print(f"FastText 数据清洗模式（跳过训练）")
        print(f"实现方式: {'官方 fasttext' if args.use_official else '自定义 fasttext'}")
        print(f"加载模型: {args.model_path}")
        print(f"输出目录: {output_dir}")
        print(f"{'='*60}\n")
    else:
        output_dir = f"src/ai_2801/homework_3/output/fasttext_{implementation}_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs("dataset", exist_ok=True)

        print(f"\n{'='*60}")
        print(f"FastText 训练脚本")
        print(f"实现方式: {'官方 fasttext' if args.use_official else '自定义 fasttext'}")
        print(f"输出目录: {output_dir}")
        print(f"{'='*60}\n")

    # 打印超参数配置
    print("========== 超参数配置 ==========")
    for key, value in HYPERPARAMETERS.items():
        print(f"  {key}: {value}")
    print(f"{'='*60}\n")

    # 统计信息
    statistics = {
        'implementation': implementation,
        'use_official': args.use_official,
        'timestamp': timestamp,
        'output_dir': output_dir,
        'hyperparameters': HYPERPARAMETERS.copy(),
        'start_time': None,
        'end_time': None,
        'duration_seconds': None,
        'training_results': None,
        'cleaning_results': None,
    }

    start_time = time.time()
    statistics['start_time'] = datetime.now().isoformat()

    try:
        # ==================== 模型训练 ====================
        model_config = {
            "model_path": os.path.join(output_dir, 'model.bin'),
            "train_file": HYPERPARAMETERS['train_file'],
            "test_file": HYPERPARAMETERS['test_file'],
            "tensorboard_log_dir": os.path.join(output_dir, 'tensorboard_logs'),
        }

        if args.no_training:
            # 跳过训练，直接使用已有模型
            model_config["model_path"] = args.model_path
            print("跳过训练步骤，直接加载已有模型...\n")
        else:
            # 构建模型参数
            if args.use_official:
                model_config["model_params"] = {
                    "input": HYPERPARAMETERS['train_file'],
                    "epoch": HYPERPARAMETERS['epoch'],
                    "lr": HYPERPARAMETERS['lr'],
                    "dim": HYPERPARAMETERS['dim'],
                    "wordNgrams": HYPERPARAMETERS['wordNgrams'],
                    "bucket": HYPERPARAMETERS['bucket'],
                    "minn": HYPERPARAMETERS['minn'],
                    "maxn": HYPERPARAMETERS['maxn'],
                    "thread": HYPERPARAMETERS['thread'],
                }
            else:
                model_config["model_params"] = {
                    "input": HYPERPARAMETERS['train_file'],
                    "epoch": HYPERPARAMETERS['epoch'],
                    "lr": HYPERPARAMETERS['lr'],
                    "dim": HYPERPARAMETERS['dim'],
                    "wordNgrams": HYPERPARAMETERS['wordNgrams'],
                    "bucket": HYPERPARAMETERS['bucket'],
                    "minn": HYPERPARAMETERS['minn'],
                    "maxn": HYPERPARAMETERS['maxn'],
                    "batch_size": HYPERPARAMETERS['batch_size'],
                    "thread": HYPERPARAMETERS['thread'],
                    "validation_split": HYPERPARAMETERS['validation_split'],
                    "tensorboard_log_dir": model_config["tensorboard_log_dir"],
                }

            # 训练
            if args.use_official:
                training_results = train_with_official_fasttext(model_config, output_dir)
            else:
                training_results = train_with_custom_fasttext(model_config, output_dir)

            statistics['training_results'] = training_results
            statistics['training_results']['config'] = model_config["model_params"]

        cleaner_config = {
            "output_dir": output_dir,
            "streaming_dataset_name": HYPERPARAMETERS['fineweb_path'],
            "streaming_dataset_config": None,
            "cleaned_output_file": os.path.join(output_dir, 'cleaned_dataset.txt'),
            "quality_threshold": HYPERPARAMETERS['quality_threshold'],
            "batch_size": HYPERPARAMETERS['cleaning_batch_size'],
            "num_workers": HYPERPARAMETERS['num_workers'],
        }

        total_processed, total_saved = clean_finetune_data(
            model_config, cleaner_config, args.use_official
        )

        statistics['cleaning_results'] = {
            'total_processed': total_processed,
            'total_saved': total_saved,
            'save_rate': total_saved / total_processed if total_processed > 0 else 0.0,
            'quality_threshold': HYPERPARAMETERS['quality_threshold'],
        }

        # ==================== 保存统计信息 ====================
        end_time = time.time()
        statistics['end_time'] = datetime.now().isoformat()
        statistics['duration_seconds'] = end_time - start_time

        statistics_path = os.path.join(output_dir, 'statistics.json')
        with open(statistics_path, 'w', encoding='utf-8') as f:
            json.dump(statistics, f, indent=2, ensure_ascii=False)

        print(f"\n{'='*60}")
        if args.no_training:
            print(f"数据清洗完成！（跳过训练模式）")
        else:
            print(f"训练和清洗完成！")
        print(f"统计信息已保存到: {statistics_path}")
        print(f"总耗时: {statistics['duration_seconds']:.2f} 秒")
        if statistics['training_results']:
            print(f"测试集准确率: {statistics['training_results']['precision']:.4f}")
        if statistics['cleaning_results']:
            print(f"数据保存率: {statistics['cleaning_results']['save_rate']:.2%}")
        print(f"{'='*60}")

        if not args.use_official and not args.no_training:
            print(f"\n查看 TensorBoard:")
            print(f"  tensorboard --logdir={model_config['tensorboard_log_dir']}")

    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
