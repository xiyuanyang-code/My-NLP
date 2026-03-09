"""
从词表中随机采样200个词，用于可视化实验
"""
import json
import random
import os

def sample_words(vocab_path, output_path, num_samples=200, seed=42):
    """
    从词表中随机采样指定数量的词

    参数:
        vocab_path: 词表文件路径
        output_path: 输出文件路径
        num_samples: 采样数量
        seed: 随机种子，确保可重复性
    """
    # 设置随机种子
    random.seed(seed)

    # 读取词表
    print(f"正在读取词表: {vocab_path}")
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab_data = json.load(f)

    token_to_idx = vocab_data["token_to_idx"]
    idx_to_token = vocab_data["idx_to_token"]

    # 获取所有词（排除特殊标记）
    all_words = []
    # idx_to_token 是一个列表，索引即为词的ID
    for idx, token in enumerate(idx_to_token):
        # 排除特殊标记（如果需要）
        # if not token.startswith('<') and not token.startswith('['):
        all_words.append(token)

    print(f"词表总词数: {len(all_words)}")

    # 随机采样
    sampled_words = random.sample(all_words, min(num_samples, len(all_words)))

    print(f"采样词数: {len(sampled_words)}")

    # 保存到文件
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(sampled_words, f, ensure_ascii=False, indent=2)

    print(f"已保存到: {output_path}")

    # 显示前10个采样词
    print("\n前10个采样词:")
    for i, word in enumerate(sampled_words[:10], 1):
        print(f"{i}. {word}")

    return sampled_words

if __name__ == "__main__":
    # 词表路径
    vocab_path = "hm1/output/initial/vocab.json"

    # 输出路径
    output_path = "hm1/data/sampled_words_200.json"

    # 采样200个词
    sampled_words = sample_words(vocab_path, output_path, num_samples=200)

    print("\n采样完成！")
    print(f"这200个词将用于所有可视化实验，确保结果的一致性和可比性。")
