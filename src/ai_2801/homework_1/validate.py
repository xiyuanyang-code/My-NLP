"""
批量实验脚本：对所有训练好的模型进行测试和可视化
"""

import os
import json
import re
import traceback
from tqdm import tqdm
from src.tester import Word2VecTester


def parse_model_folder(folder_name):
    """
    从文件夹名称中解析模型参数

    参数:
        folder_name: 文件夹名称，格式如 "embed20_lr0.001_ep10"

    返回:
        dict: 包含 embed_size, lr, epochs 的字典
    """
    params = {}

    # 提取 embed_size
    embed_match = re.search(r"embed(\d+)", folder_name)
    if embed_match:
        params["embed_size"] = int(embed_match.group(1))

    # 提取 learning rate
    lr_match = re.search(r"lr([\d.]+)", folder_name)
    if lr_match:
        params["lr"] = float(lr_match.group(1))

    # 提取 epochs
    ep_match = re.search(r"ep(\d+)", folder_name)
    if ep_match:
        params["epochs"] = int(ep_match.group(1))

    return params


def load_sampled_words(file_path="hm1/sampled_words_200.json"):
    """
    加载采样的词列表

    参数:
        file_path: 采样词文件路径

    返回:
        list: 采样词列表
    """
    with open(file_path, "r", encoding="utf-8") as f:
        words = json.load(f)
    return words


def check_words_in_vocab(tester, words):
    """
    检查词语是否在词典中

    参数:
        tester: Word2VecTester实例
        words: 要检查的词语列表

    返回:
        tuple: (在词典中的词列表, 不在词典中的词列表)
    """
    in_vocab = []
    not_in_vocab = []

    for word in words:
        if word in tester.token_to_idx:
            in_vocab.append(word)
        else:
            not_in_vocab.append(word)

    return in_vocab, not_in_vocab


def run_similarity_tests(tester, test_words, output_dir=None):
    """
    运行相似词测试

    参数:
        tester: Word2VecTester实例
        test_words: 要测试的词语列表
        output_dir: 输出目录

    返回:
        dict: 相似词测试结果
    """

    results = {}

    for word in test_words:
        if word in tester.token_to_idx:
            similar_words = tester.get_similar_tokens(word, k=10)
            if similar_words:
                results[word] = similar_words
        else:
            print(f"\n⚠️  词 '{word}' 不在词典中，跳过")
            results[word] = {"error": "word not in vocabulary"}

    # 保存结果到文件
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        results_file = os.path.join(output_dir, "similarity_results.json")
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    return results


def run_visualization(tester: Word2VecTester, words, output_dir=None):
    """
    运行词向量可视化

    参数:
        tester: Word2VecTester实例
        words: 要可视化的词语列表
        output_dir: 输出目录，None则使用模型目录
    """
    tester.visualize_embeddings(words=words, method="tsne", output_dir=output_dir)
    tester.visualize_embeddings(words=words, method="pca", output_dir=output_dir)


def run_analogy_tests(tester, analogies, output_dir=None):
    """
    运行词类比测试

    参数:
        tester: Word2VecTester实例
        analogies: 类比任务列表，每个元素是 (word1, word2, word3) 的元组
        output_dir: 输出目录

    返回:
        dict: 类比测试结果
    """

    results = {}

    for word1, word2, word3 in analogies:
        # 检查所有词是否都在词典中
        all_words = [word1, word2, word3]
        in_vocab, not_in_vocab = check_words_in_vocab(tester, all_words)

        task_key = f"{word1}_{word2}_{word3}"

        if not_in_vocab:
            print(f"\n⚠️  类比任务: {word1} - {word2} + {word3} = ?")
            print(f"   以下词不在词典中: {not_in_vocab}")
            print(f"   跳过此任务")
            results[task_key] = {"error": f"words not in vocabulary: {not_in_vocab}"}
            continue

        analogy_results = tester.word_analogy(word1, word2, word3, k=5)
        if analogy_results:
            results[task_key] = {
                "query": f"{word1} - {word2} + {word3}",
                "results": analogy_results,
            }

    # 保存结果到文件
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        results_file = os.path.join(output_dir, "analogy_results.json")
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    return results


def main():
    """主函数：批量运行所有实验"""

    # 配置
    checkpoints_dir = "checkpoints"
    sampled_words_file = "hm1/sampled_words_200.json"

    sampled_words = load_sampled_words(sampled_words_file)

    # 相似词测试词语
    similarity_test_words = ["dog", "he", "bank", "responsibility", "eagerness"]

    # 词类比任务
    analogy_tasks = [
        ("king", "man", "woman"),  # queen
        ("paris", "france", "italy"),  # rome
        ("tokyo", "japan", "germany"),  # berlin
        ("doctor", "hospital", "teacher"),  # school
    ]

    # 遍历checkpoints目录
    if not os.path.exists(checkpoints_dir):
        print(f"错误: checkpoints目录不存在: {checkpoints_dir}")
        return

    subdirs = [
        d
        for d in os.listdir(checkpoints_dir)
        if os.path.isdir(os.path.join(checkpoints_dir, d))
    ]
    subdirs.sort()  # 按名称排序
    subdir_length = len(subdirs)

    # 处理每个checkpoint
    for subdir in tqdm(subdirs, total=subdir_length):
        model_path = os.path.join(checkpoints_dir, subdir)

        try:
            # 加载模型
            tester = Word2VecTester(model_path, device="cpu")

            # 运行相似词测试，结果保存在checkpoint文件夹中
            run_similarity_tests(tester, similarity_test_words, output_dir=model_path)

            # 运行可视化，图片保存在checkpoint文件夹中
            run_visualization(tester, sampled_words, output_dir=model_path)

            # 运行词类比测试，结果保存在checkpoint文件夹中
            run_analogy_tests(tester, analogy_tasks, output_dir=model_path)

        except Exception as e:
            traceback.print_exc()
            continue


if __name__ == "__main__":
    main()
