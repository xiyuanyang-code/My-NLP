"""
从 HW2.ipynb 中提取的 tokenizer 和 PPL 计算函数
展示 GPT2 和自定义 RNN 模型的 tokenizer 使用和 PPL 计算
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.nn import Softmax, CrossEntropyLoss
import torch.nn as nn


def load_gpt2_model(model_path: str):
    """
    加载 GPT2 模型和 tokenizer

    Args:
        model_path: GPT2 模型的本地路径

    Returns:
        model: GPT2 模型
        tokenizer: GPT2 tokenizer
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    model.eval()
    return model, tokenizer, device


def demo_tokenizer(tokenizer, text):
    """
    演示 tokenizer 的使用

    Args:
        tokenizer: GPT2 tokenizer
        text: 要处理的文本

    Returns:
        inputs: tokenized 输入
        tokens: token 列表
        decoded_string: 解码后的字符串
    """
    # 将文本转换为 token ids
    inputs = tokenizer(text, return_tensors="pt")

    # 将 token ids 转换为 tokens
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    # 将 token ids 解码回文本
    decoded_string = tokenizer.decode(inputs["input_ids"][0])

    print("=== Tokenizer Demo ===")
    print(f"Original text: {text}")
    print(f"Tokens: {tokens}")
    print(f"Decoded string: {decoded_string}")
    # * G-with-dot (Ġ) 代表空格
    print()

    return inputs, tokens, decoded_string


def calculate_gpt2_ppl(model, tokenizer, text, device="cuda"):
    """
    计算 GPT2 模型在给定文本上的 PPL (Perplexity)

    PPL 计算公式:
    PPL = exp(1/N * sum(-log P(w_i | w_1, w_2, ..., w_{i-1})))

    Args:
        model: GPT2 模型
        tokenizer: GPT2 tokenizer
        text: 要计算 PPL 的文本
        device: 设备 ('cuda' 或 'cpu')

    Returns:
        ppl: perplexity 值
    """
    # 将文本转换为输入 token
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs.input_ids.to(device)

    # 获取模型输出
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits
        labels = input_ids.to(logits.device)

        # GPT2 每个位置预测下一个 token，需要将 labels 向左移动一位
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # 使用 CrossEntropyLoss 计算平均损失
        loss_fct = CrossEntropyLoss(reduction="mean")
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )

        # PPL = exp(average negative log likelihood)
        ppl = torch.exp(loss)

    return ppl


def calculate_rnn_ppl(model, vocab, text, device="cuda"):
    """
    计算自定义 RNN 模型在给定文本上的 PPL

    Args:
        model: RNN 模型
        vocab: 词表字典
        text: 要计算 PPL 的文本
        device: 设备

    Returns:
        ppl: perplexity 值
    """
    # 将文本转换为词表索引
    words = text.replace("\n", "<eos>").split()
    input_ids = []
    for w in words:
        if w in vocab:
            input_ids.append(vocab[w])
        else:
            # 使用 <unk> token 处理未知词
            input_ids.append(vocab.get("<unk>", vocab["the"]))

    # 处理空输入的情况
    if len(input_ids) == 0:
        return float("inf")

    input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(1).to(device)

    # 获取模型输出
    model.eval()
    with torch.no_grad():
        hidden = model.init_hidden(1)  # batch_size = 1
        outputs = model(input_ids, hidden)
        logits = outputs.logits

        # 将 labels 向左移动一位
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[1:, :].contiguous()

        # 处理序列太短的情况
        if shift_logits.size(0) == 0 or shift_labels.size(0) == 0:
            return float("inf")

        # 使用 CrossEntropyLoss 计算平均损失
        loss_fct = nn.CrossEntropyLoss(reduction="mean")
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )

        # PPL = exp(average negative log likelihood)
        ppl = torch.exp(loss)

    return ppl


def main():
    model, tokenizer, device = load_gpt2_model("/data/xiyuanyang/My-NLP/models/gpt2")

    text1 = "GPT-4 is a large multimodal model (accepting image and text inputs, emitting text outputs) that, while less capable than humans in many real-world scenarios, exhibits human-level performance on various professional and academic"
    text2 = "Until the rocket ship nearly imploded. On Nov. 17, OpenAI's nonprofit board of directors fired Altman, without warning or even much in the way of "
    text3 = r"Consider a random walk on a 2D integer lattice starting at the origin $(0,0)$. At each step, a particle moves one unit Up, Down, Left, or Right with equal probability $p = 1/4$. Let $P_n$ be the probability that the particle returns to the origin for the first time at step $n$.The value of $P_n$ is non-zero only when $n$ is an even integer. If we define the generating function for the return probabilities as $W(z) = \sum_{n=0}^{\infty} q_n z^n$, where $q_n$ is the probability of being at the origin at step $n$, then the probability of ever returning to the origin is exactly"
    demo_tokenizer(tokenizer, text1)

    print("=== Calculating GPT2 PPL ===")
    ppl1 = calculate_gpt2_ppl(model, tokenizer, text1, device)
    ppl2 = calculate_gpt2_ppl(model, tokenizer, text2, device)
    ppl3 = calculate_gpt2_ppl(model, tokenizer, text3, device)
    print(f"Text 1 PPL: {ppl1.item():.4f}")
    print(f"Text 2 PPL: {ppl2.item():.4f}")
    print(f"Text 3 PPL: {ppl3.item():.4f}")


if __name__ == "__main__":
    main()
