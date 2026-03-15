import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from train_rnn import RNNLM
import json
import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import time
import threading


class GPT2PPL:
    def __init__(self, model_path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path).to(self.device)
        self.model.eval()

    def calculate(self, text):
        inputs = self.tokenizer(text, return_tensors="pt")
        input_ids = inputs.input_ids.to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs.logits

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()

            loss = nn.CrossEntropyLoss()(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )

        return torch.exp(loss).item()


class RNNPPL:
    def __init__(self, model_path, vocab_path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.vocab = torch.load(vocab_path, map_location=self.device)
        checkpoint = torch.load(model_path, map_location=self.device)
        state_dict = checkpoint["model_state_dict"]

        vocab_size = len(self.vocab)
        embedding_dim = state_dict["embedding.weight"].shape[1]
        hidden_dim = state_dict["b1"].shape[0]

        self.model = self._build_model(vocab_size, embedding_dim, hidden_dim)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device).eval()

    def _build_model(self, vocab_size, embedding_dim, hidden_dim):
        return RNNLM(
            vocab_size=vocab_size, embedding_dim=embedding_dim, hidden_dim=hidden_dim
        )

    def calculate(self, text):
        words = text.replace("\n", "<eos>").split()
        unk_id = self.vocab.get("<unk>", self.vocab.get("the", 0))

        input_ids = (
            torch.tensor([self.vocab.get(w, unk_id) for w in words], dtype=torch.long)
            .unsqueeze(1)
            .to(self.device)
        )

        with torch.no_grad():
            hidden = self.model.init_hidden(1).to(self.device)
            outputs = self.model(input_ids, hidden)
            logits = outputs if torch.is_tensor(outputs) else outputs.logits

            # shift_logits: 取前 L-1 个时间步 [0, L-2]
            shift_logits = logits[:-1, :, :].contiguous()
            # shift_labels: 取后 L-1 个时间步 [1, L-1]
            shift_labels = input_ids[1:, :].contiguous()

            loss = nn.CrossEntropyLoss()(
                shift_logits.view(-1, shift_logits.size(-1)),  # 展平为 [ (L-1)*B, V ]
                shift_labels.view(-1),  # 展平为 [ (L-1)*B ]
            )

        return torch.exp(loss).item()

    def generate(self, text, max_length=50):
        words = text.replace("\n", "<eos>").split()
        unk_id = self.vocab.get("<unk>", self.vocab.get("the", 0))

        input_ids = torch.tensor(
            [self.vocab.get(w, unk_id) for w in words], dtype=torch.long
        ).unsqueeze(1)

        generated = input_ids.clone()

        with torch.no_grad():
            hidden = self.model.init_hidden(1).to(self.device)
            outputs = self.model(input_ids.to(self.device), hidden)
            hidden = outputs if torch.is_tensor(outputs) else outputs.hidden

            for _ in range(max_length):
                outputs = self.model(generated[-1:].to(self.device), hidden)
                logits = outputs if torch.is_tensor(outputs) else outputs.logits
                hidden = outputs if torch.is_tensor(outputs) else outputs.hidden

                next_token = torch.argmax(logits[-1, :, :], dim=-1).unsqueeze(0)
                generated = torch.cat([generated, next_token.cpu()], dim=0)

                if next_token.item() == self.vocab.get("<eos>", -1):
                    break

        # Convert back to words
        inv_vocab = {i: w for w, i in self.vocab.items()}
        result_words = [inv_vocab.get(idx.item(), "<unk>") for idx in generated]
        result_text = " ".join(result_words).replace("<eos>", "\n")
        return result_text


def process_single_item(args):
    """
    处理单个文本样本的函数，用于多进程执行
    每个进程会独立加载和运行模型

    Args:
        args: 元组 (text, model_name, model_path, vocab_path)

    Returns:
        dict: 包含 ppl, text, time 的字典
    """
    text, model_name, model_path, vocab_path = args

    start_time = time.time()
    result = {
        "text": text.strip(),
        "time": 0,
        "ppl": None
    }

    try:
        # 每个进程中创建模型实例（多进程各自独立）
        if model_name == "gpt_2_baseline":
            ppl_calculator = GPT2PPL(model_path)
        else:
            ppl_calculator = RNNPPL(model_path, vocab_path)

        ppl = ppl_calculator.calculate(text)
        result["ppl"] = ppl

    except Exception as e:
        result["ppl"] = None
        result["error"] = str(e)

    result["time"] = time.time() - start_time
    return result


def batch_validate(
    test_file_path,
    output_file_path,
    model_name,
    model_path,
    vocab_path=None,
    num_workers=10,
):
    """
    批量验证函数，使用并发处理多个样本

    Args:
        test_file_path: 测试文件路径
        output_file_path: 输出 JSONL 文件路径
        model_name: 模型名称 ("gpt_2_baseline" 或其他 RNN 模型名)
        model_path: 模型路径
        vocab_path: 词表路径（RNN 模型需要）
        num_workers: 并发工作进程数（对于 GPT-2 建议设为 1）
    """
    # 读取测试数据
    print(f"Loading test data from {test_file_path}...")
    test_texts = []
    with open(test_file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:  # 跳过空行
                test_texts.append(line)

    print(f"Loaded {len(test_texts)} test samples")
    print(f"Model: {model_name}")

    # 准备任务参数
    tasks = [
        (text, model_name, model_path, vocab_path)
        for text in test_texts
    ]

    # 使用进程池并发处理
    print(f"Starting validation with {num_workers} workers...")
    start_time = time.time()

    # 创建输出目录并清空/创建输出文件
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    # 清空输出文件（如果存在）
    open(output_file_path, "w").close()

    results = []
    write_lock = threading.Lock()

    def write_result(result):
        """线程安全地写入单个结果"""
        with write_lock:
            with open(output_file_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # 提交所有任务
        futures = {executor.submit(process_single_item, task): task for task in tasks}

        # 使用 tqdm 显示进度，实时写入结果
        for future in tqdm(as_completed(futures), total=len(tasks), desc="Processing"):
            try:
                result = future.result()
                results.append(result)
                # 实时写入文件
                write_result(result)
            except Exception as e:
                print(f"Error processing task: {e}")

    elapsed_time = time.time() - start_time
    print(f"Completed in {elapsed_time:.2f} seconds")

    # 打印统计信息
    print("\n=== Validation Statistics ===")
    print(f"Total samples: {len(results)}")
    success_count = sum(1 for r in results if r["ppl"] is not None)
    error_count = sum(1 for r in results if r["ppl"] is None)
    print(f"Successful: {success_count}")
    print(f"Errors: {error_count}")

    if success_count > 0:
        ppls = [int(r["ppl"]) for r in results if r["ppl"] is not None]
        avg_ppl = sum(ppls) / len(ppls)
        print(f"Average Perplexity: {avg_ppl:.4f}")

    total_time = sum(r["time"] for r in results)
    print(f"Total time: {total_time:.2f}s")
    print(f"Average time per sample: {total_time / len(results):.4f}s")

    print(f"\nResults saved to: {output_file_path}")


def main(model_name, model_path, vocab_path):
    test_file_path = "src/ai_2801/homework_2/data/ptb.test.txt"
    output_file_path = f"src/ai_2801/homework_2/result/{model_name}/results.jsonl"

    # 根据 model_name 确定 model_path
    if model_name == "gpt_2_baseline":
        actual_model_path = "/data/xiyuanyang/My-NLP/models/gpt2"
        actual_vocab_path = None
    else:
        actual_model_path = model_path
        actual_vocab_path = vocab_path

    # 运行批量验证
    batch_validate(
        test_file_path=test_file_path,
        output_file_path=output_file_path,
        model_name=model_name,
        model_path=actual_model_path,
        vocab_path=actual_vocab_path,
        num_workers=10,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate language models on PTB test set")
    parser.add_argument("--model_name", type=str, required=True,
                        help="Model name (e.g., 'gpt_2_baseline' for GPT-2, or custom name for RNN)")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Model path (required for RNN models, ignored for GPT-2)")
    parser.add_argument("--vocab_path", type=str, default=None,
                        help="Vocab path (required for RNN models, ignored for GPT-2)")

    args = parser.parse_args()
    model_name = args.model_name
    model_path = args.model_path
    vocab_path = args.vocab_path

    main(model_name=model_name, model_path=model_path, vocab_path=vocab_path)
