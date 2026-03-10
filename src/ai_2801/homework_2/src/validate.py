import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from train_rnn import RNNLM


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


if __name__ == "__main__":
    gpt2_ppl = GPT2PPL("/data/xiyuanyang/My-NLP/models/gpt2")
    ppl = gpt2_ppl.calculate(
        "My research interests lie in constructing general autonomous agents and multi-agent collaborations."
    )
    print(ppl)
    rnn_ppl = RNNPPL(
        "src/ai_2801/homework_2/checkpoints/best_model.pt",
        "src/ai_2801/homework_2/checkpoints/vocab.pt",
    )
    ppl = rnn_ppl.calculate(
        "My research interests lie in constructing general autonomous agents and multi-agent collaborations."
    )
    print(ppl)

    # 文本生成示例
    generated_text = rnn_ppl.generate("My research interests lie in constructing general autonomous agents and multi-agent collaborations.", max_length=20)
    print(f"Generated: {generated_text}")
