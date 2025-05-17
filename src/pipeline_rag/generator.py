from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch


class GPT2Generator:
    def __init__(self, max_tokens=50, temperature=0.7, top_p=0.8):
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.model = GPT2LMHeadModel.from_pretrained("gpt2")
        self.model.eval()
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p

    def generate(self, query: str, docs: list) -> str:
        prompt = f"Context:\n- " + "\n- ".join(docs) + f"\nQuestion: {query}\nAnswer:"
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=512
        )
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                do_sample=False,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
