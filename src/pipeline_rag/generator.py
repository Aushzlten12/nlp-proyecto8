from transformers import GPT2Tokenizer, GPT2LMHeadModel


class Generator:
    def __init__(self, model_name: str = "gpt2"):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)

    def generate(self, context: str, query: str, max_new_tokens: int = 50) -> str:
        """Concatena contexto+pregunta y genera la respuesta."""
        prompt = context + "\n\nQ: " + query + "\nA:"
        inputs = self.tokenizer(prompt, return_tensors="pt")
        out = self.model.generate(
            **inputs,
            max_length=inputs.input_ids.shape[1] + max_new_tokens,
            num_beams=5,
            no_repeat_ngram_size=2,
            early_stopping=True
        )
        text = self.tokenizer.decode(out[0], skip_special_tokens=True)
        return text.split("A:")[-1].strip()
