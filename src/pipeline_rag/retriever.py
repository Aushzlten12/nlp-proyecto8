from preprocess_index import preprocess


class Retriever:
    def __init__(self, bm25, docs):
        self.bm25 = bm25
        self.docs = docs

    def retrieve(self, query: str, k: int = 5) -> list[str]:
        """Devuelve los top-k documentos para la query."""
        q_tokens = preprocess(query)
        scores = self.bm25.get_scores(q_tokens)
        topk_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [self.docs[i] for i in topk_idx]
