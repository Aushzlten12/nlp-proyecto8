from typing import List
from nltk.tokenize import word_tokenize
from rank_bm25 import BM25Okapi


class BM25Retriever:
    def __init__(self, bm25: BM25Okapi, original_docs: List[str], k: int):
        self.bm25 = bm25
        self.original_docs = original_docs
        self.k = k

    def retrieve(self, query: str) -> List[str]:
        tokenized_query = word_tokenize(query.lower())
        return self.bm25.get_top_n(tokenized_query, self.original_docs, n=self.k)
