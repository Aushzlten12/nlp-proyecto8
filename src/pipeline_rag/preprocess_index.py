from typing import List
from nltk.tokenize import word_tokenize
from rank_bm25 import BM25Okapi
import nltk

nltk.download("punkt")


def load_documents(file_path: str) -> List[str]:
    with open(file_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines() if line.strip()]


def preprocess_documents(docs: List[str]) -> List[List[str]]:
    return [word_tokenize(doc.lower()) for doc in docs]


def build_bm25(tokenized_docs: List[List[str]]) -> BM25Okapi:
    return BM25Okapi(tokenized_docs)
