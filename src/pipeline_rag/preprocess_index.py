import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from rank_bm25 import BM25Okapi

nltk.download("punkt")
nltk.download("stopwords")

stop_words = set(stopwords.words("english"))


def preprocess(text: str):
    """Limpia, tokeniza y quita stopwords."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    tokens = word_tokenize(text)
    return [t for t in tokens if t not in stop_words]


class BM25Indexer:
    def __init__(self, docs: list[str]):
        # indice BM25
        self.docs = docs
        corpus = [preprocess(d) for d in docs]
        self.bm25 = BM25Okapi(corpus)

    def get_index(self):
        return self.bm25, self.docs
