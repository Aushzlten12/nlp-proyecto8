from preprocess_index import BM25Indexer
from retriever import Retriever
from generator import Generator
from evaluator import compute_bleu, manual_review


def load_corpus(path: str) -> list[str]:
    with open(path, encoding="utf-8") as f:
        # Cada párrafo separado por línea en blanco como en el txt
        return [p.strip() for p in f.read().split("\n\n") if p.strip()]


def main():

    docs = load_corpus("data/wiki_paragraphs.txt")
    indexer = BM25Indexer(docs)
    bm25, docs = indexer.get_index()

    # Instancia retriever y generator
    retriever = Retriever(bm25, docs)
    generator = Generator("gpt2")

    # prueba
    examples = [
        ("Who wrote the novel Dune?", "Frank Herbert"),
        ("When did Columbus discover America?", "1492"),
    ]

    predictions, references, queries = [], [], []
    for query, ref in examples:
        topk = retriever.retrieve(query, k=5)
        context = "\n\n".join(topk)
        pred = generator.generate(context, query)
        print(f">>> Query: {query}\n→ {pred}\n")
        predictions.append(pred)
        references.append(ref)
        queries.append(query)

    #  BLEU
    bleu_score = compute_bleu(predictions, references)
    print(f"BLEU score: {bleu_score:.2f}\n")

    # Evaluación de fluidez
    manual_samples = list(zip(queries, references, predictions))
    manual_review(manual_samples)


if __name__ == "__main__":
    main()
