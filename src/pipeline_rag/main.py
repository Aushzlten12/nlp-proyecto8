from preprocess_index import load_documents, preprocess_documents, build_bm25
from retriever import BM25Retriever
from generator import GPT2Generator
from evaluator import Evaluator
import csv

# carga los documentos que son lineas de texto por ahora
documents = load_documents("data/corpus.txt")
preprocessed_docs = preprocess_documents(documents)
bm25 = build_bm25(preprocessed_docs)

# carga querys y references
queries = load_documents("data/queries.txt")
references = load_documents("data/references.txt")

# diferentes top-k values
top_k_values = [1, 2, 3, 5]

results = []

generator = GPT2Generator(max_tokens=50, temperature=0.7, top_p=0.8)
evaluator = Evaluator()

for query, reference in zip(queries, references):
    print(f"\n=== Consulta: {query} ===")
    for k in top_k_values:
        print(f"\n--- Top-{k} Documentos ---")
        retriever = BM25Retriever(bm25, documents, k)
        top_docs = retriever.retrieve(query)

        for i, doc in enumerate(top_docs):
            print(f"[{i+1}] {doc}")

        response = generator.generate(query, top_docs)

        print("\n--- Respuesta Generada ---")
        print(response)

        bleu_score = evaluator.compute_bleu([reference], [response])
        print("\n--- Evaluación BLEU ---")
        print(f"BLEU: {bleu_score:.4f}")

        results.append([query, k, response, reference, bleu_score])

# en CSV
with open("data/results.csv", mode="w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["query", "top_k", "response", "reference", "bleu"])
    writer.writerows(results)
