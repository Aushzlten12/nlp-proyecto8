from sacrebleu import corpus_bleu


def compute_bleu(predictions: list[str], references: list[str]) -> float:
    """
    Calcula BLEU con SacreBLEU.
    """
    refs = [[r] for r in references]
    bleu = corpus_bleu(predictions, refs)
    return bleu.score


def manual_review(samples: list[tuple[str, str, str]]):
    """
    evaluación manual de fluidez.
    """
    for q, ref, pred in samples:
        print("=== Consulta ===\n", q)
        print("Referencia:\n", ref)
        print("Generación:\n", pred)
        print(
            "\nPUNTUAR fluidez 1–5 según coherencia, gramática, naturalidad.\n"
            + "-" * 40
        )
