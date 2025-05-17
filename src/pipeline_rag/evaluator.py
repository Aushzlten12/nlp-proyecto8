from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize


class Evaluator:
    def compute_bleu(self, references: list, candidates: list) -> float:
        tokenized_refs = [[word_tokenize(ref.lower())] for ref in references]
        tokenized_cands = [word_tokenize(cand.lower()) for cand in candidates]
        chencherry = SmoothingFunction()
        return sentence_bleu(
            tokenized_refs[0], tokenized_cands[0], smoothing_function=chencherry.method1
        )
