from context import build_context, finalize_answer
from generators import get_generator
from retrieval import Retriever
from schemas import RagResult
from settings import REFUSAL_TEXT, SCORE_THRESHOLD


def answer_question(retriever: Retriever, question: str, mode: str,
                    top_k: int = 3, threshold: float = SCORE_THRESHOLD) -> RagResult:
    hits = retriever.search(question, top_k)
    top_score = hits[0].score if hits else 0.0
    if not hits or top_score < threshold:
        return RagResult(question=question, mode=mode,
                         answer=REFUSAL_TEXT, refused=True,
                         top_score=top_score, sources=[], hits=hits)
    context = build_context(hits)
    raw = get_generator(mode).generate(question, context, hits)
    answer, sources = finalize_answer(raw, hits)
    return RagResult(question=question, mode=mode, answer=answer,
                     refused=False, top_score=top_score, sources=sources, hits=hits)
