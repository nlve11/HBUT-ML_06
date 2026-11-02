import argparse
import time

from corpus import load_chunks
from embedding import Embedder
from pipeline import answer_question
from retrieval import Retriever
from settings import BATCH_SIZE, CHUNK_OVERLAP, CHUNK_SIZE, DOCS_DIR, EMBED_MODEL, OUTPUT_DIR


def build_retriever() -> Retriever:
    chunks = load_chunks(DOCS_DIR, CHUNK_SIZE, CHUNK_OVERLAP)
    print(f"documents={len({c.source for c in chunks})}, chunks={len(chunks)}")
    embedder = Embedder(EMBED_MODEL, BATCH_SIZE)
    return Retriever.build(chunks, embedder, OUTPUT_DIR)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", default="这门课有多少次实验，每次多少课时？")
    parser.add_argument("--mode", choices=["extractive", "api", "local"], default="extractive")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    started = time.perf_counter()
    result = answer_question(build_retriever(), args.question, args.mode, args.top_k)
    print(result.model_dump_json(indent=2))
    print(f"elapsed_seconds={time.perf_counter() - started:.2f}")


if __name__ == "__main__":
    main()
