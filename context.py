import re

from schemas import SearchHit


def build_context(hits: list[SearchHit]) -> str:
    """把检索片段编号为 [1]、[2]……供生成器引用。"""
    blocks = []
    for number, hit in enumerate(hits, start=1):
        blocks.append(
            f"[{number}] source={hit.chunk.source}; "
            f"chunk={hit.chunk.chunk_id}; score={hit.score:.4f}\n"
            f"{hit.chunk.text}")
    return "\n\n".join(blocks)


def finalize_answer(raw: str, hits: list[SearchHit]) -> tuple[str, list[str]]:
    """删除越界引用，并追加机器可核对的来源映射。"""
    valid = {str(i) for i in range(1, len(hits) + 1)}
    cleaned = re.sub(
        r"\[(\d+)\]",
        lambda m: m.group(0) if m.group(1) in valid else "",
        raw,
    )
    sources = [f"[{i}] {h.chunk.source}#{h.chunk.chunk_id}"
               for i, h in enumerate(hits, start=1)]
    return cleaned.strip() + "\n\n来源：\n" + "\n".join(sources), sources
