import re
from pathlib import Path

from schemas import Chunk


def read_documents(docs_dir: Path) -> list[tuple[Path, str]]:
    """按文件名排序读取 Markdown/文本，保证每次索引顺序一致。"""
    paths = sorted([*docs_dir.glob("*.md"), *docs_dir.glob("*.txt")])
    documents = []
    for path in paths:
        text = path.read_text(encoding="utf-8").strip()
        if text:
            documents.append((path, text))
    if not documents:
        raise ValueError(f"知识库为空：{docs_dir}")
    return documents


def split_document(path: Path, text: str, size: int, overlap: int) -> list[Chunk]:
    """先按空行分段；过长段落再用带重叠的字符窗口切分。"""
    if not (0 <= overlap < size):
        raise ValueError(f"overlap 必须满足 0 <= overlap < size，当前 overlap={overlap}, size={size}")

    paragraphs = re.split(r"\n\s*\n", text)
    chunks: list[Chunk] = []
    title = path.stem
    pending_h2 = ""
    paragraph_no = 0

    for raw in paragraphs:
        block = raw.strip()
        if not block:
            continue
        # 一级标题只作元数据，不进入片段正文。
        if block.startswith("# "):
            title = block[2:].strip()
            continue
        # 二级标题暂存，与紧随其后的正文段落合并。
        if block.startswith("## "):
            pending_h2 = block
            continue

        paragraph_no += 1
        paragraph = (pending_h2 + "\n" + block) if pending_h2 else block
        pending_h2 = ""

        starts = range(0, len(paragraph), size - overlap)
        for window_no, start in enumerate(starts, start=1):
            piece = paragraph[start:start + size].strip()
            if not piece:
                continue
            end = min(start + size, len(paragraph))
            chunk_id = f"{path.stem}-p{paragraph_no:02d}-w{window_no:02d}"
            chunks.append(Chunk(chunk_id=chunk_id, source=path.name,
                                title=title, paragraph=paragraph_no,
                                char_start=start, char_end=end, text=piece))
            if start + size >= len(paragraph):
                break

    return chunks


def load_chunks(docs_dir: Path, size: int, overlap: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path, text in read_documents(docs_dir):
        chunks.extend(split_document(path, text, size, overlap))
    return chunks
