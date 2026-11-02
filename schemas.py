from pydantic import BaseModel, Field


class Chunk(BaseModel):
    chunk_id: str
    source: str
    title: str
    paragraph: int
    char_start: int
    char_end: int
    text: str = Field(min_length=1)


class SearchHit(BaseModel):
    chunk: Chunk
    score: float


class RagResult(BaseModel):
    question: str
    mode: str
    answer: str
    refused: bool
    top_score: float
    sources: list[str]
    hits: list[SearchHit]
