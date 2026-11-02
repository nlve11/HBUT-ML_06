import os
import re
from typing import Protocol

from schemas import SearchHit


SYSTEM_PROMPT = (
    "你是严格依据资料回答的课程助手。只能使用上下文，不得补充外部知识；"
    "每个事实后写对应的[编号]。资料不足时只回答：资料中没有足够依据，无法回答。"
)


class Generator(Protocol):
    def generate(self, question: str, context: str, hits: list[SearchHit]) -> str: ...


class ExtractiveGenerator:
    """确定式抽取基线，不是大语言模型。"""

    def generate(self, question: str, context: str, hits: list[SearchHit]) -> str:
        sentences = []
        for i, hit in enumerate(hits[:2], start=1):
            first = re.split(
                r"(?<=[。！？])", hit.chunk.text, maxsplit=1
            )[0].strip()
            sentences.append(f"{first} [{i}]")
        return "抽取式基线（非大语言模型）：\n" + "\n".join(sentences)


class ApiGenerator:
    def generate(self, question: str, context: str, hits: list[SearchHit]) -> str:
        key = os.environ.get("LLM_API_KEY", "").strip()
        base_url = os.environ.get("LLM_BASE_URL", "").strip()
        model = os.environ.get("LLM_MODEL", "").strip()
        if not (key and base_url and model):
            raise RuntimeError(
                "API 模式需要 LLM_API_KEY/LLM_BASE_URL/LLM_MODEL 环境变量")

        from openai import OpenAI
        client = OpenAI(api_key=key, base_url=base_url,
                        timeout=90.0, max_retries=2)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": SYSTEM_PROMPT},
                      {"role": "user",
                       "content": f"上下文：\n{context}\n\n问题：{question}"}],
            temperature=0, max_tokens=300,
        )
        return (response.choices[0].message.content or "").strip()


def get_generator(mode: str) -> Generator:
    if mode == "extractive":
        return ExtractiveGenerator()
    if mode == "api":
        return ApiGenerator()
    if mode == "local":
        from local_qwen import LocalQwenGenerator
        return LocalQwenGenerator()
    raise ValueError("mode 只能是 extractive、api 或 local")
