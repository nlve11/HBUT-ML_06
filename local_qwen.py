import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from generators import SYSTEM_PROMPT
from schemas import SearchHit
from settings import LOCAL_MODEL


class LocalQwenGenerator:
    def __init__(self) -> None:
        # float32 在 CPU 上兼容性最好；0.5B 模型适合 16GB 内存基线。
        self.tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL)
        self.model = AutoModelForCausalLM.from_pretrained(
            LOCAL_MODEL, dtype=torch.float32
        ).to("cpu").eval()

    def generate(self, question: str, context: str, hits: list[SearchHit]) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",
             "content": f"上下文：\n{context}\n\n问题：{question}"},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        with torch.inference_mode():
            output = self.model.generate(
                **inputs, max_new_tokens=180, do_sample=False,
                repetition_penalty=1.05,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_tokens = output[0, inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(
            new_tokens, skip_special_tokens=True).strip()
