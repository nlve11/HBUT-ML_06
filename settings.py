from pathlib import Path

# 所有路径都从当前文件推导，避免依赖启动终端所在目录。
ROOT = Path(__file__).resolve().parent
DOCS_DIR = ROOT / "docs"
OUTPUT_DIR = ROOT / "outputs"

EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
LOCAL_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

# 过长段落采用字符滑窗；相邻窗口共享 60 个字符。
CHUNK_SIZE = 260
CHUNK_OVERLAP = 60
BATCH_SIZE = 16
DEFAULT_TOP_K = 3

# 该值必须结合本实验评测集校准，不能照搬到其他知识库。
SCORE_THRESHOLD = 0.34
REFUSAL_TEXT = "资料中没有足够依据，无法回答。"
