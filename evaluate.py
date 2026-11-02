import json
from pathlib import Path

from app import build_retriever
from pipeline import answer_question
from settings import OUTPUT_DIR


# 只保存“正确来源”，不保存答案句子，避免用答案文字泄漏评测规则。
CASES = [
    {"q": "课程共有多少课时，分成几次实验？", "gold": ["course_overview.md"], "scope": True},
    {"q": "程序失败时应先检查哪些内容？", "gold": ["course_overview.md"], "scope": True},
    {"q": "没有 GPU 的十六 GB 电脑怎样配置 Python 环境？", "gold": ["course_overview.md"], "scope": True},
    {"q": "提交包必须包含哪些文件？", "gold": ["submission_rules.md"], "scope": True},
    {"q": "为什么不能提交虚拟环境和 API 密钥？", "gold": ["submission_rules.md", "api_safety.md"], "scope": True},
    {"q": "RAG 从读取资料到回答经过哪些步骤？", "gold": ["rag_principles.md"], "scope": True},
    {"q": "嵌入模型输出多少维，为什么点积等于余弦相似度？", "gold": ["rag_principles.md"], "scope": True},
    {"q": "如何区分检索错误和生成错误？", "gold": ["rag_principles.md"], "scope": True},
    {"q": "明天香港会不会下雨？", "gold": [], "scope": False},
    {"q": "学校一等奖学金金额是多少？", "gold": [], "scope": False},
]


def retrieval_metrics(retriever, k: int) -> dict:
    hit_sum = 0.0
    recall_sum = 0.0
    rows = []
    for case in CASES:
        if not case["scope"]:
            continue
        hits = retriever.search(case["q"], k)
        got = {h.chunk.source for h in hits}
        gold = set(case["gold"])
        hit = float(bool(got & gold))
        recall = len(got & gold) / len(gold)
        hit_sum += hit
        recall_sum += recall
        rows.append({"question": case["q"],
                     "retrieved": sorted(got),
                     "hit": hit, "recall": recall})
    n = len(rows)
    return {"k": k,
            "hit_at_k": hit_sum / n,
            "recall_at_k": recall_sum / n,
            "rows": rows}


def end_to_end_metrics(retriever) -> dict:
    rows = []
    refused_correct = 0
    refused_total = 0
    cited_complete = 0
    cited_total = 0

    for case in CASES:
        result = answer_question(retriever, case["q"], "extractive", top_k=3)
        got = {h.chunk.source for h in result.hits}
        gold = set(case["gold"])

        markers = [f"[{i}]" for i in range(1, len(result.hits) + 1)]
        present = [m for m in markers if m in result.answer]
        valid_markers = present

        # 拒答评价：仅资料外样本
        if not case["scope"]:
            refused_total += 1
            if result.refused:
                refused_correct += 1

        # 引用评价：非拒答样本检查每个来源编号是否出现在答案中
        if not result.refused:
            cited_total += 1
            if len(present) == len(markers):
                cited_complete += 1

        # 分层归因
        if case["scope"] and not (got & gold):
            diagnosis = "RETRIEVAL_ERROR"
        elif case["scope"] and result.refused:
            diagnosis = "GATING_ERROR"
        elif not result.refused and not valid_markers:
            diagnosis = "GENERATION_OR_POSTPROCESS_ERROR"
        else:
            diagnosis = "PASS"

        rows.append({"question": case["q"], "scope": case["scope"],
                     "retrieved": sorted(got), "refused": result.refused,
                     "diagnosis": diagnosis})

    return {
        "refusal_accuracy": (refused_correct / refused_total) if refused_total else 0.0,
        "citation_completeness": (cited_complete / cited_total) if cited_total else 0.0,
        "rows": rows,
    }


def main() -> None:
    retriever = build_retriever()
    report = {"retrieval": [retrieval_metrics(retriever, k) for k in (1, 3, 5)],
              "end_to_end": end_to_end_metrics(retriever)}
    OUTPUT_DIR.mkdir(exist_ok=True)
    Path(OUTPUT_DIR / "evaluation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    for item in report["retrieval"]:
        print(f"K={item['k']}: Hit@K={item['hit_at_k']:.3f}, Recall@K={item['recall_at_k']:.3f}")
    print(f"refusal_accuracy={report['end_to_end']['refusal_accuracy']:.3f}")
    print(f"citation_completeness={report['end_to_end']['citation_completeness']:.3f}")


if __name__ == "__main__":
    main()
