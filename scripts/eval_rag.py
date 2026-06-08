"""RAG evaluation script — run golden Q&A pairs against the pipeline."""
import json
import sys
from typing import List

GOLDEN_SET = [
    {
        "question": "What is the main topic of the document?",
        "expected_keywords": ["topic", "about"],
    },
    {
        "question": "Summarize the key points.",
        "expected_keywords": ["summary", "key", "point"],
    },
]


def evaluate_answer(answer: str, expected_keywords: List[str]) -> dict:
    answer_lower = answer.lower()
    hits = [kw for kw in expected_keywords if kw in answer_lower]
    score = len(hits) / len(expected_keywords) if expected_keywords else 0
    return {"score": score, "hits": hits, "total_keywords": len(expected_keywords)}


def run_evaluation(user_id: int = 0):
    from app.core.rag import ask_question

    results = []
    for item in GOLDEN_SET:
        try:
            response = ask_question(user_id, item["question"])
            eval_result = evaluate_answer(response["answer"], item["expected_keywords"])
            results.append({
                "question": item["question"],
                "answer_preview": response["answer"][:200],
                "sources_count": len(response.get("sources", [])),
                **eval_result,
            })
        except Exception as e:
            results.append({"question": item["question"], "error": str(e), "score": 0})

    avg_score = sum(r.get("score", 0) for r in results) / len(results) if results else 0
    report = {"average_score": avg_score, "results": results}
    print(json.dumps(report, indent=2))
    return 0 if avg_score >= 0.5 else 1


if __name__ == "__main__":
    sys.exit(run_evaluation())
