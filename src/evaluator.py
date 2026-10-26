import json
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

from src.rag_engine import RAGEngine, RAGResponse

@dataclass
class TestCaseResult:
    test_id: str
    category: str
    question: str
    is_answerable: bool
    expected_document: Optional[str]
    retrieved_documents: List[str]
    retrieval_hit: bool  # Was target doc in top-K?
    answer_generated: str
    groundedness_passed: bool  # Did it correctly answer or correctly reject?
    citation_passed: bool  # Did it cite expected document if answerable?
    verdict: str  # "PASS" or "FAIL"
    explanation: str

@dataclass
class BenchmarkSummary:
    total_tests: int
    passed_tests: int
    failed_tests: int
    accuracy_percentage: float
    retrieval_hit_rate: float
    unanswerable_rejection_rate: float
    answerable_accuracy: float
    results: List[TestCaseResult]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_tests": self.total_tests,
            "passed_tests": self.passed_tests,
            "failed_tests": self.failed_tests,
            "accuracy_percentage": round(self.accuracy_percentage, 1),
            "retrieval_hit_rate": round(self.retrieval_hit_rate, 1),
            "unanswerable_rejection_rate": round(self.unanswerable_rejection_rate, 1),
            "answerable_accuracy": round(self.answerable_accuracy, 1),
        }

class RAGEvaluator:
    """Benchmark runner evaluating groundedness, retrieval precision, and unanswerable detection."""

    def __init__(self, rag_engine: RAGEngine):
        self.rag_engine = rag_engine

    def load_benchmark(self, json_path: Path) -> List[Dict[str, Any]]:
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def run_benchmark(self, test_cases: List[Dict[str, Any]], progress_callback=None) -> BenchmarkSummary:
        results: List[TestCaseResult] = []
        retrieval_hits = 0
        answerable_count = 0
        answerable_success = 0
        unanswerable_count = 0
        unanswerable_success = 0

        for idx, tc in enumerate(test_cases):
            if progress_callback:
                progress_callback(idx + 1, len(test_cases), tc.get("question", ""))

            t_id = tc.get("id", f"TEST-{idx+1}")
            category = tc.get("category", "General")
            question = tc.get("question", "")
            is_answerable = tc.get("is_answerable", True)
            expected_doc = tc.get("expected_document")
            expected_facts = tc.get("expected_key_facts", [])

            # Run through RAG engine
            response: RAGResponse = self.rag_engine.answer_question(question=question)
            retrieved_docs = [chunk.filename for chunk, _ in response.retrieved_chunks]

            # 1. Retrieval check
            if expected_doc:
                retrieval_hit = expected_doc in retrieved_docs
                if retrieval_hit:
                    retrieval_hits += 1
            else:
                retrieval_hit = True  # Not applicable for unanswerable

            # 2. Logic & Groundedness check
            if is_answerable:
                answerable_count += 1
                # Must be found and contain at least one expected fact
                ans_text = response.answer.lower()
                facts_found = any(f.lower() in ans_text for f in expected_facts) if expected_facts else True
                groundedness_passed = response.found_in_docs and facts_found

                # Citation check
                citation_passed = False
                if expected_doc:
                    citation_passed = any(c.filename == expected_doc for c in response.citations) or (expected_doc in response.answer)
                else:
                    citation_passed = len(response.citations) > 0

                verdict = "PASS" if (groundedness_passed and retrieval_hit and citation_passed) else "FAIL"
                if verdict == "PASS":
                    answerable_success += 1
                
                explanation = (
                    f"Retrieved: {', '.join(set(retrieved_docs)) or 'None'}. "
                    f"Found in Docs: {response.found_in_docs}. "
                    f"Facts Match: {facts_found}. "
                    f"Citation Match: {citation_passed}."
                )

            else:
                unanswerable_count += 1
                # Must NOT say found in docs or must trigger unanswerable phrase
                rejected_properly = (not response.found_in_docs) or ("could not find" in response.answer.lower())
                groundedness_passed = rejected_properly
                citation_passed = True  # Unanswerables should not produce citations
                verdict = "PASS" if rejected_properly else "FAIL"
                if verdict == "PASS":
                    unanswerable_success += 1

                explanation = (
                    f"Correctly detected lack of evidence: {rejected_properly}. "
                    f"Top similarity score: {response.retrieved_chunks[0][1]:.1%}" if response.retrieved_chunks else "No chunks retrieved."
                )

            results.append(TestCaseResult(
                test_id=t_id,
                category=category,
                question=question,
                is_answerable=is_answerable,
                expected_document=expected_doc,
                retrieved_documents=list(set(retrieved_docs)),
                retrieval_hit=retrieval_hit,
                answer_generated=response.answer,
                groundedness_passed=groundedness_passed,
                citation_passed=citation_passed,
                verdict=verdict,
                explanation=explanation
            ))

        total = len(results)
        passed = sum(1 for r in results if r.verdict == "PASS")
        failed = total - passed

        retrieval_hit_rate = (retrieval_hits / answerable_count * 100) if answerable_count > 0 else 100.0
        unanswerable_rate = (unanswerable_success / unanswerable_count * 100) if unanswerable_count > 0 else 100.0
        answerable_acc = (answerable_success / answerable_count * 100) if answerable_count > 0 else 100.0
        overall_acc = (passed / total * 100) if total > 0 else 0.0

        return BenchmarkSummary(
            total_tests=total,
            passed_tests=passed,
            failed_tests=failed,
            accuracy_percentage=overall_acc,
            retrieval_hit_rate=retrieval_hit_rate,
            unanswerable_rejection_rate=unanswerable_rate,
            answerable_accuracy=answerable_acc,
            results=results
        )
