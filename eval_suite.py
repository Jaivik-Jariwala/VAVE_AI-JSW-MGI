"""
VAVE AI Evaluation Suite — Feature 3: LLM-as-Judge + Golden Dataset
Provides:
  - Golden dataset of (query, expected_keywords) pairs specific to VAVE domain
  - LLM-as-Judge scoring function (accuracy, completeness, faithfulness)
  - run_eval_suite() — full regression run against the golden set
  - Results logged to PostgreSQL eval_results table
  - /run_eval Flask endpoint wires into this module

Usage:
    from eval_suite import run_eval_suite
    results = run_eval_suite(pg_conn_func, get_response_func, gemini_judge_model)
"""
import json
import logging
import re
from datetime import datetime
from typing import Callable, Dict, List, Any, Optional

logger = logging.getLogger(__name__)

# ─── GOLDEN DATASET ───────────────────────────────────────────────────────────
# A curated set of (query, expected_keywords) pairs for regression testing.
# Add more as the product matures.

GOLDEN_SET: List[Dict] = [
    {
        "id": "G001",
        "query": "Show top 5 ideas by cost saving for Hector Plus",
        "expected_keywords": ["idea_id", "saving", "Hector", "INR"],
        "min_results": 3,
        "category": "retrieval",
    },
    {
        "id": "G002",
        "query": "What is the total saving value for Body department ideas?",
        "expected_keywords": ["total", "body", "INR", "crore"],
        "min_results": 1,
        "category": "aggregation",
    },
    {
        "id": "G003",
        "query": "List approved HVAC ideas with status OK",
        "expected_keywords": ["HVAC", "OK", "idea_id"],
        "min_results": 2,
        "category": "filtering",
    },
    {
        "id": "G004",
        "query": "Compare MG Hector vs competitor for brake system cost reduction",
        "expected_keywords": ["brake", "compare", "cost"],
        "min_results": 1,
        "category": "comparison",
    },
    {
        "id": "G005",
        "query": "Which electrical ideas have highest saving value?",
        "expected_keywords": ["electrical", "saving", "INR"],
        "min_results": 2,
        "category": "ranking",
    },
    {
        "id": "G006",
        "query": "Show ideas requiring CAE validation",
        "expected_keywords": ["CAE", "idea_id"],
        "min_results": 1,
        "category": "compliance",
    },
    {
        "id": "G007",
        "query": "What are material substitution ideas for ZS EV?",
        "expected_keywords": ["material", "ZS EV", "substitut"],
        "min_results": 1,
        "category": "domain_specific",
    },
    {
        "id": "G008",
        "query": "List ideas with investment less than 5 crore",
        "expected_keywords": ["investment", "crore", "idea_id"],
        "min_results": 1,
        "category": "financial_filter",
    },
]


# ─── LLM-AS-JUDGE ────────────────────────────────────────────────────────────

def llm_judge_score(
    query: str,
    response: str,
    expected_keywords: List[str],
    source_data: str,
    gemini_judge_model,
) -> Dict[str, Any]:
    """
    Feature 3: Use Gemini as a judge to score a VAVE AI response.
    Returns dict: {accuracy, completeness, faithfulness, reason, verdict}
    """
    import prompt_registry

    try:
        judge_prompt = prompt_registry.get(
            "llm_judge_response",
            query=query,
            response=response[:3000],
            expected_keywords=", ".join(expected_keywords),
            source_data=source_data[:500],
        )

        raw = gemini_judge_model.generate_content(judge_prompt).text.strip()

        # Parse JSON
        match = re.search(r'\{.*?\}', raw, re.DOTALL)
        if match:
            result = json.loads(match.group(0))
            return {
                "accuracy": int(result.get("accuracy", 0)),
                "completeness": int(result.get("completeness", 0)),
                "faithfulness": int(result.get("faithfulness", 0)),
                "reason": result.get("reason", ""),
                "verdict": result.get("verdict", "FAIL"),
                "raw_judge_output": raw[:500],
            }
    except Exception as e:
        logger.error(f"[EvalSuite] Judge scoring error: {e}")

    # Fallback: keyword-based scoring
    return _keyword_fallback_score(query, response, expected_keywords)


def _keyword_fallback_score(
    query: str, response: str, expected_keywords: List[str]
) -> Dict[str, Any]:
    """Fallback scoring when LLM judge unavailable: uses keyword presence."""
    resp_lower = response.lower()
    found = sum(1 for kw in expected_keywords if kw.lower() in resp_lower)
    score = round(10 * found / max(len(expected_keywords), 1))
    verdict = "PASS" if score >= 7 else "FAIL"
    return {
        "accuracy": score,
        "completeness": score,
        "faithfulness": 8 if len(response) > 50 else 3,
        "reason": f"Keyword fallback: {found}/{len(expected_keywords)} keywords found",
        "verdict": verdict,
        "raw_judge_output": "",
    }


# ─── FULL EVAL SUITE ─────────────────────────────────────────────────────────

def run_eval_suite(
    get_response_func: Callable[[str], str],
    gemini_judge_model,
    pg_conn_func: Optional[Callable] = None,
    golden_set: Optional[List[Dict]] = None,
    run_label: str = "manual",
) -> Dict[str, Any]:
    """
    Feature 3: Run the full evaluation suite.

    Args:
        get_response_func: Callable(query) -> str  (calls VAVE AI pipeline)
        gemini_judge_model: Gemini model instance for LLM-as-Judge scoring
        pg_conn_func: Optional PostgreSQL connection factory for logging results
        golden_set: Override default GOLDEN_SET
        run_label: Label for this eval run (e.g. 'daily', 'post_deploy', 'manual')

    Returns:
        {
            "run_label": ...,
            "total": N,
            "passed": N,
            "failed": N,
            "pass_rate_pct": N,
            "avg_accuracy": N,
            "avg_completeness": N,
            "avg_faithfulness": N,
            "cases": [...],
        }
    """
    cases_to_run = golden_set or GOLDEN_SET
    logger.info(f"[EvalSuite] Starting eval run '{run_label}' with {len(cases_to_run)} cases.")

    results = []
    for case in cases_to_run:
        query = case["query"]
        expected_keywords = case.get("expected_keywords", [])

        logger.info(f"[EvalSuite] Running case {case['id']}: {query[:60]}")
        try:
            response = get_response_func(query)
        except Exception as e:
            logger.error(f"[EvalSuite] Response generation failed for case {case['id']}: {e}")
            response = f"ERROR: {e}"

        score = llm_judge_score(
            query=query,
            response=response,
            expected_keywords=expected_keywords,
            source_data="",
            gemini_judge_model=gemini_judge_model,
        )

        result = {
            "case_id": case["id"],
            "category": case.get("category", "general"),
            "query": query,
            "response_snippet": response[:200],
            **score,
        }
        results.append(result)

    # ── Aggregate ────────────────────────────────────────────────────
    total = len(results)
    passed = sum(1 for r in results if r["verdict"] == "PASS")
    avg_acc = round(sum(r["accuracy"] for r in results) / max(total, 1), 2)
    avg_comp = round(sum(r["completeness"] for r in results) / max(total, 1), 2)
    avg_faith = round(sum(r["faithfulness"] for r in results) / max(total, 1), 2)
    pass_rate = round(100 * passed / max(total, 1), 1)

    summary = {
        "run_label": run_label,
        "run_at": datetime.now().isoformat(),
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "pass_rate_pct": pass_rate,
        "avg_accuracy": avg_acc,
        "avg_completeness": avg_comp,
        "avg_faithfulness": avg_faith,
        "cases": results,
    }

    logger.info(
        f"[EvalSuite] Run '{run_label}' complete: {passed}/{total} PASS "
        f"({pass_rate}%) | Acc={avg_acc} Comp={avg_comp} Faith={avg_faith}"
    )

    # ── Persist to PostgreSQL if available ───────────────────────────
    if pg_conn_func:
        _log_eval_run_to_db(pg_conn_func, summary)

    return summary


def _log_eval_run_to_db(pg_conn_func: Callable, summary: Dict):
    """Store eval run summary in PostgreSQL eval_results table."""
    try:
        conn = pg_conn_func()
        cur = conn.cursor()
        # Create table if not exists
        cur.execute("""
            CREATE TABLE IF NOT EXISTS eval_results (
                id SERIAL PRIMARY KEY,
                run_label TEXT,
                run_at TIMESTAMP,
                total INTEGER,
                passed INTEGER,
                pass_rate_pct REAL,
                avg_accuracy REAL,
                avg_completeness REAL,
                avg_faithfulness REAL,
                cases JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cur.execute(
            """INSERT INTO eval_results
               (run_label, run_at, total, passed, pass_rate_pct,
                avg_accuracy, avg_completeness, avg_faithfulness, cases)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)""",
            (
                summary["run_label"],
                summary["run_at"],
                summary["total"],
                summary["passed"],
                summary["pass_rate_pct"],
                summary["avg_accuracy"],
                summary["avg_completeness"],
                summary["avg_faithfulness"],
                json.dumps(summary["cases"]),
            ),
        )
        conn.commit()
        cur.close()
        conn.close()
        logger.info("[EvalSuite] Eval results persisted to DB.")
    except Exception as e:
        logger.error(f"[EvalSuite] DB persist error: {e}")


# ─── SCHEMA SETUP ─────────────────────────────────────────────────────────────

def ensure_eval_table(pg_conn_func: Callable):
    """Create eval_results table at startup."""
    try:
        conn = pg_conn_func()
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS eval_results (
                id SERIAL PRIMARY KEY,
                run_label TEXT,
                run_at TIMESTAMP,
                total INTEGER,
                passed INTEGER,
                pass_rate_pct REAL,
                avg_accuracy REAL,
                avg_completeness REAL,
                avg_faithfulness REAL,
                cases JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        cur.close()
        conn.close()
        logger.info("[EvalSuite] eval_results table ensured.")
    except Exception as e:
        logger.error(f"[EvalSuite] Table creation error: {e}")
