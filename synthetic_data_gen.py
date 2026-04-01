"""
VAVE AI Synthetic Data Generator — Feature 15
Generates (question, answer) training pairs from the VAVE ideas database
for fine-tuning Mistral 7B / Llama 3 via LoRA/QLoRA.

Usage:
    python synthetic_data_gen.py --output training_data.jsonl --n_pairs 5

Or import and call:
    from synthetic_data_gen import generate_all_training_data
    count = generate_all_training_data(pg_conn_func, gemini_model, output_path)
"""
import json
import logging
import re
import argparse
import os
from pathlib import Path
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

CATEGORIES = ["technical", "financial", "compliance", "risk", "comparison"]


def _fetch_ideas(pg_conn_func: Callable, limit: int = 2000) -> List[Dict]:
    """Fetch idea records from PostgreSQL."""
    try:
        conn = pg_conn_func()
        cur = conn.cursor()
        cur.execute("""
            SELECT idea_id, cost_reduction_idea, dept, mgi_carline,
                   saving_value_inr, status, way_forward,
                   homologation_required, cae_required
            FROM ideas
            WHERE cost_reduction_idea IS NOT NULL
            LIMIT %s
        """, (limit,))
        cols = [d[0] for d in cur.description]
        rows = [dict(zip(cols, r)) for r in cur.fetchall()]
        cur.close()
        conn.close()
        logger.info(f"[SyntheticGen] Fetched {len(rows)} ideas from DB.")
        return rows
    except Exception as e:
        logger.error(f"[SyntheticGen] DB fetch error: {e}")
        return []


def generate_pairs_for_idea(
    idea: Dict,
    gemini_model,
    n_pairs: int = 5,
) -> List[Dict]:
    """
    Generate n_pairs Q&A training pairs for a single idea using Gemini.
    """
    import prompt_registry

    try:
        prompt = prompt_registry.get(
            "synthetic_qa_generator",
            idea_id=idea.get("idea_id", "N/A"),
            cost_reduction_idea=idea.get("cost_reduction_idea", "")[:500],
            dept=idea.get("dept", "N/A"),
            mgi_carline=idea.get("mgi_carline", "N/A"),
            saving_value_inr=idea.get("saving_value_inr", "N/A"),
            status=idea.get("status", "N/A"),
            way_forward=str(idea.get("way_forward", "N/A"))[:300],
            n_pairs=n_pairs,
        )

        response = gemini_model.generate_content(prompt)
        raw = response.text.strip()

        # Extract JSON array
        match = re.search(r'\[.*?\]', raw, re.DOTALL)
        if not match:
            return []

        pairs = json.loads(match.group(0))
        # Add metadata fields for fine-tuning
        for p in pairs:
            p["idea_id"] = idea.get("idea_id", "")
            p["dept"] = idea.get("dept", "")
            p["carline"] = idea.get("mgi_carline", "")
        return pairs

    except Exception as e:
        logger.error(f"[SyntheticGen] Pair generation failed for {idea.get('idea_id')}: {e}")
        return []


def generate_all_training_data(
    pg_conn_func: Callable,
    gemini_model,
    output_path: str = "training_data_vave.jsonl",
    ideas_limit: int = 1000,
    pairs_per_idea: int = 5,
    format: str = "alpaca",        # 'alpaca' | 'sharegpt' | 'dpo'
) -> int:
    """
    Feature 15: Main pipeline.
    1. Fetch ideas from PostgreSQL
    2. For each idea, generate pairs_per_idea Q&A pairs using Gemini
    3. Write as JSONL in the specified fine-tuning format
    Returns total pairs written.
    """
    ideas = _fetch_ideas(pg_conn_func, limit=ideas_limit)
    if not ideas:
        logger.error("[SyntheticGen] No ideas retrieved — cannot generate training data.")
        return 0

    total_written = 0
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path_obj, "w", encoding="utf-8") as f:
        for i, idea in enumerate(ideas):
            pairs = generate_pairs_for_idea(idea, gemini_model, n_pairs=pairs_per_idea)
            for pair in pairs:
                q = pair.get("q", "")
                a = pair.get("a", "")
                if not q or not a:
                    continue

                if format == "alpaca":
                    record = {
                        "instruction": q,
                        "input": f"Idea ID: {pair.get('idea_id','')} | Dept: {pair.get('dept','')} | Carline: {pair.get('carline','')}",
                        "output": a,
                        "category": pair.get("category", "general"),
                    }
                elif format == "sharegpt":
                    record = {
                        "conversations": [
                            {"from": "human", "value": q},
                            {"from": "gpt", "value": a},
                        ]
                    }
                elif format == "dpo":
                    record = {
                        "prompt": q,
                        "chosen": a,
                        "rejected": "I don't know.",   # Minimal negative
                    }
                else:
                    record = {"prompt": q, "completion": a}

                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                total_written += 1

            if (i + 1) % 50 == 0:
                logger.info(
                    f"[SyntheticGen] Processed {i+1}/{len(ideas)} ideas → {total_written} pairs so far"
                )

    logger.info(
        f"[SyntheticGen] Done: {total_written} training pairs written to {output_path}"
    )
    return total_written


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from dotenv import load_dotenv
    load_dotenv()

    import google.generativeai as genai
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY", ""))

    parser = argparse.ArgumentParser(description="VAVE AI Synthetic Training Data Generator")
    parser.add_argument("--output", default="training_data_vave.jsonl", help="Output JSONL file path")
    parser.add_argument("--n_pairs", type=int, default=5, help="Q&A pairs per idea")
    parser.add_argument("--limit", type=int, default=500, help="Max ideas to process")
    parser.add_argument("--format", choices=["alpaca", "sharegpt", "dpo", "raw"], default="alpaca")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # Import app's pg_conn_func
    from app import get_db_connection
    model = genai.GenerativeModel("gemini-3.1-flash-lite")

    count = generate_all_training_data(
        pg_conn_func=get_db_connection,
        gemini_model=model,
        output_path=args.output,
        ideas_limit=args.limit,
        pairs_per_idea=args.n_pairs,
        format=args.format,
    )
    print(f"\n✅ Generated {count} training pairs → {args.output}")
