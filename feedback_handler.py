"""
VAVE AI Feedback Handler — Feature 7: Human-in-the-Loop (HITL) Feedback
Provides:
  - POST /feedback  → save thumbs up/down + optional correction to PostgreSQL
  - Feedback table schema creation
  - Weekly analytics job (call from jobs.py or cron)
  - Data Flywheel: exports feedback as fine-tuning training pairs
"""
import json
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)


# ─── DB SCHEMA ────────────────────────────────────────────────────────────────

FEEDBACK_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS feedback (
    id SERIAL PRIMARY KEY,
    username TEXT NOT NULL,
    query TEXT,
    response TEXT,
    rating INTEGER CHECK (rating IN (-1, 1)),
    correction TEXT,
    prompt_version TEXT,
    model_name TEXT,
    session_id TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

FEEDBACK_STATS_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_feedback_username ON feedback(username);
CREATE INDEX IF NOT EXISTS idx_feedback_rating ON feedback(rating);
CREATE INDEX IF NOT EXISTS idx_feedback_created ON feedback(created_at);
"""


def ensure_feedback_table(pg_conn_func):
    """Create feedback table if it doesn't exist. Call at app startup."""
    try:
        conn = pg_conn_func()
        cur = conn.cursor()
        cur.execute(FEEDBACK_TABLE_SQL)
        cur.execute(FEEDBACK_STATS_INDEX_SQL)
        conn.commit()
        cur.close()
        conn.close()
        logger.info("[FeedbackHandler] Feedback table ensured.")
    except Exception as e:
        logger.error(f"[FeedbackHandler] Table creation error: {e}")


# ─── SAVE FEEDBACK ────────────────────────────────────────────────────────────

def save_feedback(
    pg_conn_func,
    username: str,
    query: str,
    response: str,
    rating: int,                  # +1 = thumbs up, -1 = thumbs down
    correction: Optional[str] = None,
    prompt_version: Optional[str] = None,
    model_name: Optional[str] = None,
    session_id: Optional[str] = None,
) -> bool:
    """
    Persist user feedback to PostgreSQL.
    Called from the /feedback Flask endpoint.

    Returns True on success, False on failure.
    """
    if rating not in (-1, 1):
        logger.warning(f"[FeedbackHandler] Invalid rating: {rating}. Must be -1 or 1.")
        return False

    try:
        conn = pg_conn_func()
        cur = conn.cursor()
        cur.execute(
            """INSERT INTO feedback
               (username, query, response, rating, correction, prompt_version,
                model_name, session_id, created_at)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)""",
            (
                username,
                query[:2000] if query else None,
                response[:5000] if response else None,
                rating,
                correction[:2000] if correction else None,
                prompt_version,
                model_name,
                session_id,
                datetime.now(),
            ),
        )
        conn.commit()
        cur.close()
        conn.close()
        logger.info(
            f"[FeedbackHandler] Saved feedback from '{username}': rating={rating}"
        )
        return True
    except Exception as e:
        logger.error(f"[FeedbackHandler] Save error: {e}")
        return False


# ─── ANALYTICS ────────────────────────────────────────────────────────────────

def get_feedback_stats(pg_conn_func, days: int = 30) -> Dict[str, Any]:
    """
    Return feedback analytics for the admin dashboard.
    Covers: total counts, thumbs up/down ratio, worst-rated queries.
    """
    try:
        conn = pg_conn_func()
        cur = conn.cursor()

        cur.execute(
            """SELECT
               COUNT(*) AS total,
               SUM(CASE WHEN rating = 1  THEN 1 ELSE 0 END) AS thumbs_up,
               SUM(CASE WHEN rating = -1 THEN 1 ELSE 0 END) AS thumbs_down,
               COUNT(correction) AS corrections_given
               FROM feedback
               WHERE created_at >= NOW() - INTERVAL '%s days'""",
            (days,),
        )
        row = cur.fetchone()

        cur.execute(
            """SELECT query, rating, correction
               FROM feedback
               WHERE rating = -1
               ORDER BY created_at DESC
               LIMIT 10""",
        )
        worst = cur.fetchall()
        cur.close()
        conn.close()

        total = row[0] or 0
        thumbs_up = row[1] or 0
        sat_rate = round(thumbs_up / total * 100, 1) if total > 0 else 0

        return {
            "total_feedback": total,
            "thumbs_up": thumbs_up,
            "thumbs_down": row[2] or 0,
            "corrections_given": row[3] or 0,
            "satisfaction_rate_pct": sat_rate,
            "recent_bad_queries": [
                {"query": r[0], "rating": r[1], "correction": r[2]}
                for r in worst
            ],
            "period_days": days,
        }
    except Exception as e:
        logger.error(f"[FeedbackHandler] Stats error: {e}")
        return {}


# ─── DATA FLYWHEEL — Export for Fine-Tuning ───────────────────────────────────

def export_feedback_as_training_data(
    pg_conn_func,
    output_path: str = "training_data_from_feedback.jsonl",
    min_rating: int = 1,
) -> int:
    """
    Export high-rated feedback + corrections as JSONL training pairs
    for LoRA/QLoRA fine-tuning (Feature 15 integration).

    Format: {"prompt": "...", "completion": "..."} per line (Alpaca/RLHF format)
    Returns number of pairs exported.
    """
    try:
        conn = pg_conn_func()
        cur = conn.cursor()
        # Export thumbs-up responses as positive pairs
        cur.execute(
            """SELECT query, response FROM feedback
               WHERE rating = 1 AND query IS NOT NULL AND response IS NOT NULL
               ORDER BY created_at DESC
               LIMIT 5000"""
        )
        positive_pairs = cur.fetchall()

        # Export queries where engineer provided a correction (gold standard)
        cur.execute(
            """SELECT query, correction FROM feedback
               WHERE correction IS NOT NULL AND length(correction) > 20
               ORDER BY created_at DESC
               LIMIT 2000"""
        )
        correction_pairs = cur.fetchall()
        cur.close()
        conn.close()

        count = 0
        with open(output_path, "w", encoding="utf-8") as f:
            for query, response in positive_pairs:
                record = {
                    "prompt": f"VAVE Engineering Query: {query}",
                    "completion": response,
                    "source": "positive_feedback",
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1
            for query, correction in correction_pairs:
                record = {
                    "prompt": f"VAVE Engineering Query: {query}",
                    "completion": correction,
                    "source": "human_correction",
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1

        logger.info(f"[FeedbackHandler] Exported {count} training pairs to {output_path}")
        return count
    except Exception as e:
        logger.error(f"[FeedbackHandler] Export error: {e}")
        return 0


# ─── WEEKLY ANALYSIS JOB ─────────────────────────────────────────────────────

def run_weekly_feedback_analysis(pg_conn_func) -> Dict[str, Any]:
    """
    Weekly job: compute feedback metrics, identify improvement areas,
    and export updated training data.
    Call from jobs.py or a cron endpoint.
    """
    stats = get_feedback_stats(pg_conn_func, days=7)
    n_pairs = export_feedback_as_training_data(
        pg_conn_func, output_path="Human_Align_correction_feedback/weekly_training.jsonl"
    )
    stats["training_pairs_exported"] = n_pairs
    logger.info(f"[FeedbackHandler] Weekly analysis done: {stats}")
    return stats
