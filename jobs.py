"""
jobs.py

Utility "orchestration" module that groups together offline/batch
big-data style jobs for VAVE AI. In a real setup these would be
scheduled by Airflow/Prefect; here you can run them manually or
trigger from an admin console.
"""

from datetime import datetime
import json

from app import (
    get_db_connection,
    log_event,
)
import data_lake


def refresh_gold_layer():
    """
    Example job: recompute GOLD aggregates in the DuckDB data lake
    from the SILVER ideas Parquet.
    """
    # For now this just calls analytics_aggregate_ideas so that the
    # derived results are materialized once and can be queried fast.
    stats = data_lake.analytics_aggregate_ideas()
    log_event(
        "job_refresh_gold",
        username=None,
        payload={"stats_snapshot": stats, "run_at": datetime.utcnow().isoformat()},
    )
    return stats


def recompute_user_features():
    """
    Very simple feature-store job:
    derive avg query/response length per user from chat_history.
    """
    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute(
        """
        SELECT username,
               AVG(LENGTH(user_query)) AS avg_q,
               AVG(LENGTH(response_text)) AS avg_r
        FROM chat_history
        GROUP BY username;
        """
    )
    rows = cur.fetchall()

    for username, avg_q, avg_r in rows:
        cur.execute(
            """
            INSERT INTO user_features (username, avg_query_length, avg_response_length, last_updated)
            VALUES (%s, %s, %s, NOW())
            ON CONFLICT (username)
            DO UPDATE SET
                avg_query_length = EXCLUDED.avg_query_length,
                avg_response_length = EXCLUDED.avg_response_length,
                last_updated = NOW();
            """,
            (username, float(avg_q or 0), float(avg_r or 0)),
        )

    conn.commit()
    cur.close()
    conn.close()

    log_event(
        "job_recompute_user_features",
        username=None,
        payload={"user_count": len(rows)},
    )


