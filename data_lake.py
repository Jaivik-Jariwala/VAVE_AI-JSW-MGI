import os
from pathlib import Path
from datetime import datetime
import duckdb

BASE_DIR = Path(__file__).parent.resolve()
DATA_LAKE_DIR = BASE_DIR / "data_lake"
RAW_DIR = DATA_LAKE_DIR / "raw"
PROCESSED_DIR = DATA_LAKE_DIR / "processed"
DB_PATH = DATA_LAKE_DIR / "lake.duckdb"


def init_data_lake():
    """
    Ensure data lake directories and DuckDB file exist.
    This is idempotent and safe to call on every app startup.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    DATA_LAKE_DIR.mkdir(parents=True, exist_ok=True)

    # Touch the duckdb file by opening a connection once
    con = duckdb.connect(str(DB_PATH))
    con.close()


def _connect():
    """Internal helper to connect to the DuckDB lake."""
    return duckdb.connect(str(DB_PATH))


def ingest_uploaded_csv(local_csv_path: Path, logical_dataset: str = "ideas") -> Path:
    """
    Move an uploaded CSV into the RAW zone with a timestamped, organized path.

    Returns the final RAW path of the file.
    """
    init_data_lake()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Folder per ingestion timestamp -> simulates append-only raw zone
    dataset_dir = RAW_DIR / logical_dataset / ts
    dataset_dir.mkdir(parents=True, exist_ok=True)

    dest_path = dataset_dir / local_csv_path.name
    # We keep original as the app may still need it; for now copy.
    # If you prefer, you can change this to os.replace to move instead.
    if str(local_csv_path) != str(dest_path):
        os.replace(local_csv_path, dest_path)

    return dest_path


def run_etl_on_raw_csv(raw_csv_path: Path, logical_dataset: str = "ideas") -> dict:
    """
    ETL step:
    - Read raw CSV into DuckDB
    - Apply light cleaning / typing
    - Write columnar Parquet files partitioned by key columns (big-data style)
    """
    init_data_lake()
    con = _connect()

    # You can adjust schema and types here as your CSV evolves.
    # For now we rely on DuckDB's auto schema inference.
    raw_rel = con.read_csv(str(raw_csv_path), header=True, auto_detect=True)

    # NOTE: We intentionally skip column renaming here to avoid compatibility
    # issues with different DuckDB Python versions. The lake operates on
    # whatever column names exist in the CSV.

    # Example: derive simple date partition based on current date
    today = datetime.now().strftime("%Y-%m-%d")
    load_ts = datetime.now().isoformat(timespec="seconds")

    # Big-data concept: write to Parquet and PARTITION BY some dimensions
    target_dir = PROCESSED_DIR / logical_dataset
    target_dir.mkdir(parents=True, exist_ok=True)

    # Partition by load_date to simulate large-scale partitioned lake
    parquet_path = target_dir / f"load_date={today}"
    parquet_path.mkdir(parents=True, exist_ok=True)

    # Write a Parquet file for this load
    out_file = parquet_path / (raw_csv_path.stem + ".parquet")

    # Actually write Parquet for this load using the relation API
    raw_rel.write_parquet(str(out_file))

    # Persist per-load stats inside DuckDB for monitoring (lake-wide metadata)
    # Use a simple BIGINT for id to avoid unsupported IDENTITY constraints
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS lake_loads (
            id BIGINT,
            logical_dataset TEXT,
            raw_path TEXT,
            parquet_path TEXT,
            load_date TEXT,
            load_ts TEXT,
            row_count BIGINT
        );
        """
    )

    row_count = raw_rel.count("*").execute().fetchone()[0]

    con.execute(
        """
        INSERT INTO lake_loads (logical_dataset, raw_path, parquet_path, load_date, load_ts, row_count)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        [
            logical_dataset,
            str(raw_csv_path),
            str(out_file),
            today,
            load_ts,
            int(row_count),
        ],
    )

    con.close()

    return {
        "logical_dataset": logical_dataset,
        "raw_path": str(raw_csv_path),
        "parquet_path": str(out_file),
        "load_date": today,
        "load_ts": load_ts,
        "row_count": int(row_count),
    }


def analytics_aggregate_ideas():
    """
    Example analytics over the processed Parquet data.

    Returns:
        dict with a few aggregated metrics that can be exposed via Flask.
    """
    init_data_lake()
    con = _connect()

    target_dir = str(PROCESSED_DIR / "ideas")

    # If no processed data yet, return empty result
    if not os.path.exists(target_dir):
        con.close()
        return {
            "total_ideas": 0,
            "by_dept": [],
            "top_savings": [],
        }

    # DuckDB doesn't support prepared parameters in CREATE VIEW statements
    # Use string formatting instead (safe here as target_dir is controlled)
    con.execute(
        f"""
        CREATE OR REPLACE TEMP VIEW ideas AS
        SELECT *
        FROM read_parquet('{target_dir}/**/*.parquet');
        """
    )

    # Total ideas count
    total_ideas = con.execute("SELECT COUNT(*) FROM ideas").fetchone()[0]

    # Aggregation by department if column is available
    by_dept = []
    try:
        by_dept = con.execute(
            """
            SELECT
                COALESCE(dept, 'Unknown') AS dept,
                COUNT(*) AS idea_count
            FROM ideas
            GROUP BY dept
            ORDER BY idea_count DESC
            LIMIT 50;
            """
        ).fetchall()
    except duckdb.CatalogException:
        by_dept = []

    # Top ideas by saving_value_inr if available
    top_savings = []
    try:
        top_savings = con.execute(
            """
            SELECT
                idea_id,
                cost_reduction_idea,
                saving_value_inr
            FROM ideas
            WHERE saving_value_inr IS NOT NULL
            ORDER BY saving_value_inr DESC
            LIMIT 20;
            """
        ).fetchall()
    except duckdb.CatalogException:
        top_savings = []

    con.close()

    return {
        "total_ideas": int(total_ideas),
        "by_dept": [
            {"dept": row[0], "idea_count": int(row[1])} for row in by_dept
        ],
        "top_savings": [
            {
                "idea_id": row[0],
                "cost_reduction_idea": row[1],
                "saving_value_inr": float(row[2]) if row[2] is not None else None,
            }
            for row in top_savings
        ],
    }


def lake_status(logical_dataset: str = "ideas") -> dict:
    """
    Advanced lake-wide stats for big-data style monitoring.

    Shows:
    - Number of loads
    - Total rows across all Parquet files
    - Approx rows per load_date
    """
    init_data_lake()
    con = _connect()

    # Ensure lake_loads metadata table exists (no unsupported identity constraints)
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS lake_loads (
            id BIGINT,
            logical_dataset TEXT,
            raw_path TEXT,
            parquet_path TEXT,
            load_date TEXT,
            load_ts TEXT,
            row_count BIGINT
        );
        """
    )

    # Summary over loads for this dataset
    summary = con.execute(
        """
        SELECT 
            COUNT(*) AS load_count,
            COALESCE(SUM(row_count), 0) AS total_rows
        FROM lake_loads
        WHERE logical_dataset = ?
        """,
        [logical_dataset],
    ).fetchone()

    load_count = summary[0] or 0
    total_rows = summary[1] or 0

    by_date = con.execute(
        """
        SELECT load_date, SUM(row_count) AS rows
        FROM lake_loads
        WHERE logical_dataset = ?
        GROUP BY load_date
        ORDER BY load_date
        """,
        [logical_dataset],
    ).fetchall()

    con.close()

    return {
        "logical_dataset": logical_dataset,
        "load_count": int(load_count),
        "total_rows": int(total_rows),
        "by_date": [
            {"load_date": r[0], "rows": int(r[1])}
            for r in by_date
        ],
    }


