#!/usr/bin/env python3
"""
Insert enriched coarse-unit rows into the database using a connection pool.

流程：
1. 读取 .env 中的 DATABASE_URL。
2. 遍历 fine_unit_all_flat_enriched.jsonl，将每条记录转换为数据库字段。
3. 使用最多 40 个连接的连接池 + 线程池并行写入 semantic.coarse_unit。
4. 每条记录至少重试一次；成功写入 success_log，失败写入 fail_log。
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List
import threading

try:
    import psycopg2
    from psycopg2 import pool as pg_pool
    from psycopg2.extras import Json
except ImportError as exc:  # pragma: no cover - runtime dependency hint
    raise SystemExit(
        "psycopg2 is required. Install with `pip install psycopg2-binary`."
    ) from exc


def load_database_url(env_path: Path) -> str:
    """Parse DATABASE_URL from a .env file."""
    if not env_path.exists():
        raise FileNotFoundError(f".env file not found at {env_path}")
    database_url = None
    with env_path.open("r", encoding="utf-8") as env_file:
        for line in env_file:
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            if key == "DATABASE_URL":
                database_url = value
                break
    if not database_url:
        raise ValueError("DATABASE_URL not found in .env")
    return database_url


def iter_enriched_rows(path: Path) -> Iterable[Dict[str, Any]]:
    """Yield records from the enriched JSONL file."""
    with path.open("r", encoding="utf-8") as infile:
        for line_number, line in enumerate(infile, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                yield json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Failed to parse JSON at line {line_number} in {path}"
                ) from exc


def prepare_payload(record: Dict[str, Any]) -> Dict[str, Any]:
    """Map the JSON record to database column payload."""
    fine_unit_ids = [
        int(unit_id) for unit_id in record.get("fine_unit_ids", []) if unit_id
    ]
    original_defs = record.get("original_defs", [])
    pattern_raw = record.get("pattern")
    pattern_json: Any = None
    if pattern_raw:
        if isinstance(pattern_raw, str):
            try:
                pattern_json = json.loads(pattern_raw)
            except json.JSONDecodeError:
                pattern_json = pattern_raw
        else:
            pattern_json = pattern_raw

    payload = {
        "kind": record.get("kind"),
        "label": record.get("label"),
        "pos": record.get("pos") or None,
        "english_def": record.get("english_def") or None,
        "chinese_def": record.get("chinese_def") or None,
        "chinese_criteria": record.get("chinese_criteria") or None,
        "chinese_label": record.get("chinese_label") or None,
        "english_label": record.get("english_label") or None,
        "pattern": pattern_json,
        "fine_unit_ids": fine_unit_ids,
        "original_defs": original_defs,
    }
    return payload


def insert_row(conn, payload: Dict[str, Any]) -> None:
    """Execute the insert statement using the provided connection."""
    with conn.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO semantic.coarse_unit (
                kind, label, pos, english_def, chinese_def, chinese_criteria,
                chinese_label, english_label, pattern, fine_unit_ids, original_defs
            )
            VALUES (
                %(kind)s, %(label)s, %(pos)s, %(english_def)s, %(chinese_def)s,
                %(chinese_criteria)s, %(chinese_label)s, %(english_label)s,
                %(pattern)s, %(fine_unit_ids)s, %(original_defs)s
            )
            """,
            {
                **payload,
                "pattern": Json(payload["pattern"])
                if payload["pattern"] is not None
                else None,
            },
        )


def append_log(path: Path, entry: Dict[str, Any], lock: threading.Lock) -> None:
    """Append a JSON entry to the log file in a thread-safe manner."""
    with lock:
        with path.open("a", encoding="utf-8") as logfile:
            json.dump(entry, logfile, ensure_ascii=False)
            logfile.write("\n")


def handle_record(
    record: Dict[str, Any],
    pool: pg_pool.SimpleConnectionPool,
    success_log: Path,
    fail_log: Path,
    log_lock: threading.Lock,
    retry_delay: float,
    max_attempts: int,
) -> None:
    """Insert a single record with retry logic."""
    payload = prepare_payload(record)
    attempts = 0
    last_error: Exception | None = None

    while attempts < max_attempts:
        attempts += 1
        conn = pool.getconn()
        try:
            insert_row(conn, payload)
            conn.commit()
            append_log(
                success_log,
                {"label": payload["label"], "fine_unit_ids": payload["fine_unit_ids"]},
                log_lock,
            )
            last_error = None
            return
        except Exception as exc:  # pragma: no cover - runtime logging
            conn.rollback()
            last_error = exc
        finally:
            pool.putconn(conn)

        if attempts < max_attempts:
            time.sleep(retry_delay)
        else:
            append_log(
                fail_log,
                {
                    "label": payload["label"],
                    "fine_unit_ids": payload["fine_unit_ids"],
                    "error": str(last_error),
                },
                log_lock,
            )


def process_records(
    pool: pg_pool.SimpleConnectionPool,
    records: Iterable[Dict[str, Any]],
    success_log: Path,
    fail_log: Path,
    retry_delay: float,
    max_attempts: int,
    workers: int,
) -> None:
    """Insert all records using a thread pool and connection pool."""
    log_lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                handle_record,
                record,
                pool,
                success_log,
                fail_log,
                log_lock,
                retry_delay,
                max_attempts,
            )
            for record in records
        ]
        for future in as_completed(futures):
            # Propagate any unexpected exceptions.
            future.result()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Insert enriched coarse-unit records into PostgreSQL."
    )
    default_dir = Path(__file__).resolve().parent
    parser.add_argument(
        "--env-path",
        type=Path,
        default=default_dir.parent / ".env",
        help="Path to .env file containing DATABASE_URL.",
    )
    parser.add_argument(
        "--input-jsonl",
        type=Path,
        default=default_dir / "fine_unit_all_flat_enriched.jsonl",
        help="Enriched JSONL source (default: fine_unit_all_flat_enriched.jsonl).",
    )
    parser.add_argument(
        "--success-log",
        type=Path,
        default=default_dir / "success_log.jsonl",
        help="File to append successful inserts (default: success_log.jsonl).",
    )
    parser.add_argument(
        "--fail-log",
        type=Path,
        default=default_dir / "fail_log.jsonl",
        help="File to append failed inserts (default: fail_log.jsonl).",
    )
    parser.add_argument(
        "--retry-delay",
        type=float,
        default=1.0,
        help="Seconds to wait between retries (default: 1.0).",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=2,
        help="Maximum attempts per row (default: 2, meaning 1 retry).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=40,
        help="Number of concurrent workers / max connections (default: 40).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    database_url = load_database_url(args.env_path)
    records: List[Dict[str, Any]] = list(iter_enriched_rows(args.input_jsonl))

    pool = pg_pool.SimpleConnectionPool(1, args.workers, dsn=database_url)
    try:
        process_records(
            pool=pool,
            records=records,
            success_log=args.success_log,
            fail_log=args.fail_log,
            retry_delay=args.retry_delay,
            max_attempts=args.max_attempts,
            workers=args.workers,
        )
    finally:
        pool.closeall()


if __name__ == "__main__":
    main()
