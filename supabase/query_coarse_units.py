#!/usr/bin/env python3
"""
Query coarse-unit label RPC functions via DATABASE_URL in the repo .env file.

Examples:
  python supabase/query_coarse_units.py exact apple
  python supabase/query_coarse_units.py contain apple "be addicted to" "want to"
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any


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
                database_url = value.strip().strip('"').strip("'")
                break

    if not database_url:
        raise ValueError("DATABASE_URL not found in .env")
    return database_url


def normalize_mode(raw_mode: str) -> tuple[str, str]:
    """Map user mode to RPC function name."""
    mode = raw_mode.strip().lower()
    if mode == "exact":
        return "exact", "public.coarse_unit_label_exact"
    if mode in {"contain", "contains"}:
        return "contain", "public.coarse_unit_label_contains"
    raise ValueError("mode must be one of: exact, contain, contains")


def sql_quote(value: str) -> str:
    """Escape a Python string into a SQL single-quoted literal."""
    return "'" + value.replace("'", "''") + "'"


def run_query(database_url: str, function_name: str, query: str, limit_n: int) -> list[dict[str, Any]]:
    """Call the target RPC function and return rows as Python objects."""
    sql = (
        "select coalesce(json_agg(t), '[]'::json)::text "
        f"from {function_name}({sql_quote(query)}, {limit_n}) as t;"
    )
    completed = subprocess.run(
        [
            "psql",
            database_url,
            "-X",
            "-A",
            "-t",
            "-v",
            "ON_ERROR_STOP=1",
            "-c",
            sql,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    output = completed.stdout.strip()
    if not output:
        return []
    return json.loads(output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Query semantic.coarse_unit label RPC functions."
    )
    default_dir = Path(__file__).resolve().parent
    parser.add_argument(
        "mode",
        help="Query mode: exact or contain/contains.",
    )
    parser.add_argument(
        "queries",
        nargs="+",
        help="One or more query strings. Each is executed independently.",
    )
    parser.add_argument(
        "--env-path",
        type=Path,
        default=default_dir.parent / ".env",
        help="Path to .env file containing DATABASE_URL.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Requested row limit per query. RPC function still caps at 20.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mode, function_name = normalize_mode(args.mode)
    database_url = load_database_url(args.env_path)
    limit_n = max(1, args.limit)

    results = []
    for raw_query in args.queries:
        rows = run_query(database_url, function_name, raw_query, limit_n)
        rows = [row for row in rows if row.get("status") == "active"]
        rows = [{key: value for key, value in row.items() if key != "status"} for row in rows]
        results.append(
            {
                "query": raw_query,
                "count": len(rows),
                "rows": rows,
            }
        )

    print(
        json.dumps(
            {"results": results},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
