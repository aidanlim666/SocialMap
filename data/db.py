"""Database query helpers for SocialMap."""

import sqlite3
from pathlib import Path

import numpy as np

DB_PATH = Path(__file__).parent / "socialmap.db"


def _conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def db_ready() -> bool:
    if not DB_PATH.exists():
        return False
    with _conn() as conn:
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        return "counties" in tables and "sci" in tables


def search_counties(query: str, limit: int = 10) -> list[dict]:
    """Full-text search on county name and state, ordered by population descending."""
    like = f"%{query}%"
    with _conn() as conn:
        rows = conn.execute(
            """
            SELECT fips, name, state, population, country
            FROM counties
            WHERE name LIKE ? COLLATE NOCASE OR state LIKE ? COLLATE NOCASE
            ORDER BY population DESC
            LIMIT ?
            """,
            (like, like, limit),
        ).fetchall()
    return [dict(r) for r in rows]


def get_county_info(fips: str) -> dict | None:
    with _conn() as conn:
        row = conn.execute(
            "SELECT fips, name, state, population, country FROM counties WHERE fips = ?",
            (fips,),
        ).fetchone()
    return dict(row) if row else None


def get_sci_scores(fips: str) -> dict[str, float]:
    """
    Returns raw (already log-normalised 0-100) SCI scores for all counties
    connected to `fips`.  The caller re-normalises these relative to each
    other so the top-connected county always anchors at 100.

    Both pair directions are stored in the DB, so querying fips_a alone
    is sufficient — no fallback needed.
    """
    with _conn() as conn:
        rows = conn.execute(
            "SELECT fips_b, sci FROM sci WHERE fips_a = ?", (fips,)
        ).fetchall()
    return {r[0]: r[1] for r in rows}


def normalise_scores(raw: dict[str, float]) -> dict[str, float]:
    """
    Re-normalise a {fips: score} dict to a 0–100 range using min-max scaling.

    Stored SCI values are already on a global log scale, so their distribution
    is already log-shaped.  A second log pass would compress everything into
    a narrow band; min-max preserves the full spread while anchoring the
    best-connected county at 100 and the least-connected at 0.
    """
    if not raw:
        return {}
    values = np.array(list(raw.values()), dtype=float)
    lo, hi = values.min(), values.max()
    if hi == lo:
        return {k: 50.0 for k in raw}
    normalised = ((values - lo) / (hi - lo) * 100).round(1)
    return dict(zip(raw.keys(), normalised.tolist()))
