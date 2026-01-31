import sqlite3
import os
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


def _get_db_path() -> str:
    path = os.environ.get("AUDIT_DB_PATH", "data/audit.db")
    # Ensure directory exists
    dirpath = os.path.dirname(path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    return path


def _ensure_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS ai_audit (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT,
            system_message TEXT,
            user_message TEXT,
            response TEXT,
            served_from_fallback INTEGER,
            metadata TEXT
        )
        """
    )
    conn.commit()


def audit_ai_call(system_message: str, user_message: str, response: str, served_from_fallback: bool = False, metadata: dict | None = None) -> None:
    """Append an audit record to a local SQLite DB. Failures are logged but do not raise."""
    try:
        db_path = _get_db_path()
        conn = sqlite3.connect(db_path, timeout=5)
        try:
            _ensure_table(conn)
            conn.execute(
                "INSERT INTO ai_audit (ts, system_message, user_message, response, served_from_fallback, metadata) VALUES (?,?,?,?,?,?)",
                (
                    datetime.utcnow().isoformat(),
                    system_message,
                    user_message,
                    response,
                    1 if served_from_fallback else 0,
                    json.dumps(metadata) if metadata else None,
                ),
            )
            conn.commit()
        finally:
            conn.close()
    except Exception as e:
        logger.warning("Failed to write AI audit record: %s", e)
