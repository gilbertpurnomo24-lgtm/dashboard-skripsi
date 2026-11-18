"""Database access for the Streamlit dashboard.
All application state is stored in TiDB Cloud; there is no SQLite/CSV fallback.
"""
from __future__ import annotations

import hashlib
import re
from contextlib import contextmanager
from datetime import datetime

import pandas as pd
import streamlit as st

LABEL_TO_SENT = {0: "negatif", 1: "netral", 2: "positif"}
SLANG = {
    "gak": "tidak", "ga": "tidak", "nggak": "tidak", "enggak": "tidak", "tdk": "tidak",
    "gk": "tidak", "yg": "yang", "dgn": "dengan", "utk": "untuk", "sdh": "sudah",
    "udh": "sudah", "blm": "belum", "jgn": "jangan", "krn": "karena", "tp": "tapi",
}


def _cfg():
    if "mysql" not in st.secrets:
        raise RuntimeError("Konfigurasi [mysql] TiDB Cloud belum tersedia di .streamlit/secrets.toml")
    return st.secrets["mysql"]


def _connect():
    import mysql.connector
    cfg = _cfg()
    kwargs = dict(
        host=cfg["host"], port=int(cfg.get("port", 4000)), user=cfg["user"],
        password=cfg["password"], database=cfg.get("database", "paylater_warehouse"),
        connection_timeout=15, autocommit=False,
    )
    # TiDB Cloud Serverless requires TLS. The CA can optionally be supplied in secrets.
    kwargs["ssl_disabled"] = False
    if cfg.get("ssl_ca"):
        kwargs["ssl_ca"] = cfg["ssl_ca"]
    return mysql.connector.connect(**kwargs)


@contextmanager
def connection():
    con = _connect()
    try:
        yield con
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()


def test_connection() -> dict:
    with connection() as con:
        cur = con.cursor()
        cur.execute("SELECT DATABASE(), VERSION()")
        database, version = cur.fetchone()
        cur.close()
    return {"database": database, "version": version}


def init_db() -> None:
    """Create only dashboard-owned tables. Research star-schema tables are migrated separately."""
    with connection() as con:
        cur = con.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS dashboard_users (
                email VARCHAR(255) PRIMARY KEY,
                name VARCHAR(255),
                role VARCHAR(20) NOT NULL DEFAULT 'user',
                first_login DATETIME NOT NULL,
                last_login DATETIME NOT NULL,
                login_count INT NOT NULL DEFAULT 1
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS dashboard_reviews_new (
                review_id VARCHAR(191) PRIMARY KEY,
                app_name VARCHAR(100) NOT NULL,
                score INT NOT NULL,
                review_at DATETIME NOT NULL,
                content TEXT,
                clean_text TEXT,
                svm_pred INT NULL,
                added_by VARCHAR(255),
                added_at DATETIME NOT NULL,
                INDEX idx_dashboard_reviews_app (app_name),
                INDEX idx_dashboard_reviews_date (review_at)
            )
        """)
        cur.close()


def record_login(email: str, name: str | None, role: str) -> None:
    now = datetime.now().replace(microsecond=0)
    with connection() as con:
        cur = con.cursor()
        cur.execute("SELECT email FROM dashboard_users WHERE email=%s", (email,))
        if cur.fetchone() is None:
            cur.execute(
                "INSERT INTO dashboard_users (email,name,role,first_login,last_login,login_count) VALUES (%s,%s,%s,%s,%s,1)",
                (email, name, role, now, now),
            )
        else:
            cur.execute(
                "UPDATE dashboard_users SET name=%s, role=%s, last_login=%s, login_count=login_count+1 WHERE email=%s",
                (name, role, now, email),
            )
        cur.close()


def list_users() -> pd.DataFrame:
    with connection() as con:
        return pd.read_sql_query(
            "SELECT email,name,role,first_login,last_login,login_count FROM dashboard_users ORDER BY first_login DESC",
            con,
        )


def clean_text(text: str) -> str:
    t = str(text).lower()
    t = re.sub(r"https?://\S+|www\.\S+", " ", t)
    t = re.sub(r"[^a-z\s]", " ", t)
    return " ".join(SLANG.get(w, w) for w in t.split())


def score_to_label(score: int) -> int:
    return 0 if int(score) <= 2 else (1 if int(score) == 3 else 2)


def add_reviews(df: pd.DataFrame, added_by: str, existing_ids: set[str]) -> tuple[int, int]:
    now = datetime.now().replace(microsecond=0)
    saved = skipped = 0
    with connection() as con:
        cur = con.cursor()
        cur.execute("SELECT review_id FROM dashboard_reviews_new")
        have = {str(r[0]) for r in cur.fetchall()} | {str(x) for x in existing_ids}
        for r in df.itertuples(index=False):
            rid = str(getattr(r, "reviewId", "") or "").strip()
            if not rid:
                key = f"{r.app_name}|{pd.Timestamp(r.at).isoformat()}|{r.content}"
                rid = "new-" + hashlib.sha1(key.encode("utf-8")).hexdigest()[:20]
            if rid in have:
                skipped += 1
                continue
            pred = getattr(r, "svm_pred", None)
            pred = None if pred is None or pd.isna(pred) else int(pred)
            clean = getattr(r, "clean_text", None)
            cur.execute(
                "INSERT INTO dashboard_reviews_new (review_id,app_name,score,review_at,content,clean_text,svm_pred,added_by,added_at) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                (rid, str(r.app_name), int(r.score), pd.Timestamp(r.at).to_pydatetime(), str(r.content), clean, pred, added_by, now),
            )
            have.add(rid); saved += 1
        cur.close()
    return saved, skipped


def load_new_reviews() -> pd.DataFrame:
    with connection() as con:
        df = pd.read_sql_query(
            "SELECT review_id AS reviewId, app_name, score, review_at AS at, content, clean_text, svm_pred, added_by, added_at FROM dashboard_reviews_new",
            con,
        )
    if df.empty:
        return df
    df["at"] = pd.to_datetime(df["at"])
    df["label"] = df["score"].map(score_to_label)
    df["sentiment"] = df["label"].map(LABEL_TO_SENT)
    df["date"] = df["at"].dt.date
    df["month"] = df["at"].dt.to_period("M").dt.to_timestamp()
    df["clean_text"] = df["clean_text"].fillna("")
    return df


def delete_reviews(ids: list[str]) -> int:
    if not ids:
        return 0
    with connection() as con:
        cur = con.cursor()
        cur.executemany("DELETE FROM dashboard_reviews_new WHERE review_id=%s", [(str(i),) for i in ids])
        n = cur.rowcount
        cur.close()
    return n
