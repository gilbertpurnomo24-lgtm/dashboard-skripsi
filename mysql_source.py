"""TiDB Cloud source for the Streamlit dashboard.
TiDB Cloud is MySQL-compatible, so the existing MySQL module is retained by name.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st
import db

LABEL_BY_SENT = {"negatif": 0, "netral": 1, "positif": 2}


def configured() -> bool:
    try:
        return "mysql" in st.secrets and bool(st.secrets["mysql"].get("host")) and bool(st.secrets["mysql"].get("user"))
    except Exception:
        return False


def _connect():
    return db._connect()


@st.cache_data(ttl=300, show_spinner="Memuat data dari TiDB Cloud...")
def load_reviews() -> pd.DataFrame:
    con = _connect()
    try:
        cur = con.cursor()
        cur.execute("SHOW COLUMNS FROM Fact_Reviews LIKE 'clean_text'")
        has_clean = cur.fetchone() is not None
        clean_col = ", f.clean_text" if has_clean else ""
        cur.execute(f"""
            SELECT f.source_review_id, a.app_code, f.score, s.sentiment_name,
                   d.full_date, f.review_text{clean_col}
            FROM Fact_Reviews f
            JOIN Dim_Application a ON a.app_id = f.app_id
            JOIN Dim_Sentiment s ON s.sentiment_id = f.actual_sentiment_id
            JOIN Dim_Date d ON d.date_id = f.date_id
            WHERE f.source_review_id IS NOT NULL
        """)
        rows = cur.fetchall()
        cols = ["reviewId", "app_name", "score", "sentiment", "at", "content"] + (["clean_text"] if has_clean else [])
    finally:
        con.close()
    df = pd.DataFrame(rows, columns=cols)
    if df.empty:
        raise ValueError("Tabel Fact_Reviews kosong di TiDB Cloud.")
    if not has_clean:
        df["clean_text"] = df["content"].map(db.clean_text)
    df["at"] = pd.to_datetime(df["at"], errors="coerce")
    df = df.dropna(subset=["at"]).copy()
    df["reviewId"] = df["reviewId"].astype(str)
    df["app_name"] = df["app_name"].astype(str)
    df["sentiment"] = df["sentiment"].astype(str).str.lower()
    df["label"] = df["sentiment"].map(LABEL_BY_SENT).astype(int)
    df["score"] = df["score"].astype(int)
    df["content"] = df["content"].fillna("").astype(str)
    df["clean_text"] = df["clean_text"].fillna("").astype(str)
    df["date"] = df["at"].dt.date
    df["month"] = df["at"].dt.to_period("M").dt.to_timestamp()
    return df


@st.cache_data(ttl=300, show_spinner="Memuat prediksi model dari TiDB Cloud...")
def load_model_predictions(model_name: str) -> pd.DataFrame:
    con = _connect()
    try:
        cur = con.cursor()
        cur.execute("""
            SELECT p.review_id, p.actual_label AS label, p.predicted_label AS pred
            FROM Model_Predictions p
            WHERE p.model_name=%s
        """, (model_name,))
        rows = cur.fetchall()
    finally:
        con.close()
    pred = pd.DataFrame(rows, columns=["reviewId", "label", "pred"])
    if pred.empty:
        return pred
    base = load_reviews()
    cols = ["reviewId", "app_name", "at", "clean_text", "content"]
    pred = pred.merge(base[cols], on="reviewId", how="left", validate="one_to_one")
    pred["label"] = pred["label"].astype(int)
    pred["pred"] = pred["pred"].astype(int)
    pred["at"] = pd.to_datetime(pred["at"], errors="coerce")
    pred["date"] = pred["at"].dt.date
    pred["month"] = pred["at"].dt.to_period("M").dt.to_timestamp()
    return pred


def load_model_evaluation() -> pd.DataFrame:
    con = _connect()
    try:
        return pd.read_sql_query(
            "SELECT model_name, accuracy, precision_macro, recall_macro, f1_macro, f1_weighted, test_count FROM Model_Evaluation ORDER BY model_name",
            con,
        )
    finally:
        con.close()
