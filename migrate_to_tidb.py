"""Migrate a local MySQL warehouse + model prediction CSVs into TiDB Cloud.

Usage:
  python migrate_to_tidb.py --svm svm_test_predictions.csv --xlm xlm_test_predictions.csv

The source database is configured in .streamlit/secrets.toml under [source_mysql].
The TiDB target uses [mysql]. Passwords are never embedded in this file.
"""
from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import mysql.connector
import pandas as pd
from scipy.stats import binomtest, chi2
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

LABELS = [0, 1, 2]
MODELS = {"SVM": "svm", "XLM-RoBERTa": "xlm"}


def read_cfg(section: str) -> dict:
    # Prefer Streamlit secrets when available; otherwise fail with a clear instruction.
    try:
        import streamlit as st
        return dict(st.secrets[section])
    except Exception as exc:
        raise RuntimeError(f"Bagian [{section}] tidak tersedia di .streamlit/secrets.toml: {exc}") from exc


def connect(cfg: dict):
    user = cfg.get("user") or cfg.get("username")
    if not user:
        raise RuntimeError("Konfigurasi database tidak memiliki `user` atau `username`.")

    kwargs = dict(
        host=cfg["host"],
        port=int(cfg.get("port", 3306)),
        user=user,
        password=cfg.get("password", ""),
        database=cfg["database"],
        connection_timeout=20,
        autocommit=False,
    )
    if cfg.get("ssl_ca"):
        kwargs["ssl_ca"] = cfg["ssl_ca"]
        kwargs["ssl_verify_cert"] = True
        kwargs["ssl_verify_identity"] = True
    else:
        kwargs["ssl_disabled"] = False
    return mysql.connector.connect(**kwargs)


def qi(name: str) -> str:
    return "`" + name.replace("`", "``") + "`"


def get_tables(con):
    cur = con.cursor()
    cur.execute("SHOW FULL TABLES WHERE Table_type='BASE TABLE'")
    rows = cur.fetchall()
    cur.close()
    return [r[0] for r in rows]


def show_create(con, table):
    cur = con.cursor()
    cur.execute(f"SHOW CREATE TABLE {qi(table)}")
    ddl = cur.fetchone()[1]
    cur.close()
    # TiDB is MySQL-compatible. Remove MySQL-only table options that are commonly
    # rejected by TiDB versions, while retaining keys and constraints.
    ddl = re.sub(r"\s+AUTO_INCREMENT=\d+", "", ddl, flags=re.I)
    ddl = re.sub(
        r"\)\s+(ENGINE|AUTO_INCREMENT|DEFAULT CHARSET|CHARSET|COLLATE|ROW_FORMAT|"
        r"KEY_BLOCK_SIZE|STATS_PERSISTENT|COMMENT).*$",
        ")",
        ddl,
        flags=re.I,
    )
    ddl = ddl.replace("utf8mb4_0900_ai_ci", "utf8mb4_general_ci")
    return ddl


def columns(con, table):
    cur = con.cursor()
    cur.execute(f"SHOW COLUMNS FROM {qi(table)}")
    rows = cur.fetchall()
    cur.close()
    return [r[0] for r in rows]


def migrate_tables(src, dst, replace: bool):
    tables = get_tables(src)
    print(f"[1/4] Tabel sumber ditemukan: {len(tables)}")
    dcur = dst.cursor()
    dcur.execute("SET FOREIGN_KEY_CHECKS=0")
    if replace:
        for table in tables:
            dcur.execute(f"DROP TABLE IF EXISTS {qi(table)}")
    for table in tables:
        ddl = show_create(src, table)
        if not replace:
            ddl = re.sub(r"^CREATE TABLE\s+", "CREATE TABLE IF NOT EXISTS ", ddl, count=1, flags=re.I)
        try:
            dcur.execute(ddl)
        except mysql.connector.Error as exc:
            raise RuntimeError(f"Gagal membuat tabel {table} di TiDB:\n{exc}\nDDL:\n{ddl}") from exc
    if replace:
        for table in tables:
            dcur.execute(f"TRUNCATE TABLE {qi(table)}")
    dst.commit()

    print("[2/4] Menyalin isi tabel...")
    for i, table in enumerate(tables, 1):
        cols = columns(src, table)
        scur = src.cursor()
        scur.execute(f"SELECT * FROM {qi(table)}")
        rows = scur.fetchall()
        scur.close()
        if not rows:
            print(f"  {i:>2}/{len(tables)} {table}: 0 baris")
            continue
        placeholders = ",".join(["%s"] * len(cols))
        sql = f"INSERT IGNORE INTO {qi(table)} ({','.join(qi(c) for c in cols)}) VALUES ({placeholders})"
        cur = dst.cursor()
        batch = 1000
        for start in range(0, len(rows), batch):
            cur.executemany(sql, rows[start:start+batch])
            dst.commit()
        cur.close()
        print(f"  {i:>2}/{len(tables)} {table}: {len(rows):,} baris")
    dcur.execute("SET FOREIGN_KEY_CHECKS=1")
    dst.commit()
    dcur.close()
    return tables


def ensure_model_tables(dst):
    cur = dst.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS Model_Predictions (
            prediction_id BIGINT AUTO_INCREMENT PRIMARY KEY,
            review_id VARCHAR(191) NOT NULL,
            model_name VARCHAR(50) NOT NULL,
            actual_label TINYINT NOT NULL,
            predicted_label TINYINT NOT NULL,
            UNIQUE KEY uq_model_review (model_name, review_id),
            KEY idx_model_name (model_name),
            KEY idx_model_review (review_id)
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS Model_Evaluation (
            model_name VARCHAR(50) PRIMARY KEY,
            accuracy DOUBLE NOT NULL,
            precision_macro DOUBLE NOT NULL,
            recall_macro DOUBLE NOT NULL,
            f1_macro DOUBLE NOT NULL,
            f1_weighted DOUBLE NOT NULL,
            test_count INT NOT NULL,
            evaluated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS Model_Class_Metrics (
            model_name VARCHAR(50) NOT NULL,
            label_id TINYINT NOT NULL,
            sentiment_name VARCHAR(20) NOT NULL,
            precision_score DOUBLE NOT NULL,
            recall_score DOUBLE NOT NULL,
            f1_score DOUBLE NOT NULL,
            support INT NOT NULL,
            PRIMARY KEY (model_name, label_id)
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS Model_Confusion_Matrix (
            model_name VARCHAR(50) NOT NULL,
            actual_label TINYINT NOT NULL,
            predicted_label TINYINT NOT NULL,
            count_value INT NOT NULL,
            PRIMARY KEY (model_name, actual_label, predicted_label)
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS Model_Comparison (
            comparison_id INT PRIMARY KEY,
            model_a VARCHAR(50) NOT NULL,
            model_b VARCHAR(50) NOT NULL,
            both_correct INT NOT NULL,
            a_only_correct INT NOT NULL,
            b_only_correct INT NOT NULL,
            both_wrong INT NOT NULL,
            p_value DOUBLE NOT NULL,
            test_method VARCHAR(100) NOT NULL,
            sample_count INT NOT NULL,
            evaluated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cur.close(); dst.commit()


def read_prediction(path: Path, model_name: str):
    df = pd.read_csv(path)
    if "reviewId" not in df.columns or "label" not in df.columns:
        raise ValueError(f"{path} wajib memiliki kolom reviewId dan label.")
    pred_col = next((c for c in df.columns if "pred" in c.lower()), None)
    if pred_col is None:
        raise ValueError(f"{path} tidak memiliki kolom prediksi.")
    out = df[["reviewId", "label", pred_col]].copy()
    out.columns = ["review_id", "actual_label", "predicted_label"]
    out["review_id"] = out["review_id"].astype(str)
    out["actual_label"] = out["actual_label"].astype(int)
    out["predicted_label"] = out["predicted_label"].astype(int)
    out["model_name"] = model_name
    if not out["actual_label"].isin(LABELS).all() or not out["predicted_label"].isin(LABELS).all():
        raise ValueError(f"Label/prediksi di {path} harus bernilai 0, 1, atau 2.")
    if out["review_id"].duplicated().any():
        raise ValueError(f"reviewId duplikat ditemukan pada {path}.")
    return out


def store_predictions(dst, svm_path: Path, xlm_path: Path):
    ensure_model_tables(dst)
    data = {"SVM": read_prediction(svm_path, "SVM"), "XLM-RoBERTa": read_prediction(xlm_path, "XLM-RoBERTa")}
    cur = dst.cursor()
    cur.execute("TRUNCATE TABLE Model_Predictions")
    for model, df in data.items():
        sql = "INSERT INTO Model_Predictions (review_id,model_name,actual_label,predicted_label) VALUES (%s,%s,%s,%s)"
        rows = [(r.review_id, model, int(r.actual_label), int(r.predicted_label)) for r in df.itertuples(index=False)]
        cur.executemany(sql, rows)
    dst.commit()

    # Confirm all prediction IDs exist in the warehouse.
    for model, df in data.items():
        cur.execute("""
            SELECT COUNT(*) FROM Model_Predictions p
            JOIN Fact_Reviews f ON f.source_review_id=p.review_id
            WHERE p.model_name=%s
        """, (model,))
        matched = int(cur.fetchone()[0])
        if matched != len(df):
            raise RuntimeError(f"{model}: hanya {matched}/{len(df)} reviewId prediksi ditemukan di Fact_Reviews.")

        y, p = df.actual_label.to_numpy(), df.predicted_label.to_numpy()
        pr, rc, f1, sup = precision_recall_fscore_support(y, p, labels=LABELS, zero_division=0)
        acc = float(accuracy_score(y, p))
        cur.execute("REPLACE INTO Model_Evaluation (model_name,accuracy,precision_macro,recall_macro,f1_macro,f1_weighted,test_count) VALUES (%s,%s,%s,%s,%s,%s,%s)",
                    (model, acc, float(pr.mean()), float(rc.mean()), float(f1.mean()), float((f1*sup).sum()/sup.sum()), len(df)))
        cur.execute("DELETE FROM Model_Class_Metrics WHERE model_name=%s", (model,))
        for i, sentiment in enumerate(["Negatif","Netral","Positif"]):
            cur.execute("INSERT INTO Model_Class_Metrics VALUES (%s,%s,%s,%s,%s,%s,%s)",
                        (model,i,sentiment,float(pr[i]),float(rc[i]),float(f1[i]),int(sup[i])))
        cur.execute("DELETE FROM Model_Confusion_Matrix WHERE model_name=%s", (model,))
        cm = confusion_matrix(y, p, labels=LABELS)
        for a in LABELS:
            for pp in LABELS:
                cur.execute("INSERT INTO Model_Confusion_Matrix VALUES (%s,%s,%s,%s)", (model,a,pp,int(cm[a,pp])))
    dst.commit()

    a = data["SVM"].set_index("review_id")
    b = data["XLM-RoBERTa"].set_index("review_id")
    common = a.index.intersection(b.index)
    if len(common) != len(a) or len(common) != len(b):
        raise RuntimeError("SVM dan XLM-RoBERTa tidak memiliki reviewId yang sama lengkap untuk McNemar.")
    y = a.loc[common,"actual_label"].to_numpy()
    s_ok = a.loc[common,"predicted_label"].to_numpy() == y
    x_ok = b.loc[common,"predicted_label"].to_numpy() == y
    aa = int((s_ok & x_ok).sum()); bb = int((s_ok & ~x_ok).sum()); cc = int((~s_ok & x_ok).sum()); dd = int((~s_ok & ~x_ok).sum())
    pval = float(binomtest(min(bb,cc), bb+cc, .5).pvalue) if bb+cc else 1.0
    cur.execute("REPLACE INTO Model_Comparison VALUES (1,'SVM','XLM-RoBERTa',%s,%s,%s,%s,%s,'McNemar exact (binomial dua sisi)',%s,NOW())",
                (aa,bb,cc,dd,pval,len(common)))
    dst.commit(); cur.close()
    print(f"[4/4] Prediksi + evaluasi tersimpan: SVM={len(data['SVM']):,}, XLM-RoBERTa={len(data['XLM-RoBERTa']):,}; McNemar p={pval:.6f}")


def update_application_names(dst):
    cur = dst.cursor()
    try:
        cur.execute("UPDATE Dim_Application SET app_name='Kredivo' WHERE app_code='X'")
        cur.execute("UPDATE Dim_Application SET app_name='AdaKami' WHERE app_code='Y'")
        cur.execute("UPDATE Dim_Application SET app_name='Indodana' WHERE app_code='Z'")
        dst.commit()
    except mysql.connector.Error:
        dst.rollback()
        print("[info] Pembaruan nama aplikasi dilewati: kolom app_name/app_code tidak cocok dengan schema sumber.")
    finally:
        cur.close()


def main():
    ap = argparse.ArgumentParser(
        description="Migrasi MySQL lokal + hasil prediksi model ke TiDB Cloud."
    )
    ap.add_argument(
        "--svm",
        default="svm_test_predictions.csv",
        help="Path ke CSV prediksi SVM (default: svm_test_predictions.csv)",
    )
    ap.add_argument(
        "--xlm",
        default="xlm_test_predictions.csv",
        help="Path ke CSV prediksi XLM-RoBERTa (default: xlm_test_predictions.csv)",
    )
    ap.add_argument(
        "--replace",
        action="store_true",
        help="Kosongkan tabel yang sama di TiDB sebelum menyalin data.",
    )
    ap.add_argument(
        "--db-only",
        action="store_true",
        help="Hanya migrasikan database MySQL lokal ke TiDB, tanpa CSV model.",
    )
    args = ap.parse_args()

    src_cfg = read_cfg("source_mysql")
    dst_cfg = read_cfg("mysql")

    src = connect(src_cfg)
    dst = connect(dst_cfg)
    try:
        print("TiDB target:", dst_cfg["host"], "/", dst_cfg["database"])

        tables = migrate_tables(src, dst, args.replace)
        update_application_names(dst)

        if not args.db_only:
            svm_path = Path(args.svm)
            xlm_path = Path(args.xlm)

            if not svm_path.is_file():
                raise FileNotFoundError(
                    f"File prediksi SVM tidak ditemukan: {svm_path.resolve()}"
                )
            if not xlm_path.is_file():
                raise FileNotFoundError(
                    f"File prediksi XLM-RoBERTa tidak ditemukan: {xlm_path.resolve()}"
                )

            store_predictions(dst, svm_path, xlm_path)
        else:
            print("[3/4] Mode --db-only: CSV prediksi tidak dimigrasikan.")

        print("[VALIDASI] Jumlah baris source vs TiDB:")
        for table in tables:
            sc = src.cursor()
            tc = dst.cursor()
            try:
                sc.execute(f"SELECT COUNT(*) FROM {qi(table)}")
                s_count = int(sc.fetchone()[0])
                tc.execute(f"SELECT COUNT(*) FROM {qi(table)}")
                t_count = int(tc.fetchone()[0])
            finally:
                sc.close()
                tc.close()

            status = "OK" if s_count == t_count else "BERBEDA"
            print(f"  {table}: {s_count:,} -> {t_count:,} [{status}]")
            if s_count != t_count:
                raise RuntimeError(f"Validasi gagal pada tabel {table}.")

        print("\nMIGRASI DATABASE SELESAI. Dashboard selanjutnya dapat diarahkan ke TiDB Cloud.")
    finally:
        src.close()
        dst.close()


if __name__ == "__main__":
    main()
