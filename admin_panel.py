"""Halaman Admin Panel: tambah data ulasan, kelola data tambahan, dan lihat pengguna."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

import db

BASE_DIR = Path(__file__).resolve().parent
REQUIRED = ["app_name", "at", "content", "score"]


@st.cache_resource(show_spinner=False)
def load_svm_model():
    """Cari model SVM tersimpan (.joblib/.pkl). Kembalikan None kalau tidak ada atau tidak cocok."""
    try:
        import joblib
    except ImportError:
        return None
    roots = [BASE_DIR, BASE_DIR.parent, BASE_DIR.parent / "output_etl"]
    for root in roots:
        if not root.exists():
            continue
        for pat in ("*svm*.joblib", "*svm*.pkl", "*svm*.pickle"):
            for f in sorted(root.rglob(pat)) if root.name == "output_etl" else sorted(root.glob(pat)):
                try:
                    m = joblib.load(f)
                    if hasattr(m, "predict"):
                        return m
                except Exception:
                    continue
    return None


def predict_svm(texts: list[str]):
    model = load_svm_model()
    if model is None:
        return None
    try:
        pred = model.predict(texts)
        rev = {v: k for k, v in db.LABEL_TO_SENT.items()}
        return [int(p) if str(p).lstrip("-").isdigit() else rev.get(str(p).lower()) for p in pred]
    except Exception:
        return None


def _prepare(df: pd.DataFrame) -> tuple[pd.DataFrame | None, str | None]:
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        return None, f"Kolom wajib tidak ada: {', '.join(missing)}"
    out = df.copy()
    out["app_name"] = out["app_name"].astype(str).str.strip()
    out["score"] = pd.to_numeric(out["score"], errors="coerce")
    out["at"] = pd.to_datetime(out["at"], errors="coerce")
    out["content"] = out["content"].fillna("").astype(str)
    bad = out["score"].isna() | ~out["score"].isin([1, 2, 3, 4, 5]) | out["at"].isna() | (out["content"].str.strip() == "")
    if bad.all():
        return None, "Tidak ada baris valid (cek score 1-5, format tanggal, dan isi ulasan)."
    note = f"{int(bad.sum())} baris dibuang karena skor/tanggal/isi ulasan tidak valid." if bad.any() else None
    out = out[~bad].copy()
    out["score"] = out["score"].astype(int)
    out["clean_text"] = out["content"].map(db.clean_text)
    out = out[out["clean_text"].str.strip() != ""]
    preds = predict_svm(out["clean_text"].tolist())
    out["svm_pred"] = preds if preds is not None else None
    return out, note


def render(existing_ids: set[str], app_labels: dict[str, str], user: dict) -> None:
    st.markdown('<p class="page-title">Admin Panel</p>'
                '<p class="page-sub">Tambah data ulasan, kelola data tambahan, dan lihat pengguna.</p>',
                unsafe_allow_html=True)
    if msg := st.session_state.pop("flash", None):
        st.success(msg)

    tab_add, tab_data, tab_users = st.tabs(["Tambah Data", "List Data", "Pengguna"])

    # ---------------- tambah data ----------------
    with tab_add:
        st.markdown("**Unggah CSV ulasan baru**")
        st.caption("Kolom wajib: `app_name` (X/Y/Z), `at` (tanggal), `content` (teks ulasan), `score` (1-5). "
                   "Kolom `reviewId` opsional (dibuat otomatis kalau kosong).")
        up = st.file_uploader("File CSV", type="csv", key="csv_up")
        if up is not None:
            try:
                raw = pd.read_csv(up)
            except Exception as e:
                st.error(f"CSV tidak bisa dibaca: {e}")
                raw = None
            if raw is not None:
                prepared, note = _prepare(raw)
                if prepared is None:
                    st.error(note)
                else:
                    if note:
                        st.warning(note)
                    unknown = sorted(set(prepared["app_name"]) - set(app_labels))
                    if unknown:
                        st.warning(f"Kode aplikasi tidak dikenal: {', '.join(unknown)}. "
                                   f"Dashboard mengenal: {', '.join(app_labels)}.")
                    st.write(f"Pratinjau ({len(prepared)} baris valid):")
                    st.dataframe(prepared[["app_name", "at", "score", "content", "clean_text"]].head(10),
                                 hide_index=True, width="stretch")
                    if st.button("Simpan ke database", type="primary", key="save_csv"):
                        saved, skipped = db.add_reviews(prepared, user["email"], existing_ids)
                        st.session_state["flash"] = f"{saved} ulasan disimpan, {skipped} dilewati (reviewId sudah ada)."
                        st.rerun()

        st.markdown("---")
        st.markdown("**Atau tambah satu ulasan**")
        with st.form("manual", clear_on_submit=True):
            c1, c2, c3 = st.columns(3)
            app = c1.selectbox("Aplikasi", list(app_labels), format_func=lambda k: app_labels[k])
            tgl = c2.date_input("Tanggal ulasan")
            score = c3.slider("Rating (bintang)", 1, 5, 3)
            teks = st.text_area("Teks ulasan")
            if st.form_submit_button("Simpan ulasan"):
                row = pd.DataFrame([{"app_name": app, "at": pd.Timestamp(tgl), "content": teks, "score": score}])
                prepared, note = _prepare(row)
                if prepared is None or prepared.empty:
                    st.error("Teks ulasan kosong atau tidak valid.")
                else:
                    saved, _ = db.add_reviews(prepared, user["email"], existing_ids)
                    st.session_state["flash"] = f"{saved} ulasan disimpan."
                    st.rerun()

        if load_svm_model() is None:
            st.caption("Model SVM tersimpan (.joblib) tidak ditemukan, jadi hanya label dari rating yang dipakai. "
                       "Simpan modelnya dengan `joblib.dump(model, 'svm_model.joblib')` di train_svm.py untuk "
                       "mengaktifkan prediksi otomatis.")

    # ---------------- data tambahan ----------------
    with tab_data:
        new = db.load_new_reviews()
        if new.empty:
            st.info("Belum ada data tambahan.")
        else:
            st.metric("Jumlah ulasan tambahan", f"{len(new):,}".replace(",", "."))
            show = new[["reviewId", "app_name", "at", "score", "sentiment", "content", "added_by"]]
            sel = st.dataframe(show, hide_index=True, width="stretch",
                               on_select="rerun", selection_mode="multi-row", key="tbl_new")
            rows = sel.selection.rows if sel and sel.selection else []
            if rows and st.button(f"Hapus {len(rows)} baris terpilih"):
                n = db.delete_reviews(show.iloc[rows]["reviewId"].tolist())
                st.session_state["flash"] = f"{n} ulasan dihapus."
                st.rerun()

    # ---------------- pengguna ----------------
    with tab_users:
        users = db.list_users()
        a, b = st.columns(2)
        a.metric("Total pengguna terdaftar", len(users))
        b.metric("Admin", int((users["role"] == "admin").sum()) if len(users) else 0)
        st.dataframe(users, hide_index=True, width="stretch")
        st.download_button("Unduh daftar pengguna (CSV)", users.to_csv(index=False).encode("utf-8-sig"),
                           "pengguna_dashboard.csv", "text/csv")
        st.caption("Data ini berisi email pribadi. Gunakan seperlunya untuk penelitian dan jangan dibagikan.")
