"""
Dashboard Sentimen Paylater (Streamlit)
Halaman: Beranda | Perbandingan Aplikasi | Tren Waktu | Kata Kunci | Evaluasi Model

Jalankan dari folder ini:
    streamlit run app.py
"""
from __future__ import annotations

import io
import os
import re
import secrets
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from authlib.integrations.requests_client import OAuth2Session
from scipy.stats import binomtest, chi2
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from wordcloud import WordCloud

import admin_panel
import db
import mysql_source

# =====================================================================
# 1. KONFIGURASI  (ubah bagian ini kalau perlu)
# =====================================================================
BASE_DIR = Path(__file__).resolve().parent

# Nama file yang dicari otomatis (di folder dashboard, folder induknya,
# dan di dalam folder output_etl).
# Nama tampilan aplikasi. Kunci = isi kolom app_name di dataset.
# Ganti nilainya kalau mau memakai nama asli (misalnya "Kredivo").
APP_LABELS = {"X": "Kredivo", "Y": "AdaKami", "Z": "Indodana"}

SENTIMENTS = ["positif", "negatif", "netral"]
SENT_TITLE = {"positif": "Positif", "negatif": "Negatif", "netral": "Netral"}
LABEL_TO_SENT = {0: "negatif", 1: "netral", 2: "positif"}
COLORS = {"positif": "#1F9D6B", "negatif": "#E5484D", "netral": "#4F7CDB"}
SOFT = {"positif": "#E8F6EF", "negatif": "#FDECEC", "netral": "#EAF0FB"}

# Metrik evaluasi dihitung dari hasil prediksi yang tersimpan di TiDB Cloud.

STOPWORDS = set(
    """
    yang dan di ke dari ini itu dengan untuk pada adalah saya aku kamu anda kita kami
    mereka dia nya ya yg dg dgn utk tp tapi atau juga sudah udah belum sdh lagi masih
    sangat banget bgt aja saja hanya cuma karena krn jadi jd kalau kalo kl bisa bs
    ada akan mau mah sih deh dong lah kok nih tuh pun lalu terus trus sama dalam oleh
    agar supaya saat ketika sampai sampe hingga setelah sebelum bukan tidak gak nggak
    ga enggak tdk gk si para se
    """.split()
)

st.set_page_config(
    page_title="Sentimen Paylater",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =====================================================================
# 2. TAMPILAN (CSS)
# =====================================================================
st.markdown(
    """
<style>
    [data-testid="stSidebar"] { background: #1E2A44; }
    
        /* Tulisan menu navigasi: Beranda, Tren Waktu, dll. */
    [data-testid="stSidebar"] [data-testid="stRadio"] label,
    [data-testid="stSidebar"] [data-testid="stRadio"] label p,
    [data-testid="stSidebar"] [data-testid="stRadio"] label span {
        color: #FFFFFF !important;
    }

    /* Label Sumber Sentimen */
    [data-testid="stSidebar"] [data-testid="stSelectbox"] label {
        color: #FFFFFF !important;
    }

    /* Isi dropdown Sumber Sentimen */
    [data-testid="stSidebar"] [data-baseweb="select"] input {
        color: #000000 !important;
        -webkit-text-fill-color: #000000 !important;
    }

    [data-testid="stSidebar"] [data-baseweb="select"] span {
        color: #000000 !important;
    }

    /* Pilihan di dalam dropdown */
    [data-baseweb="popover"] [role="option"],
    [data-baseweb="popover"] [role="option"] * {
        color: #000000 !important;
    }

    /* Judul Sentimen Paylater */
    [data-testid="stSidebar"] .side-title {
        color: #FFFFFF !important;
        opacity: 1 !important;
    }

    /* Identitas Admin / Pengguna */
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"],
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p,
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] strong {
        color: #FFFFFF !important;
        opacity: 1 !important;
    }

    /* Email Admin / Pengguna */
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] a {
        color: #93C5FD !important;
        opacity: 1 !important;
    }
    /* Tombol Keluar */
    [data-testid="stSidebar"] button,
    [data-testid="stSidebar"] button * {
        color: #000000 !important;
    }

    /* Garis pembatas */
    [data-testid="stSidebar"] hr {
        border-color: #33415F;
    }
    [data-testid="stSidebar"] .stRadio label { padding: 6px 4px; }
    [data-testid="stSidebar"] hr { border-color: #33415f; }
    .side-title { font-size: 1.05rem; font-weight: 700; letter-spacing: .02em; margin-bottom: .2rem; }
    .side-foot { font-size: .75rem; letter-spacing: .12em; opacity: .55; margin-top: 2rem; }
    .block-container { padding-top: 4.5rem !important; max-width: 1280px; }
    [data-testid="stAppViewContainer"] .main .block-container {
        padding-top: 4.5rem !important;
        padding-bottom: 3rem !important;
    }
    header[data-testid="stHeader"] { height: 3.5rem; }
    .page-title { font-size: 2rem; font-weight: 800; margin: 0; color: #1B2236; }
    .page-sub { color: #5A6478; margin: 0 0 1rem 0; }
    .kpi { border: 1px solid #E3E7EF; border-radius: 14px; padding: 16px 18px; background: #fff; height: 100%; }
    .kpi .lbl { font-size: .9rem; color: #5A6478; display:flex; align-items:center; gap:8px; }
    .kpi .dot { width: 12px; height: 12px; border-radius: 50%; display:inline-block; }
    .kpi .val { font-size: 2.1rem; font-weight: 800; color: #1B2236; line-height: 1.2; margin-top: 6px; }
    .kpi .sub { font-size: .85rem; color: #5A6478; }
    .kwbox { border-radius: 10px; padding: 10px 12px 4px 12px; margin-bottom: 14px; border: 1px solid rgba(0,0,0,.06); }
    .kwbox .t { font-size: .8rem; font-weight: 600; margin-bottom: 4px; }
    .keyword-strip { display:flex; gap:14px; overflow-x:auto; padding:8px 2px 16px; }
    .keyword-card { flex:0 0 190px; min-width:190px; background:#fff; border:1px solid #E3E7EF;
        border-radius:14px; padding:16px; box-shadow:0 4px 14px rgba(20,35,60,.06); }
    .keyword-rank { font-size:.78rem; font-weight:700; color:#64748B; margin-bottom:12px; }
    .keyword-word { font-size:1.25rem; font-weight:800; color:#1B2236; overflow-wrap:anywhere; }
    .keyword-count { font-size:.88rem; color:#5A6478; margin-top:8px; }
    .keyword-count strong { color:#1B2236; }
</style>
""",
    unsafe_allow_html=True,
)


# =====================================================================
# 2b. LOGIN (st.login bawaan Streamlit + Google) DAN PERAN PENGGUNA
# =====================================================================
db.init_db()


def fmt(value):
    try:
        return f"{int(value):,}".replace(",", ".")
    except (TypeError, ValueError):
        return "0"

def pct(value):
    try:
        return f"{float(value):.1f}%"
    except (TypeError, ValueError):
        return "0.0%"

def app_label(code):
    return str(code)


def _secret(key, default=None):
    try:
        return st.secrets.get(key, default)
    except Exception:
        return default


def auth_configured() -> bool:
    try:
        return "auth" in st.secrets
    except Exception:
        return False


def github_configured() -> bool:
    try:
        return "github" in st.secrets and bool(st.secrets["github"].get("client_id")) and bool(st.secrets["github"].get("client_secret"))
    except Exception:
        return False


def _github_client(state: str | None = None) -> OAuth2Session:
    cfg = st.secrets["github"]
    return OAuth2Session(
        cfg["client_id"], cfg["client_secret"],
        scope="read:user user:email",
        redirect_uri=cfg["redirect_uri"],
        token_endpoint_auth_method="client_secret_post",
        state=state,
    )


def _github_user_from_callback() -> dict | None:
    """Tukar authorization code GitHub menjadi identitas pengguna terverifikasi."""
    if not github_configured() or "code" not in st.query_params:
        return None
    cfg = st.secrets["github"]
    returned_state = st.query_params.get("state", "")
    expected_state = st.session_state.get("github_oauth_state", "")
    if not expected_state or not secrets.compare_digest(str(returned_state), str(expected_state)):
        st.error("Validasi login GitHub gagal. Silakan coba masuk kembali.")
        st.query_params.clear()
        st.stop()
    try:
        client = _github_client(state=expected_state)
        callback_url = f'{cfg["redirect_uri"]}?code={st.query_params.get("code")}&state={returned_state}'
        client.fetch_token(
            "https://github.com/login/oauth/access_token",
            authorization_response=callback_url,
            grant_type="authorization_code",
        )
        profile = client.get("https://api.github.com/user").json()
        emails = client.get("https://api.github.com/user/emails").json()
        verified = next((e["email"] for e in emails if e.get("primary") and e.get("verified")), None)
        email = (verified or profile.get("email") or "").strip().lower()
        if not email:
            st.error("GitHub tidak menyediakan email terverifikasi. Atur email menjadi publik atau berikan izin email pada akun GitHub.")
            st.query_params.clear()
            st.stop()
        st.session_state["github_user"] = {
            "email": email,
            "name": profile.get("name") or profile.get("login") or email,
        }
        st.session_state.pop("github_oauth_state", None)
        st.query_params.clear()
        st.rerun()
    except Exception as exc:
        st.error(f"Login GitHub gagal: {exc}")
        st.query_params.clear()
        st.stop()
    return None


def login_gate() -> dict:
    """Login Google (OIDC) atau GitHub OAuth; hasilkan identitas dan peran pengguna."""
    admins = {str(e).strip().lower() for e in (_secret("admin_emails", []) or [])}

    # Selesaikan callback GitHub terlebih dahulu jika browser kembali dari GitHub.
    _github_user_from_callback()

    google_logged_in = auth_configured() and getattr(st.user, "is_logged_in", False)
    github_user = st.session_state.get("github_user")

    if not google_logged_in and not github_user:
        _, mid, _ = st.columns([1, 1.4, 1])
        with mid:
            st.markdown('<p class="page-title" style="margin-top:3rem">Sentimen Paylater</p>'
                        '<p class="page-sub">Dashboard analisis sentimen ulasan aplikasi paylater di Indonesia.</p>',
                        unsafe_allow_html=True)
            with st.container(border=True):
                st.markdown("**Masuk untuk melanjutkan**")
                if auth_configured():
                    st.button("Masuk dengan Google", on_click=st.login, type="primary", width="stretch")
                else:
                    st.caption("Login Google belum dikonfigurasi.")
                if github_configured():
                    if st.button("Masuk dengan GitHub", width="stretch"):
                        state = secrets.token_urlsafe(32)
                        st.session_state["github_oauth_state"] = state
                        oauth = _github_client(state=state)
                        auth_url, _ = oauth.create_authorization_url("https://github.com/login/oauth/authorize")
                        # Callback ditangani pada halaman utama Streamlit (root URL).
                        # Gunakan link_button agar perpindahan ke GitHub tidak bergantung
                        # pada JavaScript di iframe components.html.
                        st.link_button("Lanjutkan ke GitHub", auth_url, width="stretch")
                        st.info("Klik tombol di atas untuk melanjutkan proses masuk melalui GitHub.")
                        st.stop()
                else:
                    st.caption("Login GitHub akan aktif setelah konfigurasi OAuth GitHub ditambahkan.")
                st.caption("Dengan masuk, alamat email dan nama akun dicatat untuk keperluan penelitian skripsi.")
        st.stop()

    if google_logged_in:
        email = str(st.user.get("email", "")).strip().lower()
        if not email or st.user.get("email_verified") is False:
            st.error("Email akun Google belum terverifikasi.")
            st.button("Keluar", on_click=st.logout)
            st.stop()
        name = st.user.get("name") or email
        provider = "google"
    else:
        email = github_user["email"]
        name = github_user["name"]
        provider = "github"

    role = "admin" if email in admins else "user"
    if st.session_state.get("_recorded") != email:
        db.record_login(email, name, role)
        st.session_state["_recorded"] = email
    return {"email": email, "name": name, "role": role, "provider": provider}


USER = login_gate()
IS_ADMIN = USER["role"] == "admin"


# =====================================================================
# 3. MUAT DATA DARI TiDB CLOUD
# =====================================================================

if not mysql_source.configured():
    st.error("Konfigurasi TiDB Cloud belum tersedia. Isi [mysql] pada .streamlit/secrets.toml.")
    st.stop()

try:
    db.init_db()
    full_df = mysql_source.load_reviews()
    if full_df.empty:
        raise ValueError("Fact_Reviews di TiDB Cloud kosong.")
    svm_df = mysql_source.load_model_predictions("SVM")
    xlm_df = mysql_source.load_model_predictions("XLM-RoBERTa")
    new_df = db.load_new_reviews()
    DATA_SOURCE = "TiDB Cloud"
except Exception as exc:
    st.error(f"Tidak dapat membaca TiDB Cloud: {exc}")
    st.info("Dashboard tidak menggunakan CSV fallback. Periksa kredensial, TLS, tabel, dan data di TiDB Cloud.")
    st.stop()

# =====================================================================
# 4. SIDEBAR: NAVIGASI + SUMBER SENTIMEN
# =====================================================================

# =====================================================================
PAGES = ["Beranda", "Perbandingan Aplikasi", "Tren Waktu", "Kata Kunci", "Evaluasi Model"]
if IS_ADMIN:
    PAGES.append("Admin Panel")
ICONS = {"Beranda": "🏠", "Perbandingan Aplikasi": "📈", "Tren Waktu": "🗓️",
         "Kata Kunci": "🏷️", "Evaluasi Model": "✅", "Admin Panel": "🛠️"}
with st.sidebar:
    page = st.radio("Menu", PAGES, format_func=lambda p: f"{ICONS[p]}  {p}",
                    label_visibility="collapsed")
    st.caption("Sumber data: **TiDB Cloud**")
    st.markdown("---")

    sources = ["Label rating (seluruh data)"]
    if svm_df is not None and "app_name" in svm_df.columns:
        sources.append("Prediksi SVM (data uji)")
    if xlm_df is not None and "app_name" in xlm_df.columns:
        sources.append("Prediksi XLM-RoBERTa (data uji)")
    source = st.selectbox("Sumber sentimen", sources,
                          help="Label rating = sentimen dari bintang (1-2 negatif, 3 netral, 4-5 positif). "
                               "Prediksi = hasil model pada data uji saja.")
    include_new = False
    if not new_df.empty and source.startswith("Label"):
        include_new = st.checkbox(f"Sertakan data tambahan ({fmt(len(new_df))})", value=True)
    st.markdown("---")
    st.caption(f"{'Admin' if IS_ADMIN else 'Pengguna'}: **{USER['name']}**  \n{USER['email']}")
    if USER["provider"] == "google":
        st.button("Keluar", on_click=st.logout, width="stretch")
    else:
        if st.button("Keluar", width="stretch"):
            st.session_state.pop("github_user", None)
            st.session_state.pop("_recorded", None)
            st.rerun()

if source.startswith("Prediksi SVM"):
    base = svm_df.copy()
    base["sentiment"] = base["pred"].map(LABEL_TO_SENT)
elif source.startswith("Prediksi XLM"):
    base = xlm_df.copy()
    base["sentiment"] = base["pred"].map(LABEL_TO_SENT)
else:
    base = full_df.copy()
    if include_new:
        base = pd.concat([base, new_df.drop(columns=["added_by", "added_at", "svm_pred"], errors="ignore")],
                         ignore_index=True)
base = base.dropna(subset=["at"]).copy()

# Pastikan sumber data memiliki tanggal valid sebelum halaman dashboard dirender.
if base.empty or "date" not in base.columns or base["date"].isna().all():
    st.error(
        "Tidak ada data ulasan dengan tanggal yang valid. "
        "Periksa kolom 'at' pada dataset dan file prediksi yang digunakan."
    )
    st.stop()

# =====================================================================
# 5. FILTER TANGGAL (kanan atas)
# =====================================================================
min_d, max_d = base["date"].min(), base["date"].max()


def page_header(title: str, subtitle: str, use_dates: bool = True):
    left, right = st.columns([3, 1.25])
    with left:
        st.markdown(f'<p class="page-title">{title}</p><p class="page-sub">{subtitle}</p>',
                    unsafe_allow_html=True)
    if not use_dates:
        return min_d, max_d
    with right:
        val = st.date_input("Rentang tanggal", (min_d, max_d), min_value=min_d,
                            max_value=max_d, format="DD/MM/YYYY", key=f"dr_{title}")
    if isinstance(val, (tuple, list)) and len(val) == 2:
        return val[0], val[1]
    return min_d, max_d


def filter_dates(df: pd.DataFrame, d0, d1) -> pd.DataFrame:
    return df[(df["date"] >= d0) & (df["date"] <= d1)]


def kpi(col, label, value, sub, color=None):
    dot = f'<span class="dot" style="background:{color}"></span>' if color else ""
    col.markdown(
        f'<div class="kpi"><div class="lbl">{dot}{label}</div>'
        f'<div class="val">{value}</div><div class="sub">{sub}</div></div>',
        unsafe_allow_html=True,
    )


def counts_by_app(df: pd.DataFrame) -> pd.DataFrame:
    ct = pd.crosstab(df["app_name"], df["sentiment"]).reindex(columns=SENTIMENTS, fill_value=0)
    ct["total"] = ct.sum(axis=1)
    return ct.sort_index()


# =====================================================================
# 6. HALAMAN
# =====================================================================
def page_ikhtisar():
    d0, d1 = page_header("Beranda",
                         "Ringkasan hasil analisis sentimen pada layanan paylater di Indonesia.")
    df = filter_dates(base, d0, d1)
    if df.empty:
        st.warning("Tidak ada data pada rentang tanggal ini.")
        return
    total = len(df)
    c = df["sentiment"].value_counts().reindex(SENTIMENTS, fill_value=0)

    cols = st.columns(4)
    kpi(cols[0], "Total Ulasan", fmt(total), "seluruh aplikasi")
    for i, s in enumerate(SENTIMENTS, start=1):
        kpi(cols[i], f"Sentimen {SENT_TITLE[s]}", fmt(c[s]), f"{pct(c[s] / total * 100)} dari total", COLORS[s])

    st.write("")
    left, right = st.columns([1, 1.2])
    with left:
        with st.container(border=True):
            st.markdown("**Distribusi Sentimen Keseluruhan**")
            fig = go.Figure(go.Pie(
                labels=[SENT_TITLE[s] for s in SENTIMENTS], values=[c[s] for s in SENTIMENTS],
                hole=0.62, marker=dict(colors=[COLORS[s] for s in SENTIMENTS]),
                textinfo="percent", sort=False,
            ))
            fig.update_layout(height=340, margin=dict(l=0, r=0, t=10, b=0),
                              legend=dict(orientation="h", y=-0.05),
                              annotations=[dict(text=f"<b>{fmt(total)}</b><br>Total Ulasan",
                                                x=0.5, y=0.5, showarrow=False, font_size=15)])
            st.plotly_chart(fig, width="stretch")
    with right:
        with st.container(border=True):
            st.markdown("**Ringkasan per Aplikasi**")
            ct = counts_by_app(df)
            rows = []
            for code, r in ct.iterrows():
                rows.append({
                    "Aplikasi": app_label(code),
                    "Total Ulasan": fmt(r["total"]),
                    "Positif": f'{fmt(r["positif"])} ({pct(r["positif"] / r["total"] * 100)})',
                    "Negatif": f'{fmt(r["negatif"])} ({pct(r["negatif"] / r["total"] * 100)})',
                    "Netral": f'{fmt(r["netral"])} ({pct(r["netral"] / r["total"] * 100)})',
                })
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    if source.startswith("Label") and include_new:
        st.caption(f"Termasuk {fmt(len(new_df))} ulasan tambahan dari admin (tidak ikut dalam evaluasi model).")
    if source.startswith("Label"):
        st.caption("Dataset ini diseimbangkan lewat stratified sampling (jumlah rating sama), "
                   "sehingga komposisi sentimen mengikuti desain sampel, bukan komposisi populasi asli.")
    else:
        st.caption(f"Menampilkan hasil klasifikasi pada data uji ({fmt(len(base))} ulasan).")


def page_perbandingan():
    d0, d1 = page_header("Perbandingan Aplikasi",
                         "Perbandingan distribusi sentimen pada tiga aplikasi paylater.")
    df = filter_dates(base, d0, d1)
    if df.empty:
        st.warning("Tidak ada data pada rentang tanggal ini.")
        return
    ct = counts_by_app(df)
    with st.container(border=True):
        st.markdown("**Distribusi Sentimen per Aplikasi**")
        fig = go.Figure()
        names = [app_label(a) for a in ct.index]
        for s in SENTIMENTS:
            share = (ct[s] / ct["total"] * 100).values
            fig.add_bar(y=names, x=share, name=SENT_TITLE[s], orientation="h",
                        marker_color=COLORS[s], text=[pct(v) for v in share],
                        textposition="inside", insidetextanchor="middle",
                        textfont=dict(color="white"))
        fig.update_layout(barmode="stack", height=280, margin=dict(l=0, r=0, t=10, b=0),
                          xaxis=dict(range=[0, 100], ticksuffix="%"),
                          yaxis=dict(autorange="reversed"),
                          legend=dict(orientation="h", y=1.15, x=1, xanchor="right"))
        st.plotly_chart(fig, width="stretch")

    with st.container(border=True):
        st.markdown("**Detail per Aplikasi**")
        rows = []
        for code, r in ct.iterrows():
            rows.append({
                "Aplikasi": app_label(code),
                "Total Ulasan": fmt(r["total"]),
                "Positif": f'{fmt(r["positif"])} ({pct(r["positif"] / r["total"] * 100)})',
                "Negatif": f'{fmt(r["negatif"])} ({pct(r["negatif"] / r["total"] * 100)})',
                "Netral": f'{fmt(r["netral"])} ({pct(r["netral"] / r["total"] * 100)})',
                "Persentase Positif": pct(r["positif"] / r["total"] * 100),
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


def trend_figure(df: pd.DataFrame, mode: str, title: str) -> go.Figure:
    g = df.groupby(["month", "sentiment"]).size().unstack(fill_value=0).reindex(columns=SENTIMENTS, fill_value=0)
    if mode.startswith("Proporsi"):
        g = g.div(g.sum(axis=1), axis=0) * 100
        yfmt, ytitle = ".1f", "Proporsi (%)"
    else:
        yfmt, ytitle = ",d", "Jumlah ulasan"
    fig = go.Figure()
    for s in SENTIMENTS:
        fig.add_scatter(x=g.index, y=g[s], mode="lines+markers", name=SENT_TITLE[s],
                        line=dict(color=COLORS[s], width=2.5), marker=dict(size=7),
                        hovertemplate=f"{SENT_TITLE[s]}: %{{y:{yfmt}}}<extra></extra>")
    fig.update_layout(height=340, margin=dict(l=0, r=0, t=30, b=0), title=dict(text=title, font_size=14),
                      yaxis=dict(title=ytitle, rangemode="tozero"), hovermode="x unified",
                      legend=dict(orientation="h", y=1.12, x=1, xanchor="right"))
    return fig


def page_tren():
    d0, d1 = page_header("Tren Waktu", "Pergerakan sentimen dari waktu ke waktu (per bulan).")
    mode = st.radio("Tampilkan sebagai", ["Proporsi (%)", "Jumlah ulasan"], horizontal=True)
    df = filter_dates(base, d0, d1)
    if df.empty:
        st.warning("Tidak ada data pada rentang tanggal ini.")
        return
    if df["month"].nunique() < 2:
        st.info("Rentang tanggal kurang dari dua bulan, grafik tren belum informatif.")

    app_filter = st.selectbox("Aplikasi", ["Semua Aplikasi"] + [app_label(a) for a in sorted(df["app_name"].unique())])
    with st.container(border=True):
        sub = df if app_filter == "Semua Aplikasi" else df[df["app_name"].map(app_label) == app_filter]
        st.plotly_chart(trend_figure(sub, mode, "Tren Sentimen Berdasarkan Waktu"), width="stretch")

    with st.container(border=True):
        st.markdown("**Tren per Aplikasi**")
        codes = sorted(df["app_name"].unique())
        tabs = st.tabs([app_label(c) for c in codes])
        for tab, code in zip(tabs, codes):
            with tab:
                st.plotly_chart(trend_figure(df[df["app_name"] == code], mode,
                                             f"Tren Sentimen - {app_label(code)}"),
                                width="stretch", key=f"trend_{code}")

    peak = (df[df["sentiment"] == "negatif"].groupby("month").size())
    if len(peak):
        st.caption(f"Bulan dengan ulasan negatif terbanyak: **{peak.idxmax():%B %Y}** ({fmt(peak.max())} ulasan).")


@st.cache_data(show_spinner="Menghitung kata kunci...")
def word_freq(texts: tuple[str, ...], extra_stop: tuple[str, ...], min_len: int) -> Counter:
    stop = STOPWORDS | set(extra_stop)
    cnt: Counter = Counter()
    for t in texts:
        for w in re.findall(r"[a-z]+", t.lower()):
            if len(w) >= min_len and w not in stop:
                cnt[w] += 1
    return cnt


def cloud_image(freq: Counter, color: str):
    wc = WordCloud(width=1000, height=260, background_color=None, mode="RGBA",
                   max_words=40, prefer_horizontal=0.95, relative_scaling=0.5,
                   color_func=lambda *a, **k: color, random_state=42).generate_from_frequencies(freq)
    fig, ax = plt.subplots(figsize=(10, 2.6))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    fig.patch.set_alpha(0)
    plt.tight_layout(pad=0)
    return fig


def page_kata_kunci():
    d0, d1 = page_header(
        "Kata Kunci",
        "Peringkat 10 kata yang paling sering muncul pada ulasan berdasarkan kategori sentimen."
    )
    df = filter_dates(base, d0, d1)
    if "clean_text" not in df.columns or df.empty:
        st.warning("Data teks tidak tersedia pada sumber/rentang ini.")
        return

    c1, c2, c3 = st.columns([1.2, 1.4, 1])
    app_opts = ["Semua Aplikasi"] + [app_label(a) for a in sorted(df["app_name"].unique())]
    pick = c1.selectbox("Aplikasi", app_opts)
    extra = c2.text_input("Kata yang disembunyikan (pisahkan koma)", "")
    min_len = c3.slider("Panjang kata minimum", 2, 6, 3)

    if pick != "Semua Aplikasi":
        df = df[df["app_name"].map(app_label) == pick]
    extra_stop = tuple(w.strip().lower() for w in extra.split(",") if w.strip())

    for s in SENTIMENTS:
        texts = tuple(df.loc[df["sentiment"] == s, "clean_text"])
        freq = word_freq(texts, extra_stop, min_len)
        st.markdown(
            f'<div class="kwbox" style="background:{SOFT[s]}">'
            f'<div class="t" style="color:{COLORS[s]}">Top 10 Kata Kunci Sentimen '
            f'{SENT_TITLE[s]} ({fmt(len(texts))} ulasan)</div></div>',
            unsafe_allow_html=True,
        )

        top_words = freq.most_common(10)
        if not top_words:
            st.caption("Tidak ada kata yang dapat ditampilkan.")
            continue

        ranked_labels = [f"{i}. {word}" for i, (word, _) in enumerate(top_words, start=1)]
        counts = [count for _, count in top_words]
        fig = go.Figure(
            go.Bar(
                x=counts,
                y=ranked_labels,
                orientation="h",
                marker_color=COLORS[s],
                text=[fmt(v) for v in counts],
                textposition="outside",
                cliponaxis=False,
                hovertemplate="%{y}<br>Jumlah kemunculan: %{x:,}<extra></extra>",
            )
        )
        fig.update_layout(
            height=420,
            margin=dict(l=12, r=55, t=12, b=18),
            xaxis=dict(title="Jumlah kemunculan", rangemode="tozero"),
            yaxis=dict(title="", autorange="reversed", automargin=True),
            showlegend=False,
        )
        with st.container(border=True):
            st.plotly_chart(fig, width="stretch", key=f"keyword_chart_{s}")


# ---------- evaluasi model ----------
def metrics_from_preds(df: pd.DataFrame) -> dict:
    y, p = df["label"].to_numpy(), df["pred"].to_numpy()
    pr, rc, f1, sup = precision_recall_fscore_support(y, p, labels=[0, 1, 2], zero_division=0)
    return {"accuracy": accuracy_score(y, p), "precision": list(pr), "recall": list(rc),
            "f1": list(f1), "support": list(sup),
            "cm": confusion_matrix(y, p, labels=[0, 1, 2]).tolist()}


def mcnemar(svm: pd.DataFrame, xlm: pd.DataFrame):
    """Uji McNemar pada prediksi benar/salah kedua model untuk ulasan yang sama."""
    if "reviewId" in svm.columns and "reviewId" in xlm.columns:
        m = svm[["reviewId", "label", "pred"]].merge(
            xlm[["reviewId", "pred"]], on="reviewId", suffixes=("_svm", "_xlm"))
    elif len(svm) == len(xlm):
        m = pd.DataFrame({"label": svm["label"].values, "pred_svm": svm["pred"].values,
                          "pred_xlm": xlm["pred"].values})
    else:
        return None
    s_ok, x_ok = m["pred_svm"] == m["label"], m["pred_xlm"] == m["label"]
    b = int((s_ok & ~x_ok).sum())   # SVM benar, XLM salah
    c = int((~s_ok & x_ok).sum())   # SVM salah, XLM benar
    if b + c == 0:
        return {"b": b, "c": c, "n": len(m), "p": 1.0, "metode": "-"}
    if b + c < 25:
        p, metode = binomtest(min(b, c), b + c, 0.5).pvalue, "exact binomial"
    else:
        stat = (abs(b - c) - 1) ** 2 / (b + c)
        p, metode = float(chi2.sf(stat, 1)), "chi-square (koreksi kontinuitas)"
    return {"b": b, "c": c, "n": len(m), "p": p, "metode": metode}


def cm_figure(cm, title):
    names = ["Negatif", "Netral", "Positif"]
    cm = np.array(cm)
    fig = go.Figure(go.Heatmap(z=cm, x=names, y=names, colorscale="Blues", showscale=False,
                               text=cm, texttemplate="%{text}", hovertemplate="Aktual %{y} → Prediksi %{x}: %{z}<extra></extra>"))
    fig.update_layout(height=320, margin=dict(l=0, r=0, t=40, b=0), title=dict(text=title, font_size=14),
                      xaxis_title="Prediksi", yaxis_title="Aktual", yaxis=dict(autorange="reversed"))
    return fig


def page_evaluasi():
    page_header("Evaluasi Model",
                "Perbandingan kinerja SVM (TF-IDF) dan XLM-RoBERTa pada data uji yang sama.", use_dates=False)
    res = {}
    for name, d in (("SVM", svm_df), ("XLM-RoBERTa", xlm_df)):
        if d is None or d.empty:
            st.error(f"Hasil prediksi {name} belum tersedia di TiDB Cloud. Jalankan migrasi hasil model terlebih dahulu.")
            return
        res[name] = metrics_from_preds(d)
    st.caption(f"Evaluasi dihitung dari prediksi yang tersimpan di TiDB Cloud: SVM {fmt(len(svm_df))} data uji dan XLM-RoBERTa {fmt(len(xlm_df))} data uji.")

    rows = []
    for name, r in res.items():
        rows.append({"Model": name, "Accuracy": r["accuracy"],
                     "Precision (macro)": float(np.mean(r["precision"])),
                     "Recall (macro)": float(np.mean(r["recall"])),
                     "F1-Score (macro)": float(np.mean(r["f1"]))})
    summ = pd.DataFrame(rows)

    left, right = st.columns([1.1, 1])
    with left:
        with st.container(border=True):
            st.markdown("**Ringkasan Metrik**")
            st.dataframe(summ.style.format({c: "{:.4f}" for c in summ.columns if c != "Model"})
                         .highlight_max(subset=[c for c in summ.columns if c != "Model"], color="#DDF3E8"),
                         hide_index=True, width="stretch")
            best = summ.loc[summ["F1-Score (macro)"].idxmax(), "Model"]
            st.caption(f"Macro-F1 tertinggi: **{best}**. Sel hijau = nilai terbaik per metrik.")
    with right:
        with st.container(border=True):
            st.markdown("**Radar Chart**")
            cats = ["Accuracy", "Precision (macro)", "Recall (macro)", "F1-Score (macro)"]
            fig = go.Figure()
            palette = {"SVM": "#E08A1E", "XLM-RoBERTa": "#4F7CDB"}
            for _, r in summ.iterrows():
                v = [r[c] for c in cats]
                fig.add_scatterpolar(r=v + v[:1], theta=cats + cats[:1], fill="toself", name=r["Model"],
                                     line_color=palette[r["Model"]], opacity=0.75)
            fig.update_layout(height=340, margin=dict(l=40, r=40, t=20, b=20),
                              polar=dict(radialaxis=dict(range=[0.3, 0.8])),
                              legend=dict(orientation="h", y=-0.1))
            st.plotly_chart(fig, width="stretch")

    with st.container(border=True):
        st.markdown("**Kinerja per Kelas**")
        names = ["Negatif", "Netral", "Positif"]
        rows = []
        for name, r in res.items():
            for i, k in enumerate(names):
                rows.append({"Model": name, "Kelas": k, "Precision": r["precision"][i],
                             "Recall": r["recall"][i], "F1-Score": r["f1"][i], "Support": int(r["support"][i])})
        st.dataframe(pd.DataFrame(rows).style.format({"Precision": "{:.4f}", "Recall": "{:.4f}", "F1-Score": "{:.4f}"}),
                     hide_index=True, width="stretch")

    c1, c2 = st.columns(2)
    for col, name in zip((c1, c2), res):
        with col, st.container(border=True):
            if res[name]["cm"] is not None:
                st.plotly_chart(cm_figure(res[name]["cm"], f"Confusion Matrix - {name}"),
                                width="stretch", key=f"cm_{name}")
            else:
                st.caption(f"Confusion matrix {name} belum tersedia di TiDB Cloud.")

    with st.container(border=True):
        st.markdown("**Uji Signifikansi (McNemar)**")
        if svm_df is not None and xlm_df is not None:
            r = mcnemar(svm_df, xlm_df)
            if r is None:
                st.warning("Baris data uji kedua model tidak bisa disejajarkan (tidak ada reviewId dan jumlah baris berbeda).")
            else:
                a, b_, c_ = st.columns(3)
                a.metric("SVM benar, XLM-R salah", fmt(r["b"]))
                b_.metric("SVM salah, XLM-R benar", fmt(r["c"]))
                c_.metric("p-value", f'{r["p"]:.4g}')
                if r["p"] < 0.05:
                    st.success(f"Perbedaan akurasi signifikan secara statistik pada α = 0,05 ({r['metode']}, n = {fmt(r['n'])}).")
                else:
                    st.info(f"Perbedaan akurasi belum signifikan secara statistik pada α = 0,05 ({r['metode']}, n = {fmt(r['n'])}).")
        else:
            st.caption("Butuh kedua file prediksi untuk menjalankan uji McNemar.")


# =====================================================================
# 7. ROUTER
# =====================================================================
{"Beranda": page_ikhtisar, "Perbandingan Aplikasi": page_perbandingan, "Tren Waktu": page_tren,
 "Kata Kunci": page_kata_kunci, "Evaluasi Model": page_evaluasi,
 "Admin Panel": lambda: admin_panel.render(set(full_df["reviewId"].astype(str)), APP_LABELS, USER)}[page]() 