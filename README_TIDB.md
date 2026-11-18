# Migrasi Dashboard ke TiDB Cloud

## 1. Siapkan Secrets

Buat `.streamlit/secrets.toml` berdasarkan `.streamlit_secrets.example.toml`.
Isi password lokal dan TiDB Cloud hanya di komputer/deployment environment.

## 2. Migrasikan database lokal + hasil model

Jalankan dari folder dashboard:

```bash
python migrate_to_tidb.py --svm svm_test_predictions.csv --xlm xlm_test_predictions.csv
```

Untuk mengosongkan tabel yang sama di TiDB lalu menyalin ulang:

```bash
python migrate_to_tidb.py --svm svm_test_predictions.csv --xlm xlm_test_predictions.csv --replace
```

Script akan:
- membuat seluruh tabel BASE TABLE dari MySQL lokal di TiDB;
- menyalin seluruh isi tabel;
- memvalidasi jumlah baris source vs TiDB;
- membuat tabel `Model_Predictions`, `Model_Evaluation`, `Model_Class_Metrics`, `Model_Confusion_Matrix`, `Model_Comparison`;
- memasukkan hasil prediksi SVM dan XLM-RoBERTa;
- menghitung dan menyimpan metrik evaluasi serta McNemar;
- memastikan reviewId prediksi ada di `Fact_Reviews`.

## 3. Jalankan dashboard

```bash
streamlit run app.py
```

`app.py` sekarang hanya membaca data penelitian, prediksi model, pengguna, dan ulasan tambahan dari TiDB Cloud. Tidak ada CSV fallback.

## 4. Deployment Streamlit Community Cloud

Upload `app.py`, `db.py`, `mysql_source.py`, `admin_panel.py`, `requirements.txt`, dan folder `.streamlit` sesuai kebutuhan deployment. Masukkan secrets melalui menu Secrets Streamlit Cloud, bukan ke repository.

## Catatan

CSV prediksi SVM/XLM hanya dipakai sekali oleh `migrate_to_tidb.py` sebagai sumber migrasi. Setelah tabel model terisi, dashboard tidak membaca CSV tersebut lagi.
