#  Panduan Deploy Streamlit

## Langkah 1 – Simpan model dari notebook

Tambahkan kode berikut di **akhir notebook** sebelum deploy:

```python
import os
os.makedirs("models", exist_ok=True)

cnn_model.save("models/cnn_model.h5")
eff_model.save("models/eff_model.h5")
res_model.save("models/res_model.h5")
```

## Langkah 2 – Sesuaikan nama kelas

Buka `app.py`, cari baris:

```python
CLASS_NAMES = ["Kelas_1", "Kelas_2", "Kelas_3"]
```

Ganti sesuai nama folder kelas di datasetmu, contoh:

```python
CLASS_NAMES = ["Matang", "Mentah", "Busuk"]
```

## Langkah 3 – Struktur folder

Pastikan folder proyekmu seperti ini:

```
my_project/
├── app.py
├── requirements.txt
└── models/
    ├── cnn_model.h5
    ├── eff_model.h5
    └── res_model.h5
```

## Langkah 4 – Jalankan lokal (opsional)

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Langkah 5 – Deploy ke Streamlit Cloud (gratis)

1. Upload semua file ke **GitHub** (repo publik atau privat)
2. Buka [share.streamlit.io](https://share.streamlit.io)
3. Klik **"New app"** → pilih repo → pilih `app.py`
4. Klik **Deploy** ✅

> ⚠️ **Catatan ukuran file**: File `.h5` model bisa besar (>100MB).  
> Jika melebihi batas GitHub (100MB), gunakan **Git LFS** atau simpan model di Google Drive  
> lalu load dengan `gdown` di dalam `app.py`.
