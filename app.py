import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import os

# ─────────────────────────────────────────
# Konfigurasi halaman
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Deep Learning Image Classifier",
    page_icon="🧠",
    layout="wide",
)

# ─────────────────────────────────────────
# Konstanta
# ─────────────────────────────────────────
IMG_SIZE = (224, 224)

# Ganti sesuai nama folder kelas di datasetmu
CLASS_NAMES = ["Kelas_1", "Kelas_2", "Kelas_3"]   # ← sesuaikan!

MODEL_PATHS = {
    "CNN"         : "models/cnn_model.h5",
    "EfficientNet": "models/eff_model.h5",
    "ResNet50"    : "models/res_model.h5",
}

# ─────────────────────────────────────────
# Load model (cache agar tidak reload ulang)
# ─────────────────────────────────────────
@st.cache_resource
def load_model(path: str):
    if not os.path.exists(path):
        return None
    return tf.keras.models.load_model(path)

# ─────────────────────────────────────────
# Helper: preprocess gambar
# ─────────────────────────────────────────
def preprocess(image: Image.Image) -> np.ndarray:
    img = image.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32)          # model CNN pakai Rescaling layer
    return np.expand_dims(arr, 0)                   # (1, 224, 224, 3)

# ─────────────────────────────────────────
# UI
# ─────────────────────────────────────────
st.title("🧠 Deep Learning Image Classifier")
st.markdown(
    "Upload gambar lalu pilih model untuk melihat prediksi kelas. "
    "Pastikan file model `.h5` sudah disimpan di folder `models/`."
)

# Sidebar – pilih model
st.sidebar.header("⚙️ Pengaturan")
selected_model_name = st.sidebar.selectbox("Pilih Model", list(MODEL_PATHS.keys()))
show_confidence = st.sidebar.checkbox("Tampilkan grafik confidence", value=True)

# Upload gambar
uploaded_file = st.file_uploader(
    "Upload gambar (JPG / PNG / JPEG)",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file:
    col1, col2 = st.columns([1, 1])

    image = Image.open(uploaded_file)
    with col1:
        st.image(image, caption="Gambar yang diupload", use_container_width=True)

    with col2:
        model_path = MODEL_PATHS[selected_model_name]
        model = load_model(model_path)

        if model is None:
            st.error(
                f"❌ Model **{selected_model_name}** tidak ditemukan di `{model_path}`.\n\n"
                "Pastikan kamu sudah menyimpan model dengan kode:\n"
                "```python\n"
                f"# Contoh untuk {selected_model_name}\n"
                "model.save('models/nama_model.h5')\n"
                "```"
            )
        else:
            with st.spinner("Memproses gambar..."):
                input_arr = preprocess(image)
                preds = model.predict(input_arr)[0]          # (3,)
                pred_idx = int(np.argmax(preds))
                pred_class = CLASS_NAMES[pred_idx]
                confidence = float(preds[pred_idx]) * 100

            st.success(f"### ✅ Prediksi: **{pred_class}**")
            st.metric("Confidence", f"{confidence:.2f}%")

            if show_confidence:
                st.markdown("#### Distribusi Probabilitas")
                import pandas as pd
                df = pd.DataFrame(
                    {"Kelas": CLASS_NAMES, "Probabilitas (%)": preds * 100}
                ).set_index("Kelas")
                st.bar_chart(df)

# ─────────────────────────────────────────
# Bagian bawah – info model
# ─────────────────────────────────────────
st.divider()
with st.expander("ℹ️ Informasi Model"):
    st.markdown(
        """
| Model | Arsitektur | Keterangan |
|---|---|---|
| **CNN** | Custom 3-layer Conv | Dilatih dari nol dengan augmentasi |
| **EfficientNet** | EfficientNetB0 + fine-tuning | Transfer learning, 20 layer terakhir di-unfreeze |
| **ResNet50** | ResNet50 | Transfer learning, feature extraction |

**Kelas yang diklasifikasikan:** {}
        """.format(", ".join(CLASS_NAMES))
    )
