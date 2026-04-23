import streamlit as st
import numpy as np
from PIL import Image
import onnxruntime as ort
import os

st.set_page_config(
    page_title="Klasifikasi Kesegaran Ikan",
    page_icon="🐟",
    layout="wide",
)

IMG_SIZE = (224, 224)
CLASS_NAMES = ["Busuk", "Fresh", "Semi Fresh"]

MODEL_PATHS = {
    "CNN"         : "models/cnn_model.onnx",
    "EfficientNet": "models/efficientnet_model.onnx",
    "ResNet50"    : "models/resnet50_model.onnx",
}

@st.cache_resource
def load_model(path: str):
    if not os.path.exists(path):
        return None
    return ort.InferenceSession(path)

def preprocess(image: Image.Image) -> np.ndarray:
    img = image.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32)
    return np.expand_dims(arr, 0)

st.title("🐟 Klasifikasi Kesegaran Ikan")
st.markdown("Upload gambar ikan lalu pilih model untuk melihat prediksi tingkat kesegaran.")

st.sidebar.header("⚙️ Pengaturan")
selected_model_name = st.sidebar.selectbox("Pilih Model", list(MODEL_PATHS.keys()))
show_confidence = st.sidebar.checkbox("Tampilkan grafik confidence", value=True)

uploaded_file = st.file_uploader(
    "Upload gambar ikan (JPG / PNG / JPEG)",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file:
    col1, col2 = st.columns([1, 1])
    image = Image.open(uploaded_file)

    with col1:
        st.image(image, caption="Gambar yang diupload", use_container_width=True)

    with col2:
        model_path = MODEL_PATHS[selected_model_name]
        session = load_model(model_path)

        if session is None:
            st.error(f"Model {selected_model_name} tidak ditemukan di {model_path}.")
        else:
            with st.spinner("Memproses gambar..."):
                input_arr = preprocess(image)
                input_name = session.get_inputs()[0].name
                preds = session.run(None, {input_name: input_arr})[0][0]
                pred_idx = int(np.argmax(preds))
                pred_class = CLASS_NAMES[pred_idx]
                confidence = float(preds[pred_idx]) * 100

            color = {"Fresh": "🟢", "Semi Fresh": "🟡", "Busuk": "🔴"}
            icon = color.get(pred_class, "⚪")

            st.success(f"### {icon} Prediksi: **{pred_class}**")
            st.metric("Confidence", f"{confidence:.2f}%")

            if show_confidence:
                import pandas as pd
                st.markdown("#### Distribusi Probabilitas")
                df = pd.DataFrame(
                    {"Kelas": CLASS_NAMES, "Probabilitas (%)": preds * 100}
                ).set_index("Kelas")
                st.bar_chart(df)

st.divider()
with st.expander("ℹ️ Informasi Model"):
    st.markdown("""
| Model | Arsitektur | Keterangan |
|---|---|---|
| **CNN** | Custom 3-layer Conv | Dilatih dari nol dengan augmentasi |
| **EfficientNet** | EfficientNetB0 + fine-tuning | Transfer learning |
| **ResNet50** | ResNet50 | Transfer learning |

**Kelas:** Busuk 🔴 · Fresh 🟢 · Semi Fresh 🟡
    """)
