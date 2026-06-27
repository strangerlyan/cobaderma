import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import os

from PIL import Image

from tensorflow.keras.applications.efficientnet import preprocess_input

# ======================
# LOAD MODEL
# ======================

BASE_PATH = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_PATH, "cnn_ham10000_final2.keras")
CLASS_PATH = os.path.join(BASE_PATH, "class_names.npy")
META_PATH = os.path.join(BASE_PATH, "meta_cols.npy")

# Load model
model = tf.keras.models.load_model(
    MODEL_PATH
)

# Load class
class_names = np.load(
    CLASS_PATH,
    allow_pickle=True
)

# Load metadata
meta_cols = list(
    np.load(
        META_PATH,
        allow_pickle=True
    )
)

# ======================
# PREPROCESS
# ======================

def preprocess(img):

    img = img.convert("RGB")

    img = img.resize((224,224))

    img = np.array(img)

    img = np.expand_dims(img,0)

    img = preprocess_input(
        img.astype("float32")
    )

    return img

# ======================
# METADATA
# ======================

def build_metadata(
        age,
        sex,
        localization
):

    meta = np.zeros(
        (1,len(meta_cols))
    )

    meta[0,0] = age

    if sex=="Male":

        meta[0,1]=0

    else:

        meta[0,1]=1

    loc="loc_"+localization

    if loc in meta_cols:

        idx=meta_cols.index(loc)

        meta[0,idx]=1

    return meta

# ======================
# UI
# ======================

st.title(
    "DermaVision: Platform Skrining Dini Kanker Kulit Berbasis Kecerdasan Buatan"
)

st.markdown(
    """
Aplikasi ini membantu memprediksi jenis lesi kulit berdasarkan **citra lesi**, **usia**, **jenis kelamin**, dan **lokasi lesi**.

**Catatan:** Hasil prediksi merupakan bantuan awal dan **bukan pengganti diagnosis oleh dokter**.
"""
)

uploaded = st.file_uploader(
    "Unggah Foto Lesi Kulit",
    type=["jpg","jpeg","png"],
    help="Unggah foto lesi kulit dengan format JPG, JPEG, atau PNG."
)

age = st.number_input(
    "Usia Pasien (tahun)",
    min_value=0,
    max_value=120,
    value=0,
    step=1,
    help="Masukkan usia pasien dalam satuan tahun."
)

jenis_kelamin = st.selectbox(
    "Jenis Kelamin",
    [
        "Pilih jenis kelamin",
        "Laki-laki",
        "Perempuan"
    ],
    help="Pilih jenis kelamin pasien."
)

if jenis_kelamin == "Pilih jenis kelamin":
    sex = None

elif jenis_kelamin == "Laki-laki":
    sex = "male"

else:
    sex = "female"

lokasi = st.selectbox(
    "Lokasi Lesi pada Tubuh",
    [
        "Pilih lokasi lesi",
        "Perut",
        "Punggung",
        "Dada",
        "Telinga",
        "Wajah",
        "Telapak Kaki",
        "Telapak Tangan",
        "Lengan",
        "Tungkai/Kaki",
        "Leher",
        "Kulit Kepala",
        "Batang Tubuh"
    ],
    help="Pilih lokasi munculnya lesi pada tubuh."
)

location_mapping = {
    "Perut":"abdomen",
    "Punggung":"back",
    "Dada":"chest",
    "Telinga":"ear",
    "Wajah":"face",
    "Telapak Kaki":"foot",
    "Telapak Tangan":"hand",
    "Lengan":"upper extremity",
    "Tungkai/Kaki":"lower extremity",
    "Leher":"neck",
    "Kulit Kepala":"scalp",
    "Batang Tubuh":"trunk"
}

# ======================
# DETECT
# ======================

if age == 0:
    st.warning("Silakan masukkan usia pasien.")
    st.stop()

if sex is None:
    st.warning(
        "Silakan pilih jenis kelamin pasien."
    )
    st.stop()

if lokasi == "Pilih lokasi lesi":
    st.warning(
        "Silakan pilih lokasi lesi pada tubuh."
    )
    st.stop()
localization = location_mapping[lokasi]

if uploaded is not None:
    img = Image.open(uploaded)
    st.image(
        img,
        width=300
    )

    if st.button("Deteksi"):
        img_array = preprocess(img)
        meta = build_metadata(
            age,
            sex,
            localization
        )

        pred = model.predict(
            [
                img_array,
                meta
            ],
            verbose=0
        )
        
        idx = np.argmax(pred)
        st.success(
            f"Hasil Prediksi : {class_names[idx].upper()}"
        )

        st.metric(
            "Tingkat Keyakinan Hasil",
            f"{pred[0][idx]*100:.2f}%"
        )

        df = pd.DataFrame({
            "Class":class_names,
            "Probability (%)":pred[0]*100
        })

        st.bar_chart(
            df.set_index("Class")
        )
        st.dataframe(df)