# Import library yang dibutuhkan
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
import numpy as np
import json
from PIL import Image
import io

# Inisialisasi aplikasi FastAPI
app = FastAPI()

# Mengaktifkan CORS agar API dapat diakses dari frontend (berbeda origin)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Mengizinkan semua domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Path ke model dan label
MODEL_PATH = 'saved_model/model_hama.h5'
LABELS_PATH = 'saved_model/class_labels.json'

# Load model klasifikasi
model = load_model(MODEL_PATH)

# Load daftar label kelas dari file JSON
with open(LABELS_PATH, 'r') as f:
    class_labels = json.load(f)

# Endpoint untuk melakukan prediksi gambar
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Menerima file gambar, melakukan praproses, 
    lalu mengembalikan hasil klasifikasi.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Empty file name")

    try:
        # Membaca isi file gambar dari request
        contents = await file.read()
        
        # Membuka gambar menggunakan PIL dan memprosesnya
        img = Image.open(io.BytesIO(contents))
        img = img.convert("RGB")             # Konversi ke RGB (3 channel)
        img = img.resize((224, 224))         # Resize sesuai input model

        # Konversi gambar ke array dan normalisasi piksel
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)  # Tambah dimensi batch

        # Melakukan prediksi
        prediction = model.predict(img_array)
        class_index = int(np.argmax(prediction))       # Ambil index hasil tertinggi
        class_name = class_labels[class_index]         # Ambil nama kelas dari label

        # Kembalikan hasil prediksi
        return {
            "prediction": class_name,
        }

    except Exception as e:
        # Tangani error jika prediksi gagal
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")