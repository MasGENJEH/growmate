# Import library Flask untuk membuat web server, CORS untuk akses dari domain lain
from flask import Flask, request, jsonify
from flask_cors import CORS

# Import library TensorFlow dan utilitas untuk memproses gambar
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import json

# Inisialisasi aplikasi Flask
app = Flask(__name__, static_folder='../../front-end/(selanjutnya saya tak tahu)', static_url_path='/')
CORS(app)

# Aktifkan CORS supaya frontend bisa akses API meskipun beda port/domain
CORS(app)

# Load model deep learning yang sudah dilatih sebelumnya
model = load_model('saved_model/model_hama.h5')

# Load file JSON yang berisi label klasifikasi (misal: 'ulat grayak', 'belalang', dll.)
with open('saved_model/class_labels.json', 'r') as f:
    class_labels = json.load(f)

# Endpoint default '/' untuk mengirimkan file index.html ke browser
@app.route('/')
def serve_index():
    return app.send_static_file('index.html')  # ganti yang sesuai yaaa wkwk

# Endpoint '/predict' untuk melakukan klasifikasi gambar
@app.route('/predict', methods=['POST'])
def predict():
    # Periksa apakah request mengandung file
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    # Ambil file dari request
    file = request.files['file']

    # Simpan file sementara di folder 'uploads'
    filepath = os.path.join('uploads', file.filename)
    os.makedirs('uploads', exist_ok=True)  # Pastikan folder uploads ada
    file.save(filepath)

    # Baca gambar dan ubah ukurannya agar sesuai input model
    img = image.load_img(filepath, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0  # Normalisasi nilai piksel 0-1
    img_array = np.expand_dims(img_array, axis=0)  # Tambahkan dimensi batch

    # Lakukan prediksi menggunakan model
    prediction = model.predict(img_array)

    # Ambil index kelas dengan probabilitas tertinggi
    class_index = int(np.argmax(prediction))

    # Ambil nama kelas dari file label
    class_name = class_labels[class_index]

    # Kirimkan hasil prediksi ke frontend dalam format JSON
    return jsonify({'prediction': class_name})

# Jalankan aplikasi di mode debug (tidak untuk production)
if __name__ == '__main__':
    app.run(debug=True)
