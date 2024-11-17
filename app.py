from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import os

app = Flask(__name__)

# Tentukan folder untuk menyimpan gambar yang diunggah
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load model yang sudah dilatih
model = load_model('model.h5')

# Kelas sesuai dengan urutan kelas pada training
class_labels = ['biological', 'cardboard', 'clothes', 'glass', 'metal', 'paper', 'plastic', 'shoes', 'trash']

# Fungsi untuk memproses gambar dan melakukan prediksi
def model_predict(image_path, model):
    img = load_img(image_path, target_size=(128, 128))  # Sesuaikan ukuran dengan yang digunakan saat training
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)  # Ubah menjadi bentuk batch
    img = img / 255.0  # Normalisasi gambar seperti yang dilakukan di training

    pred = model.predict(img)
    pred_class = np.argmax(pred, axis=1)
    return class_labels[pred_class[0]]

# Halaman utama untuk mengunggah gambar
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Jika ada file yang diunggah
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']
        
        # Jika pengguna tidak memilih file
        if file.filename == '':
            return redirect(request.url)
        
        if file:
            # Simpan file yang diunggah ke folder uploads
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)

            # Lakukan prediksi menggunakan model yang dilatih
            result = model_predict(file_path, model)
            if result in ['biological', 'cardboard', 'paper']:
                result = "Organik"
            else:
                result = "Non Organik"

            # Tampilkan hasil di halaman web
            return render_template('index.html', result=result, image_url=file_path)
    
    return render_template('index.html', result=None)

if __name__ == "__main__":
    app.run(debug=True)
