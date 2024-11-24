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

# Dictionary untuk mapping kelas ke kategori
class_to_category = {
    'biological': 'Organik',
    'cardboard': 'Organik',
    'paper': 'Organik',
    'clothes': 'Non Organik',
    'glass': 'Non Organik',
    'metal': 'Non Organik',
    'plastic': 'Non Organik',
    'shoes': 'Non Organik',
    'trash': 'Non Organik'
}

# Fungsi untuk memproses gambar dan melakukan prediksi
def model_predict(image_path, model):
    img = load_img(image_path, target_size=(128, 128))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0

    # Dapatkan prediksi
    predictions = model.predict(img)
    
    # Dapatkan indeks kelas dengan probabilitas tertinggi
    pred_class_index = np.argmax(predictions, axis=1)[0]
    
    # Dapatkan probabilitas untuk kelas yang diprediksi
    confidence = predictions[0][pred_class_index] * 100
    
    # Dapatkan label kelas
    predicted_class = class_labels[pred_class_index]
    
    # Dapatkan kategori (Organik/Non Organik)
    category = class_to_category[predicted_class]
    
    # Hitung total confidence untuk masing-masing kategori
    organik_confidence = sum(predictions[0][i] for i, label in enumerate(class_labels) 
                           if class_to_category[label] == 'Organik') * 100
    non_organik_confidence = sum(predictions[0][i] for i, label in enumerate(class_labels) 
                                if class_to_category[label] == 'Non Organik') * 100
    
    return {
        'category': category,
        'specific_class': predicted_class,
        'class_confidence': confidence,
        'organik_confidence': organik_confidence,
        'non_organik_confidence': non_organik_confidence
    }

# Halaman utama untuk mengunggah gambar
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']
        
        if file.filename == '':
            return redirect(request.url)
        
        if file:
            # Simpan file yang diunggah
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)

            # Lakukan prediksi
            result = model_predict(file_path, model)
            
            return render_template('index.html',
                result=result['category'],
                specific_class=result['specific_class'],
                class_confidence=result['class_confidence'],
                organik_confidence=result['organik_confidence'],
                non_organik_confidence=result['non_organik_confidence'],
                image_url=file_path)
    
    return render_template('index.html', result=None)

if __name__ == "__main__":
    app.run(debug=True)