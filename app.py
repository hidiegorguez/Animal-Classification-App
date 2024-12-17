import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import skimage.transform
import pickle
import tensorflow as tf
import tensorflow_hub as hub
from flask import Flask, request, render_template, url_for

# Configuración de Flask
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'upload/'
PICKLE_FOLDER = 'pickle_files/'
MODEL_FOLDER  = 'model/'
MODEL_NAME = 'model.h5'

# Cargar modelo y mapeo
model = tf.keras.models.load_model(MODEL_FOLDER + MODEL_NAME, custom_objects={'KerasLayer': hub.KerasLayer})
IMG_WIDTH = 80
IMG_HEIGHT = 80

with open(PICKLE_FOLDER + 'diego_mapeo.pkl', 'rb') as f:
    mapping = pickle.load(f)

# Crear carpeta de uploads si no existe
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/')
def index():
    return render_template('form.html')

@app.route('/upload', methods=['POST'])
def uploader():
    if request.method == 'POST':
        f = request.files['archivo']
        filename = f.filename
        ext = filename.split('.')[-1]
        
        if ext.lower() in ['jpg', 'png', 'jpeg']:
            save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            f.save(save_path)
            results = model_pipeline(save_path, mapping, model)
            return render_template('form.html', fileupload=True, data=results, image_filename=filename)
        return '<h1>Only JPEG, JPG, and PNG files allowed</h1>'
    return '<h1>Only POST methods allowed</h1>'

def model_pipeline(file_path, mapping, model):
    img = plt.imread(file_path)
    img = skimage.transform.resize(img, [IMG_HEIGHT, IMG_WIDTH])
    img = np.array([img])
    predict = model.predict(img)
    return pd.Series(np.round(predict[0], 2), index=mapping.values()).sort_values(ascending=False)[:5].to_dict()

if __name__ == '__main__':
    app.run(debug=True)
