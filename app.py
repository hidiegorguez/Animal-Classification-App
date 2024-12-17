import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import skimage.transform
import pickle
import tensorflow as tf
import tensorflow_hub as hub

from flask import Flask, request, render_template

app=Flask(__name__)

UPLOAD_FOLDER = 'upload/'
PICKLE_FOLDER = 'pickle_files/'
MODEL_FOLDER  = 'model/'
MODEL_NAME = 'model.h5'
    
model=tf.keras.models.load_model(MODEL_FOLDER + MODEL_NAME)
model_tf=tf.keras.models.load_model(MODEL_FOLDER + MODEL_NAME, custom_objects={'KerasLayer':hub.KerasLayer})

MODEL=model
IMG_WIDTH=80 #240
IMG_HEIGHT=80 #240

with open (PICKLE_FOLDER+'diego_mapeo.pkl','rb') as f:
    mapping=pickle.load(f)
    
@app.route('/')
def index():
    return render_template('form.html')


@app.route('/upload', methods=['POST'])
def uploader():
    if request.method=='POST':
        
        #getting the image extension
        f=request.files['archivo']
        filename=f.filename
        ext=filename.split('.')[-1]
        
        #checking extension
        if ext in ['jpg','png','jpeg']:
            #saving the image
            save_path=f'{UPLOAD_FOLDER}/{filename}'
            f.save(save_path)
            
            #getting the model prediction
            results=model_pipeline(save_path,mapping,model)
            return render_template('form.html',fileupload=True,data=results,image_filename=filename)
        
        return '<H1> Only JPEG, JPG and PNG files <H1>'
    return '<H1> Only POST methods <H1>'

#function to make the prediction
def model_pipeline(file_path,mapping,model):
    img=plt.imread(file_path)
    img=skimage.transform.resize(img,[IMG_HEIGHT,IMG_WIDTH])
    img=np.array([img])
    predict=MODEL.predict(img)
    return pd.Series(np.round(predict[0],2),index=mapping.values()).sort_values(ascending=False)[:5].to_dict()



if __name__ == '__main__':
    app.run()
