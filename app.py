import os
import warnings
warnings.simplefilter("ignore")
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np
import cv2


from flask import Flask, render_template, render_template_string, request, redirect, url_for, abort
from werkzeug.utils import secure_filename
from os import listdir

app = Flask(__name__)

#load the trained model
MODEL_PATH = "models/model1.h5"
model = load_model(MODEL_PATH)
class_names = ['Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight', 'Tomato_Leaf_Mold', 'Tomato_healthy']

def model_predict(img_path, model):

    img = image.load_img(img_path, target_size=(256, 256))

    # Preprocessing the image
    x = image.img_to_array(img)
    x=x/255
    x = np.expand_dims(x, axis=0)

    predictions = model.predict(x)
    label = class_names[np.argmax(predictions[0])]
    return label

@app.route('/', methods = ["GET"])
def index():
    return render_template('index.html')


@app.route('/predict',methods = ["GET", "POST"])
def upload():
    if request.method == 'POST':
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        result=preds
        return result
    return None        





if __name__ == "__main__":
    app.run(port=5001, debug=True)