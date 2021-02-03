from flask import Flask,render_template,url_for,request,redirect,send_file
import cv2
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
import numpy as np
import os
import warnings as w
w.filterwarnings('ignore')
from IPython.display import Image, display
from keras.applications.vgg16 import VGG16
import time
from flask_caching import Cache

 
# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

app = Flask(__name__)


def predict_image(image_path):

        #loading saved trained mobilenet model 
        model_path = '/Users/priya/Downloads/CV/predict_house.h5'
        model = tf.keras.models.load_model(model_path)

        #pre-processing the image
        image = load_img(image_path, target_size=(224, 224))
        image = img_to_array(image)
        img_array = np.expand_dims(image, axis=0)
        image = preprocess_input(img_array)


        #predicting the output
        prediction = np.round(model.predict(image))
        p = prediction[:,0]
        
        return p


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        #make prediction
        p = predict_image(file_path)

        if p == 1:
            text_House = "It's a House!!!"
        else:
            text_House = "Oops!!! Not House !!!"

        return text_House

    return None




if __name__ == '__main__':
        app.run(debug=True)
