#!/usr/bin/python

# imports
import tensorflow as tf
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug import secure_filename
import time
import base64
import io
from keras.applications.resnet50 import preprocess_input
from tensorflow.python.keras.models import load_model
from keras.preprocessing import image
from PIL import Image
from urllib.parse import quote
import uuid
import configparser
config = configparser.RawConfigParser()
config.read('label.properties')
# loading model
model = load_model('classifier1.h5')
model._make_predict_function()
configTf = tf.ConfigProto()
configTf.gpu_options.allow_growth = True
sess = tf.Session(config=configTf)
# app constants
ALLOWED_EXTENSIONS = set(['jpg', 'jpeg', 'png'])
IMAGE_RESIZE = 256
IMAGE_LIMIT = 300


class Prediction:
    def __init__(self, source, label, image_name, score, label_lt, about):
        self.source = source
        self.label = label
        self.label_lt = label_lt
        self.image_name = image_name
        self.score = score
        self.id = str(uuid.uuid4())
        self.about = about


def predict(file):
    buf = io.BytesIO(file)
    img = Image.open(buf)
    img = img.resize((IMAGE_RESIZE, IMAGE_RESIZE))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    gg = model.predict(x)
    print(gg.argmax(axis=-1))
    classes = gg.argmax(axis=-1)[0]
    procent = round(gg[0, classes] * 100, 2)
    labels = ["Agaricus","Amanita","Boletus","Cantharellus","Cortinarius", "Hygrocybe", "Lactarius", "Not fungi", "Russula", "Suillus"]
    predicted_label = sorted(labels)[classes]
    return predicted_label, procent


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


app = Flask(__name__)
app.secret_key = 'bioinformatics'


@app.route("/", methods=['GET'])
def reload():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def submit_file_for_prediction():
    if request.method == 'POST':
        import time
        start_time = time.time()
        predictions = []
        print(len(request.files.getlist('images')))
        if len(request.files.getlist('images')) > 30:
            error = 'Only 30 images are allowed for prediction.'
            flash(error)
            return redirect(url_for('submit_file_for_prediction'))
        for file in request.files.getlist('images'):
            file_content = file.read()
            file_b64 = base64.b64encode(file_content).decode('ascii')
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                label, procent = predict(file_content)
                print("--- %s seconds ---" % str(time.time() - start_time))
                data_url = 'data:image/png;base64,{}'.format(quote(file_b64))
                prediction = Prediction(data_url, label, filename, procent, config.get(label, 'name_lt'), config.get(label, 'about'))
                predictions.append(prediction)
            else:
                error = 'Please select file with allowed file format: jpg/jpeg/png.'
                flash(error)
                return redirect(url_for('submit_file_for_prediction'))
        return render_template('template.html', predictions=predictions)

if __name__ == "__main__":
    app.debug=False
    app.run(host='0.0.0.0')
