#!/usr/bin/python

# imports
import tensorflow as tf
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, flash, abort
from werkzeug import secure_filename
import base64
import io
from keras.applications.resnet50 import preprocess_input
from tensorflow.python.keras.models import load_model
from keras.preprocessing import image
from PIL import Image
from urllib.parse import quote
import uuid
import configparser


# reading labels for bilingual app and fungi descriptions
config = configparser.RawConfigParser()
config.read('label.properties')


# loading model
model = load_model('classifier.h5')
model._make_predict_function()
configTf = tf.ConfigProto()
configTf.gpu_options.allow_growth = True
sess = tf.Session(config=configTf)
app = Flask(__name__)
app.secret_key = 'bioinformatics'

# app constants
ALLOWED_EXTENSIONS = ['jpg', 'jpeg', 'png']
ALLOWED_LANGUAGES = ['en', 'lt']
IMAGE_RESIZE = 256
IMAGE_LIMIT = 300
LABELS = sorted(["Agaricus", "Amanita", "Boletus", "Cantharellus", "Cortinarius", "Hygrocybe", "Lactarius",
                 "Not fungi", "Russula", "Suillus"])


# DTO for result table information
class Prediction:
    def __init__(self, source, label, image_name, score, about):
        self.source = source
        self.label = label
        self.image_name = image_name
        self.score = score
        self.id = str(uuid.uuid4())
        self.about = about


def predict(file):
    buf = io.BytesIO(file)
    img = Image.open(buf)
    img = img.resize((IMAGE_RESIZE, IMAGE_RESIZE))
    x = image.img_to_array(img)
    x = x[..., :3]  # in case of 4-channel image (jpeg, png transperancy layer)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    gg = model.predict(x)
    classes = gg.argmax(axis=-1)[0]
    percent = round(gg[0, classes] * 100, 2)
    predicted_label = LABELS[classes]
    return predicted_label, percent


# prevent selecting other files
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


@app.route("/", methods=['GET'])
def redirec_to_en():
    return redirect('/en')  # redirect to default app language


@app.route("/<lang>", methods=['GET'])
def reload(lang):
    if lang not in ALLOWED_LANGUAGES:
        abort(404)
    return render_template('index.html', copyrights=config.get(lang, 'copyrights'), h1=config.get(lang, 'h1'),
                           h2=config.get(lang, 'h2'), choose_file=config.get(lang, 'choose_file'),
                           submit=config.get(lang, 'submit'), title=config.get(lang, 'title'),  render_results=False)


@app.route('/<lang>', methods=['POST'])
def submit_file_for_prediction(lang):
    if request.method == 'POST':
        predictions = []
        if len(request.files.getlist('images')) > 30:
            flash(config.get(lang, 'error_30'))
            return redirect(url_for('submit_file_for_prediction'), lang=lang)
        for file in request.files.getlist('images'):
            file_content = file.read()
            file_b64 = base64.b64encode(file_content).decode('ascii')  # show selected images without saving
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                label, percent = predict(file_content)
                data_url = 'data:image/png;base64,{}'.format(quote(file_b64))
                prediction = Prediction(data_url, config.get(label, 'name_%s' % lang), filename, percent,
                                        config.get(label, 'about_%s' % lang))
                predictions.append(prediction)
            else:
                flash(config.get(lang, 'error_file'))
                return redirect(url_for('submit_file_for_prediction', lang=lang))
        return render_template('index.html', predictions=predictions, copyrights=config.get(lang, 'copyrights'),
                               h1=config.get(lang, 'h1'), h2=config.get(lang, 'h2'),
                               choose_file=config.get(lang, 'choose_file'), submit=config.get(lang, 'submit'),
                               prediction=config.get(lang, 'prediction'), score=config.get(lang, 'score'),
                               results=config.get(lang, 'results'), image=config.get(lang, 'image'),
                               read_more=config.get(lang, 'read_more'), title=config.get(lang, 'title'), render_results=True)


if __name__ == "__main__":
    app.debug=False
    app.run(host='0.0.0.0')
