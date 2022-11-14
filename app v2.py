# import section
import librosa

from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from tensorflow import keras
from tensorflow.keras.models import load_model
import numpy as np
from keras.models import Sequential
import os
import json

from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

with open('templates/config.json', 'r') as c:
    parameters = json.load(c) ['parameters']



# Define flask app
app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = parameters['upload_location']
app.config['FilePath'] = parameters['modelPath']

# config =  tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, allow_soft_placement=True)
# session = tf.compat.v1.Session(config=config)

# le = LabelEncoder()
# le.classes_ = np.load('static/classes.npy')

# keras.backend.set_session(session)
# # load the trained model
model = load_model(app.config['FilePath'])
model.make_predict_function()

# create predict function
def extract_features_and_predict(file):
    # here do preprocess and predict the result
    audioFile = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file))

    x, sr = librosa.load(audioFile, sr=22050)
    print(x.shape)
    result = np.array([])

    # chroma_stft = np.mean(librosa.feature.chroma_stft(y=x, sr=sr).T, axis=0)
    # result = np.hstack(result, chroma_stft)
    # mel = np.mean(librosa.feature.melspectrogram(y=x, sr=sr).T, axis=0)
    # result = np.hstack(result, mel)
    # mfcc = np.mean(librosa.feature.mfcc(y=x, sr=sr).T, axis=0)
    # result = np.hstack(result, mfcc)

    # result = StandardScaler.fit_transform(result)


    # Convert string to array of floats
    # vect = to_append.split()

    # for i in range(len(vect)):
    #     vect[i] = float(vect[i])

    res = model.predict(result)
    print(res)

@app.route('/')
def main():
    print("--------------------- You're on main page -------------------")
    # main page of WebApp
    return render_template('main.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if (request.method == 'POST'):
        f = request.files['audioInput']
        filename = f.filename
        f.save( os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(filename)))
        print("----------------- File uploaded and saved completed file path of audio is: "+ filename + " -------------------")
        extract_features_and_predict(filename)
        return render_template('result.html', data = f)

if __name__ == '__main__':
    app.run()
