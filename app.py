import flask
import preprocess
import numpy as np
import tensorflow as tf
import keras
from flask import request
from keras.models import load_model

import shutil
import os
import random

# instantiate flask
app = flask.Flask(__name__)
app.url_map.strict_slashes = False


# we need to redefine our metric function in order
# to use it when loading the model
def auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    keras.backend.get_session().run(tf.local_variables_initializer())
    return auc


# load the model, and pass in the custom metric function
global graph
graph = tf.get_default_graph()
model = load_model('test.h5', custom_objects={'auc': auc})


@app.route("/api/pathTestDada", methods=['POST'])
def pathTestDada():
    if request.method == 'POST':

        audio = request.get_json(silent=True)
        print(audio["path"])

        re = predictTest(audio["path"])
        return (re)


@app.route("/api/getPath", methods=['POST'])
def getPath():
    if request.method == 'POST':

        audio = request.get_json(silent=True)
        print(audio["path"])

        re = predict(audio["path"])
        global chemin
        chemin = audio["path"]
        print(chemin)
        return (re)



@app.route("/api/save_audio", methods=['POST'])
def saveAudio():
    if request.method == 'POST':
        print(chemin)
        dest = request.get_json(silent=True)
        destination = dest["path"]
        source = "C://Users//Stage//Downloads//"+chemin+".wav"
        dest1 = "C://Users//Stage//final project//test//"+destination

        shutil.move(source, dest1)
        return (dest1)




def predict(name):

    data = {"path": name}

    params = flask.request.json
    if (params == None):
        params = flask.request.args

    # if parameters are found, return a prediction
    if (params != None):
        with graph.as_default():
            sample = preprocess.wav2mfcc('C://Users//Stage//Downloads//'+name+'.wav')
            print(name)
            sample_reshaped = sample.reshape(1, 40, 47, 1)
            data["prediction"] = preprocess.get_labels()[0][np.argmax(model.predict(sample_reshaped))]
            data["success"] = True

    # return a response in json format
    return flask.jsonify(data)





def predictTest(name):

    data = {"path": name}

    params = flask.request.json
    if (params == None):
        params = flask.request.args

    # if parameters are found, return a prediction
    if (params != None):
        with graph.as_default():
            dir= "C://Users//Stage//final project//test//"+name
            filename = random.choice(os.listdir(dir))
            print (filename)
            sample = preprocess.wav2mfcc(dir+"//"+filename)
            print(name)
            sample_reshaped = sample.reshape(1, 40, 47, 1)
            data["prediction"] = preprocess.get_labels()[0][np.argmax(model.predict(sample_reshaped))]
            data["success"] = True

    # return a response in json format
    return flask.jsonify(data)



# start the flask app, allow remote connections
app.run(host='0.0.0.0')
