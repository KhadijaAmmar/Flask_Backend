import flask
import preprocess
import numpy as np
import tensorflow as tf
import keras
from flask import request
from keras.models import load_model

# instantiate flask
app = flask.Flask(__name__)



# we need to redefine our metric function in order
# to use it when loading the model
def auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    keras.backend.get_session().run(tf.local_variables_initializer())
    return auc


# load the model, and pass in the custom metric function
global graph
graph = tf.get_default_graph()
model = load_model('model31.h5', custom_objects={'auc': auc})


@app.route("/api/getPath", methods=['POST','GET'])
def getPath():
    if request.method == 'POST':

        audio = request.get_json(silent=True)
        print(audio["path"])
        re=predict(audio["path"])
        return (re)




def predict(name):

    data = {"path": name}

    params = flask.request.json
    if (params == None):
        params = flask.request.args

    # if parameters are found, return a prediction
    if (params != None):
        with graph.as_default():
            sample = preprocess.wav2mfcc('./test/'+name+'/00f0204f_nohash_0.wav')
            sample_reshaped = sample.reshape(1, 40, 249, 1)
            data["prediction"] = preprocess.get_labels()[0][np.argmax(model.predict(sample_reshaped))]
            data["success"] = True

    # return a response in json format
    return flask.jsonify(data)


# start the flask app, allow remote connections
app.run(host='0.0.0.0')
