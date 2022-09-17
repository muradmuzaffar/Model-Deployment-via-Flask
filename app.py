import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''

    features = [request.form.get("1"), request.form.get(
        "2"), request.form.get("3"), request.form.get("4")]

    prediction = model.predict([np.array(features)])

    return render_template('index.html', prediction_text=prediction[0])
