from flask import Flask, request, render_template
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from tensorflow.keras.models import load_model
import tensorflow_hub as hub

import pickle

import tensorflow as tf

import warnings
warnings.filterwarnings("ignore")

import cv2
from numpy import *

app = Flask(__name__)


# Load the models with the custom_objects parameter
custom_objects = {
    'KerasLayer': hub.KerasLayer
}

model = load_model('disease.keras', custom_objects=custom_objects)


# Load the data from the pickle file
with open('crop_xg.pkl', 'rb') as file:
    xg_model = pickle.load(file)

# Load the data from the pickle file
with open('fertilizer_random.pkl', 'rb') as file:
    rf_model = pickle.load(file)


@app.route('/')
def index():
    return render_template('welcome.html')


@app.route('/identify_crop')
def action1():
    return render_template('crop.html')

@app.route('/identify_fertilizer')
def action2():
    return render_template('fertilizer.html')

@app.route('/identify_disease')
def action3():
    return render_template('disease.html')





@app.route('/crop_type', methods=['GET', 'POST'])
def crop_pred():


    if request.method == 'POST':
        # Get the form data
        N = float(request.form['N'])
        P = float(request.form['P'])
        K = float(request.form['K'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])

        label_mapping ={
            0: 'apple',
            1: 'banana',
            2: 'blackgram',
            3: 'chickpea',
            4: 'coconut',
            5: 'coffee',
            6: 'cotton',
            7: 'grapes',
            8: 'jute',
            9: 'kidneybeans',
            10: 'lentil',
            11: 'maize',
            12: 'mango',
            13: 'mothbeans',
            14: 'mungbean',
            15: 'muskmelon',
            16: 'orange',
            17: 'papaya',
            18: 'pigeonpeas',
            19: 'pomegranate',
            20: 'rice',
            21: 'watermelon'
        }

        input_array = np.array([N,P,K,temperature,humidity,ph,rainfall])
        input_array = input_array.reshape(1,7)

        # Get probabilities for each class
        probabilities = xg_model.predict_proba(input_array)

        # Get indices of top three predictions
        top_three_indices = np.argsort(probabilities[0])[::-1][:3]

        # Get top three crops and their probabilities
        top_three_crops = [(label_mapping[i], probabilities[0][i]) for i in top_three_indices]

        # Construct output string
        output = "Top three suitable crops for your land: "
        for crop, prob in top_three_crops:
            output = output+crop+","
        output = output[0:len(output)-1]

        return render_template('output.html', n=output)





@app.route('/fertilizer', methods=['POST'])
def fertilizer():
    
    temperature = float(request.form['temperature'])
    humidity = float(request.form['humidity'])
    moisture = float(request.form['moisture'])
    soil_type = request.form['soil_type']
    crop_type = request.form['crop_type']
    nitrogen = float(request.form['nitrogen'])
    potassium = float(request.form['potassium'])
    phosphorous = float(request.form['phosphorous'])


    soil_type_encoding = {
    'Black': 0,
    'Clayey': 1,
    'Loamy': 2,
    'Red': 3,
    'Sandy': 4
    }

    crop_type_encoding = {
    'Barley': 0,
    'Cotton': 1,
    'Ground Nuts': 2,
    'Maize': 3,
    'Millets': 4,
    'Oil seeds': 5,
    'Paddy': 6,
    'Pulses': 7,
    'Sugarcane': 8,
    'Tobacco': 9,
    'Wheat': 10
    }


    soil_type = soil_type_encoding[soil_type.title()]
    crop_type = crop_type_encoding[crop_type.title()]

    

    input_array = np.array([temperature,humidity,moisture,soil_type,crop_type,nitrogen,potassium,phosphorous])


    fertilizer_name_decoding = {
        0: '10-26-26',
        1: '14-35-14',
        2: '17-17-17',
        3: '20-20',
        4: '28-28',
        5: 'DAP',
        6: 'Urea'
    }

    
    input_array = input_array.reshape(1,8)

    value = rf_model.predict(input_array)

    ele = fertilizer_name_decoding[value[0]]

    output = "The Crop/Land Need " + ele + " Fertilizer"

    print(output)

    return render_template('output.html',n= output)




file_path_g = ""

@app.route('/detect', methods = ['GET', 'POST'])
def upload_detection1():
    
    if(request.method == 'POST'):
        f=request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)


        global file_path_g

        file_path_g=file_path


        
    return render_template("process.html")


def pred(image_name):

    print(image_name)
    
    #load the image
    img=cv2.imread(image_name)

    #resize the image
    resize=cv2.resize(img,(256,256))
   

    #optimize the new image
    resize=resize/255

    
    #expand your image array
    img=expand_dims(resize,0)


    predictions = model.predict(img)

    # Convert the predicted probabilities to class labels
    predicted_labels = np.argmax(predictions, axis=1)

    classes = ['Angular Leaf Spot',
                'Bacterial spot',
                'Early blight',
                'Fusarium Wilt',
                'Healthy',
                'Late blight',
                'Leaf Mold',
                'Mosaic virus',
                'Phytophthora Blight',
                'Septoria leaf spot',
                'Spotted spider mite',
                'Target Spot',
                'YellowLeaf Curl Virus']

    suggestion = ['Path pro fungicide',
                   'Sulphur sprays or copper-based fungicides',
                   'GardenTech brand''s Daconil fungicides',
                   'Nirate Fertilizer',
                   'None',
                   'Dithane (mancozeb) MZ',
                   'Chlorothalonil',
                   'Isabion',
                   'Fosetyl-al (Aliette)',
                   'Chlorothalonil and mancozeb',
                   'Supreme IT',
                   'Potassium schoenite',
                   'Carbofuran']


    value = "The Detected Status Of the plant is: "+str(classes[predicted_labels[0]]) 
    value1 = "To cure the disease use "+str(suggestion[predicted_labels[0]])


    L = [value,value1]
    return L
   



@app.route('/predict')
def predict():

    output = pred(file_path_g)

    return render_template('output.html',n=output[0],n2=output[1])

if __name__ == '__main__':
    app.run(debug=False)
