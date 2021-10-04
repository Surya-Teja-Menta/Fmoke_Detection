from __future__ import division, print_function
# coding=utf-8
import os
from PIL import Image
import numpy as np
import torch
from Prediction_Service.predict import Predict

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename


# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH ='saved_models/model_final.pth'

# Load your trained model
model = torch.load(MODEL_PATH,map_location=torch.device('cpu'))




def model_predict(img_path, model):
    print(img_path)
    img = Image.open(img_path)

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    ## Scaling
    x=x/255
    x = np.expand_dims(x, axis=0)
   

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
   # x = preprocess_input(x)

    preds = Predict(img)
    print(preds)
    res=f'{preds[0]} with {preds[1]}'
    return res


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

        # Make prediction
        preds = model_predict(file_path, model)
        result=preds
        return result
    return None


if __name__ == '__main__':
    app.run(port=5001,debug=True)