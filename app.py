import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.applications import imagenet_utils
import itertools
import os
from flask import Flask, request
from PIL import Image
from flask import Flask, render_template
app = Flask(__name__)

def preprocessed_image(file):
    mobile=tf.keras.applications.mobilenet.MobileNet()
    img_path ='static/'
    img = image.load_img((str(img_path) + str(file)), target_size=(224, 224)) 
    img_array = image.img_to_array(img)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    preprocessed_image=tf.keras.applications.mobilenet.preprocess_input(img_array_expanded_dims)
    predictions=mobile.predict(preprocessed_image)
    results=imagenet_utils.decode_predictions(predictions)
    numpy_array_Result = np.array(results)
    convert_percentage_dic = {}
    for i in numpy_array_Result:
        for j in i:
            string_value = j[2]
            cleaned_value = ''.join(filter(str.isdigit, string_value)) 
            int_value = int(cleaned_value)
            convert_percentage=int(int_value / 1000000)
            convert_percentage_dic[j[1]] = convert_percentage
    return convert_percentage_dic

@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("Image_prediction_index.html")


@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
    if request.method == 'POST':	
        img  = request.files['my_image']
        split_filename=(img.filename.split('.'))
        new_name = split_filename[0] +'.' + split_filename[1].upper()
        img_path = "static/" + new_name
        img.save(img_path)
        prediction = preprocessed_image(new_name)
    return render_template("Image_prediction_output.html", prediction = prediction, img_path = img_path)

if __name__ =='__main__':
	app.run(debug = True)