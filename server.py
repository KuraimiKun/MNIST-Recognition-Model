from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras import preprocessing

app = Flask(__name__)

# Load the Keras model
model = load_model("digits_model")

# Define the function to recognize a digit
def recognize_digit(img):
    # Convert image to grayscale
    img = img.convert("L")
    # Invert each pixel color
    img = Image.eval(img, lambda x: 255 - x)
    # Resize image to 28x28 pixels
    img = img.resize((28, 28))
    # Convert image to numpy array
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    # Predict the digit
    prediction = model.predict(img_array)
    predicted_digit = np.argmax(prediction)
    return str(predicted_digit)

@app.route('/api/recognize', methods=['POST'])
def recognize():
    result = ""
    files = request.files.to_dict()
    for index in range(1, 12):
        file = files.get(f"{index}.png")
        if file:
            img = Image.open(file)
            result += recognize_digit(img)
    response = jsonify(result)
    response.headers.add('Access-Control-Allow-Origin', '*')  # Allow requests from any origin
    return response

@app.route('/')
def index():
    return open('index.html').read()

if __name__ == '__main__':
    app.run(port=8080)
