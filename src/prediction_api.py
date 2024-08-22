from flask import Flask, jsonify, request
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import cv2
from utils import center_digit


app = Flask(__name__)
CORS(app)

def load_model():
    print("Loading the mnist model...")
    global model
    model = tf.keras.models.load_model("model/mnist_model_cnn")
    print("mnist model loaded successfully!")

load_model()

@app.route('/', methods=["POST"])
def hello_world():
    data = request.json
    image = np.array(data["pixels"], dtype=np.uint8)
    cv2.imwrite("images/image.jpg", image)
    centered_image = center_digit(image)
    cv2.imwrite("images/centered_image.jpg", centered_image)
    centered_image /= 255.
    probas = model.predict(np.array([centered_image]))
    max_proba = probas.max(axis=1)[0]
    prediction = probas.argmax(axis=1)[0]
    return jsonify({"prediction": str(prediction), "proba": str(max_proba)})


if __name__ == '__main__':
    app.run(debug=True)
