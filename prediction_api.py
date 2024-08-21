from flask import Flask, jsonify, request
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from utils import data_url_to_numpy_array
from PIL import Image


app = Flask(__name__)
CORS(app)

def load_model():
    print("Loading the mnist model...")
    global model
    # model = tf.keras.models.load_model("mnist_model")
    # model = tf.keras.models.load_model("mnist-model.h5")
    model = tf.keras.models.load_model("mnist_model_cnn")
    print("mnist model loaded successfully!")

load_model()

@app.route('/', methods=["POST"])
def hello_world():
    data = request.json
    image = np.array(data["pixels"], dtype=np.uint8)
    # print(data["pixelMatrix"])
    # print(data["dataURL"])
    # image = data_url_to_numpy_array(data["dataURL"], mode="binary")
    img = Image.fromarray(image)
    img.save("image.jpg")
    print("================== Model Prediction =====================")
    print(f"image_shape={image.shape}")
    print(model.predict(np.array([image.flatten()])).argmax(axis=1))
    probas = model.predict(np.array([image.flatten()])/255.)
    print(probas)
    max_proba = probas.max(axis=1)[0]
    prediction = probas.argmax(axis=1)[0]
    return jsonify({"prediction": str(prediction), "proba": str(max_proba)})


if __name__ == '__main__':
    app.run(debug=True)
