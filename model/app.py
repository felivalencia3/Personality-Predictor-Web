from flask import Flask, jsonify, request
from nlp import load_model
from predict import tellmemyMBTI
from util import split_lines_efficient
import random


app = Flask(__name__, static_folder="../build", static_url_path="/")
model = load_model()


@app.route("/")
def index():
    return app.send_static_file("index.html")


@app.route("/api/predict", methods=["POST"])
def predict():
    print()
    name = request.json["name"]
    text = request.json["text"]
    if text:
        result, image = tellmemyMBTI(split_lines_efficient(text), name)
        return jsonify({"prediction": result, "image": image})
    else:
        return jsonify({"error": "Text parameter is missing."}), 400
