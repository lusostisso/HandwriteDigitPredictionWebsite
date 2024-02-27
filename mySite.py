from flask import Flask, render_template
from flask import Flask, request, jsonify
import numpy as np
import torch
from torchvision import transforms
import base64
import matplotlib.pyplot as plt
import io
from PIL import Image
from model import carregar_modelo
import cv2
import helper


app = Flask(__name__)

model, transform = carregar_modelo()

@app.route("/")
def homepage():
    return render_template("caixaDesenho.html")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    image = data['image']

    image = base64.b64decode(image.split(',')[1])
    image = Image.open(io.BytesIO(image)).convert('L')
    
    image = transform(image)
    image = image.unsqueeze(0)
    helper.imshow(image[0].numpy().squeeze(), cmap='gray')

    with torch.no_grad():
        pred = model(image)
        pred = pred.argmax(1)

        return jsonify({
            'prediction': pred.item()
        })

if __name__ == "__main__":
    app.run(debug=True)
