from flask import Flask, render_template
from flask import Flask, request, jsonify
import torch
from torchvision import transforms
import base64
import io
from PIL import Image
from model import carregar_modelo

app = Flask(__name__)

model = carregar_modelo()

@app.route("/")
def homepage():
    return render_template("caixaDesenho.html")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    image = data['image']
    image = base64.b64decode(image.split(',')[1])
    image = Image.open(io.BytesIO(image)).convert('L')
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        transforms.Resize(28)
    ])
    
    image = transform(image)
    image = image.view(image.shape[0], -1)
    
    with torch.no_grad():
        pred = model(image)
        pred = pred.argmax(1)

        return jsonify({
        'prediction': pred.item()
        })

if __name__ == "__main__":
    app.run(debug=True)
