from flask import Flask, render_template, request, jsonify
from PIL import Image
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch
from torchvision.models import resnet50
from urllib.request import urlretrieve
import json
import os
import pickle

app = Flask(__name__)

# Define the model architecture
model = resnet50(num_classes=100)

# Load the saved model
model_path = './model/resnet50_cifar100.pth'
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

# Set the model to evaluation mode
model.eval()

# Define transformation for input image
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Define endpoint for API
@app.route('/', methods=['GET', 'POST'])
def homepage():
    # If the request method is GET, render the web UI page
    if request.method == 'GET':
        return render_template('webui.html')
    # If the request method is POST, process the uploaded image and return the predicted class
    elif request.method == 'POST':
        # Load and preprocess input image
        image = Image.open(request.files['image']).convert('RGB')
        image_tensor = transform(image).unsqueeze(0)
        # Make prediction
        with torch.no_grad():
            output = model(image_tensor)
            prob = F.softmax(output, dim=1)
            prediction = torch.argmax(prob, dim=1)
        # Get the predicted class name and its probability
        class_name = get_class_name(prediction.item())
        probability = prob[0][prediction].item()
        # Return predicted class and probability as JSON
        return jsonify({'class': class_name, 'probability': probability})

def get_class_labels():
    # Check if class labels file exists
    if os.path.exists('./data/class_labels.txt'):
        # Load class labels from file
        with open('./data/class_labels.txt', 'r') as f:
            class_labels = f.read().splitlines()
    else:
        # Download class labels
        # urlretrieve('https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz', 'cifar-100-python.tar.gz')
        with open('./data/cifar-100-python/meta', 'rb') as f:
            meta = pickle.load(f, encoding='latin1')
        class_labels = meta['fine_label_names']
        # Save class labels to file
        with open('./data/class_labels.txt', 'w') as f:
            for label in class_labels:
                f.write(label + '')
    return class_labels

def get_class_name(class_id):
    class_labels = get_class_labels()
    return class_labels[class_id]

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5010, debug=True)
