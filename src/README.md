# ResNet50 Image Classifier
This is a Flask web app that uses a pre-trained ResNet50 model to classify uploaded images from the ImageNet dataset.

## Requirements
This app requires the following Python packages:

- Flask
- Pillow
- torchvision
- torch

## Usage
1. Clone this repository to your local machine:

`git clone https://github.com/yourusername/resnet50-image-classifier.git`

2.Change into the project directory:
`cd resnet50-image-classifier`

3. Download the pre-trained ResNet50 model:

`wget https://download.pytorch.org/models/resnet50-19c8e357.pth -P model/`

4. Start the Flask app:

`python app.py`

5. Open a web browser and navigate to http://localhost:5010.

6. Upload an image, and the app will classify it as one of the 1000 object categories in the ImageNet dataset.

## Structure
The project has the following structure:

```
resnet50-image-classifier/
  ├── app.py
  ├── model/
  │   └── resnet50-19c8e357.pth
  ├── requirements.txt
  ├── static/
  │   └── style.css
  └── templates/
      └── webui.html
```
app.py: This is the Flask app that serves the web UI and processes the uploaded images using the pre-trained ResNet50 model.
model/resnet50-19c8e357.pth: This is the pre-trained ResNet50 model that classifies images into one of the 1000 object categories in the ImageNet dataset.
requirements.txt: This file lists the required Python packages.
static/style.css: This is the CSS file that styles the web UI.
templates/webui.html: This is the HTML file that defines the web UI.

## Acknowledgements
The ResNet50 model used in this project was trained on the [ImageNet](http://www.image-net.org/) dataset and is available from the [PyTorch website](https://pytorch.org/docs/stable/torchvision/models.html#id3). The code for loading and preprocessing the input images was adapted from the [PyTorch Image Classifier](https://github.com/udacity/pytorch_challenge/blob/master/cat_to_name.json) project in the Udacity PyTorch Scholarship Challenge. The web UI was adapted from a code snippet on [MDN Web Docs](https://developer.mozilla.org/en-US/docs/Web/API/FormData/Using_FormData_Objects).