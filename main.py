from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
from PIL import Image
import os
import cv2
import numpy as np
from processing import imageProcesser, coordinateProcessor, extractSegment
from constants import classes, colourMap
import base64
from io import BytesIO

# deeplab imports
import torch
import torch.nn as nn
import models.deeplab.network as network
import models.deeplab.utils as utils
import os
from models.deeplab.datasets import Cityscapes  #, cityscapes
from torchvision import transforms as T
from glob import glob

# macro ws imports
from scripts.macro_ws import macro_ws

# ANN imports
import tensorflow as tf
import joblib

app = Flask(  # Create a flask app
  __name__,
  template_folder='templates',  # Name of html file folder
  static_folder='static'  # Name of directory for static files
)

# deeplab setup
## deeplab constants
ckptPath = "models/deeplab/checkpoints/best_deeplabv3plus_mobilenet_cityscapes_os16.pth"
num_classes = 19

## deeplab code
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device: %s" % device)
model = network.modeling.__dict__["deeplabv3plus_mobilenet"](
  num_classes=num_classes, output_stride=16)
utils.set_bn_momentum(model.backbone, momentum=0.01)
checkpoint = torch.load(ckptPath, map_location=torch.device('cpu'))
model.load_state_dict(checkpoint["model_state"])
model = nn.DataParallel(model)
model.to(device)
print("Resume model from %s" % ckptPath)
del checkpoint
transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
  ])
decode_fn = Cityscapes.decode_target

# ANN Setup
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

@app.route("/", methods=["GET"])
def health_check():
  return "Health Check"

@app.route("/upload", methods=["GET", "POST"])
def load_file():
  if request.method == "POST":
    image = request.files['file']
    
    coordinates = request.form['coordinates']
    lat, lng = coordinateProcessor(coordinates)
    #filename = secure_filename(f.filename)

    #todo: save into uploads folder
    #f.save(os.path.join(app.config['.//uploads'], filename))

    # image segmentation
    with torch.no_grad():
      global model
      model = model.eval()
      img = Image.open(image).convert('RGB')
      img = transform(img).unsqueeze(0)  # To tensor of NCHW
      img = img.to(device)

      # Obtain segmented image
      pred = model(img).max(1)[1].cpu().numpy()[0]  # HW
      colorized_preds = decode_fn(pred).astype('uint8')
      colorized_preds = Image.fromarray(colorized_preds)
      img = cv2.cvtColor(np.array(colorized_preds), cv2.COLOR_RGB2BGR)

      # Obtain segmented image string
      buffered = BytesIO()
      colorized_preds.save(buffered, format="JPEG")
      buffered.seek(0)
      img_bytes = base64.b64encode(buffered.getvalue())

      # Obtain segment percentages of image
      percentages = {}
      for feature in colourMap:
        segmentPercentage = extractSegment(img, colourMap[feature])
        percentages[feature] = segmentPercentage

      # Obtain macro walkscore
      macro = macro_ws(lat, lng)

    # Load ANN model
    ann = tf.keras.models.load_model("models/ANN")
    X = list(percentages.values())
    X.extend([macro])

    # Load scaler that we used in our training
    scaler = joblib.load("models/scaler/ann_scaler.joblib")
    X = np.array(X).reshape(1, -1)
    X_input = scaler.transform(X)

    # Retrieve walkscore and bikescores
    result = ann.predict(X_input)

    # API Response
    final_dict = {
      "walkscore": float(result[0][0]),
      "bikescore": float(result[0][1]),
      "image": img_bytes.hex()
    }
    return jsonify(final_dict)
    #simple proof of concept that can do PIL on image
    # for file in files:
    #   img = Image.open(file)

    # imgGray = imageProcesser(img)
    # imgGray.save(f"{uploadPath}/{f.filename}")
    # return send_file(f.filename)
  else:
    return "Request Method not supported"

try:
  PORT = os.environ["PORT"]
except:
  PORT = 8080

if __name__ == "__main__":
  from waitress import serve
  serve(app, host="0.0.0.0", port=PORT)
