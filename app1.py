import os
#import tensorflow as tf
import numpy as np
#from keras.preprocessing import image
from PIL import Image
import cv2
#from keras.models import load_model
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import os
import torch

app = Flask(__name__)

model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/runs/train/exp7/weights/best.pt')
print('Model loaded.') 

def getResult(img):
    image = cv2.imread(img)
    #image = Image.fromarray(image)
    #image = image.resize((64, 64))
    #image=np.array(image)
    #input_img = np.expand_dims(image, axis=0)
    result=model.predict(image)
    return result

# home page
@app.route("/")
def index():
    return render_template(
        "index.html", 
        ori_image="static/person.jpg", 
        det_image="static/person_det.jpg",
        fName=None
        )

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']

        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        value=getResult(file_path)
        result=get_className(value) 
        return result
    return None

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)