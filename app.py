from flask import Flask, render_template, Response, jsonify
import cv2
from flask.helpers import url_for
from tensorflow import keras as tensorflowkeras
from PIL import Image, ImageOps
import numpy as np
import sqlite3
import random
import qrcode
from werkzeug.utils import redirect
from datetime import datetime



app = Flask(__name__)
camera = cv2.VideoCapture(0)

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = tensorflowkeras.models.load_model('keras_model.h5')

# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1.
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)


def generate_frames():
    while True:
        #read the camera frame
        success, frame = camera.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            image = Image.fromarray(frame, 'RGB')

            #resize the image to a 224x224 with the same strategy as in TM2:
            #resizing the image to be at least 224x224 and then cropping from the center
            size = (224, 224)
            image = ImageOps.fit(image, size, Image.LANCZOS)

            #turn the image into a numpy array
            image_array = np.asarray(image)

            # display the resized image
            #image.show()
            #os.remove('../website_s_sys/static/qrcode001.png')
            # Normalize the image
            normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

            # Load the image into the array
            data[0] = normalized_image_array

            # run the inference
            prediction = model.predict(data)
            

            classes = np.argmax(prediction, axis=1)
            if classes == 0:
                print(prediction)
                status="Detected"
                print("slaughtered")
                now = datetime.now()
                input_data = "The animal is successfully slaughtered with confidence level of {}.".format(prediction)
                #Creating an instance of qrcode
                qr = qrcode.QRCode(
                        version=1,
                        box_size=10,
                        border=5)
                qr.add_data(input_data)
                qr.make(fit=True)
                img = qr.make_image(fill='black', back_color='white')
                img.save('../website_s_sys/static/qrcode001.png')
            

               
            else:
                status="error"
            
            frame = buffer.tobytes()
            
        
                
        #ulang2
        yield(b'--frame\r\n'
              b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
