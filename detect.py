import tensorflow.keras
from PIL import Image, ImageOps
import numpy as np
import sqlite3
import cv2 as cv

#connection = sqlite3.connect('detected_items.db')
#cursor = connection.cursor()

#command1 = """"CREATE TABLE IF NOT EXISTS
#stores(store_id INTEGER PRIMARY KEY, location TEXT"""

#cursor.execute(command1)



# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = tensorflow.keras.models.load_model('keras_model.h5')

# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1.
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Replace this with the path to your image
#image = Image.open('test_photo.jpg')
video = cv.VideoCapture('slaughter_video_1.mp4')

while True:
        _, frame = video.read()
        #Convert the captured frame into RGB`
        image = Image.fromarray(frame, 'RGB')

        #resize the image to a 224x224 with the same strategy as in TM2:
        #resizing the image to be at least 224x224 and then cropping from the center
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.ANTIALIAS)

        #turn the image into a numpy array
        image_array = np.asarray(image)


        # display the resized image
        image.show()

        # Normalize the image
        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

        # Load the image into the array
        data[0] = normalized_image_array

        # run the inference
        prediction = model.predict(data)
        print(prediction)

        classes = np.argmax(prediction, axis = 1)
        if classes == 0:
            print("slaughtered")
        else:
            print("error")


        cv.imshow("Prediction", frame)
        key=cv.waitKey(1)
        if key == ord('q'):
                break
video.release()
cv.destroyAllWindows()

