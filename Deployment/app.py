from flask import Flask, request, jsonify
from gevent.pywsgi import WSGIServer
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import cv2
from skimage.filters import unsharp_mask
from PIL import Image

app = Flask(__name__)
def remove_noise (image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    X = cv2.bilateralFilter(gray, 15, sigmaColor=10, sigmaSpace=10)
    median = cv2.medianBlur(X, 5)
    result_2 = unsharp_mask(median, radius=10, amount=4)*255
    result_2 = np.uint8(result_2)
    sharpen = cv2.Canny(result_2, 100,250)
    return sharpen

def predict_rf (descriptors, rf):
    predictions = np.zeros(4)
    for des in descriptors:
        pred = rf.predict([des])
        ind = np.int64(pred[0])
        predictions[ind]+=1
    return np.argmax(predictions)

rf = pickle.load(open('rf.pk1' , 'rb'))

@app.route('/predict_number', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return "No image uploaded", 400

    image_file = request.files['image']

    # Open the image using PIL
    img = Image.open(image_file)

    # Convert the image to a numpy array
    img_array = np.array(img)

    removed_noise = remove_noise(img_array)
    sift = cv2.SIFT_create(500)
    kp , descriptors= sift.detectAndCompute(removed_noise,None)
    if descriptors is not None:
        predic = predict_rf(descriptors, rf)
    else :
        predic =0

    return str(predic)

if __name__ == '__main__':
    http_server = WSGIServer(('', 5000), app)
    http_server.serve_forever()