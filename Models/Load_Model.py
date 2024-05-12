import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import cv2
from skimage.filters import unsharp_mask

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

num_train_samples=10
num_test_samples=int(0.2*num_train_samples)
sift = cv2.SIFT_create(500)



rf = pickle.load(open('rf-model.pk1' , 'rb'))

for i in range (num_test_samples):
    image = cv2.imread("../../fonts-dataset/Scheherazade New/"+str(num_train_samples+i)+".jpeg")
    print(image.shape)
    removed_noise = remove_noise(image)
    kp , descriptors= sift.detectAndCompute(removed_noise,None)
    if descriptors is not None:
        predic = predict_rf(descriptors, rf)
        # num_right3 += predic==3
        print("processing type 3 image"+str(num_train_samples+i)+" is "+str(predic))