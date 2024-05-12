import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans
import random
from PIL import Image
from skimage.filters import unsharp_mask
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

num_of_centroids=4000
num_samples=600

def remove_noise (image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    X = cv2.bilateralFilter(gray, 15, sigmaColor=10, sigmaSpace=10)
    median = cv2.medianBlur(X, 5)
    result_2 = unsharp_mask(median, radius=10, amount=4)*255
    result_2 = np.uint8(result_2)
    sharpen = cv2.Canny(result_2, 100,250)
    return sharpen

# create frequency histogram for each image in the training set and the test set
def create_histogram(descriptor_list, kmeans):
    histogram = np.zeros(kmeans.n_clusters)
    preds = kmeans.predict(descriptor_list)
    for pre in preds:
        histogram[pre] += 1
    return histogram

all_des=[]
labels = []
sift = cv2.SIFT_create(200)

for i in range (num_samples):
    image = cv2.imread("../../fonts-dataset/IBM Plex Sans Arabic/"+str(i)+".jpeg")
    removed_noise = remove_noise(image)
    kp , descriptors= sift.detectAndCompute(removed_noise,None)
    if descriptors is not None:
        all_des.append(descriptors)
        labels.append(0)
    print("processing type 0 image"+str(i))

    image = cv2.imread("../../fonts-dataset/Lemonada/"+str(i)+".jpeg")
    removed_noise = remove_noise(image)
    kp , descriptors= sift.detectAndCompute(removed_noise,None)
    if descriptors is not None:
        all_des.append(descriptors)
        labels.append(1)
    print("processing type 1 image"+str(i))

    image = cv2.imread("../../fonts-dataset/Marhey/"+str(i)+".jpeg")
    removed_noise = remove_noise(image)
    kp , descriptors= sift.detectAndCompute(removed_noise,None)
    if descriptors is not None:
        all_des.append(descriptors)
        labels.append(2)
    print("processing type 2 image"+str(i))

    image = cv2.imread("../../fonts-dataset/Scheherazade New/"+str(i)+".jpeg")
    removed_noise = remove_noise(image)
    kp , descriptors= sift.detectAndCompute(removed_noise,None)
    if descriptors is not None:
        all_des.append(descriptors)
        labels.append(3)
    print("processing type 3 image"+str(i))

X_train, X_test, y_train, y_test = train_test_split(all_des, labels, test_size=0.2, random_state=42)

# concatenate all descriptors in the training set together
descriptors = np.concatenate(X_train, axis=0)
kmeans = KMeans(n_clusters=num_of_centroids, random_state=42)
kmeans.fit(descriptors)

pickle.dump(kmeans , open('kmeans.pk1' , 'wb'))

X_train_hist = []
for des in X_train:
    hist = create_histogram(des, kmeans)
    X_train_hist.append(hist)

X_test_hist = []
for des in X_test:
    hist = create_histogram(des, kmeans)
    X_test_hist.append(hist)

## create a SVM classifier
from sklearn.svm import SVC

svm = SVC(kernel='poly', C=0.1, random_state=0, coef0=1, degree=4, gamma=10.0,class_weight= None)
svm.fit(X_train_hist, y_train)

pickle.dump(svm , open('svm.pk1' , 'wb'))

y_pred = svm.predict(X_test_hist)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))