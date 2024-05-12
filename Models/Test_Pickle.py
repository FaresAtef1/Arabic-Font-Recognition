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

num_train_samples=10
num_test_samples=int(0.2*num_train_samples)

all_des=np.empty((1,128))

sift = cv2.SIFT_create(500)

num_of_desc=[]

for i in range (num_train_samples):
    image = cv2.imread("../../fonts-dataset/IBM Plex Sans Arabic/"+str(i)+".jpeg")
    removed_noise = remove_noise(image)
    kp , descriptors= sift.detectAndCompute(removed_noise,None)
    if descriptors is not None:
        all_des=np.vstack((all_des,descriptors))
    print("processing type 1 image"+str(i))

num_of_desc+=[all_des.shape[0]]
print("FINISHED READING FIRST SET OF IMAGES")

for i in range (num_train_samples):
    image = cv2.imread("../../fonts-dataset/Lemonada/"+str(i)+".jpeg")
    removed_noise = remove_noise(image)
    kp , descriptors= sift.detectAndCompute(removed_noise,None)
    if descriptors is not None:
        all_des=np.vstack((all_des,descriptors))
    print("processing type 2 image"+str(i))

num_of_desc+=[all_des.shape[0]]
print("FINISHED READING SECOND SET OF IMAGES")

for i in range (num_train_samples):
    image = cv2.imread("../../fonts-dataset/Marhey/"+str(i)+".jpeg")
    removed_noise = remove_noise(image)
    kp , descriptors= sift.detectAndCompute(removed_noise,None)
    if descriptors is not None:
        all_des=np.vstack((all_des,descriptors))
    print("processing type 3 image"+str(i))

num_of_desc+=[all_des.shape[0]]
print("FINISHED READING THIRD SET OF IMAGES")


for i in range (num_train_samples):
    image = cv2.imread("../../fonts-dataset/Scheherazade New/"+str(i)+".jpeg")
    removed_noise = remove_noise(image)
    kp , descriptors= sift.detectAndCompute(removed_noise,None)
    if descriptors is not None:
        all_des=np.vstack((all_des,descriptors))
    print("processing type 4 image"+str(i))

num_of_desc+=[all_des.shape[0]]
print("FINISHED READING FOURTH SET OF IMAGES")

desc_labels=np.zeros(all_des.shape[0])
desc_labels[num_of_desc[0]:num_of_desc[1]]=1
desc_labels[num_of_desc[1]:num_of_desc[2]]=2
desc_labels[num_of_desc[2]:num_of_desc[3]]=3

X_train, X_test, y_train, y_test = train_test_split(all_des, desc_labels, test_size=0.2, random_state=42)
rf = RandomForestClassifier()
rf.fit(X_train, y_train)

def predict_rf (descriptors, rf):
    predictions = np.zeros(4)
    for des in descriptors:
        pred = rf.predict([des])
        ind = np.int64(pred[0])
        predictions[ind]+=1
    return np.argmax(predictions)

num_right0=0
num_right1=0
num_right2=0
num_right3=0

for i in range (num_test_samples):
    image = cv2.imread("../../fonts-dataset/IBM Plex Sans Arabic/"+str(num_train_samples+i)+".jpeg")
    removed_noise = remove_noise(image)
    kp , descriptors= sift.detectAndCompute(removed_noise,None)
    if descriptors is not None:
        predic = predict_rf(descriptors, rf)
        num_right0 += predic==0
        print("processing type 0 image"+str(num_train_samples+i)+" is "+str(predic))

print("FINISHED TESTING FIRST SET OF IMAGES")

for i in range (num_test_samples):
    image = cv2.imread("../../fonts-dataset/Lemonada/"+str(num_train_samples+i)+".jpeg")
    removed_noise = remove_noise(image)
    kp , descriptors= sift.detectAndCompute(removed_noise,None)
    if descriptors is not None:
        predic = predict_rf(descriptors, rf)
        num_right1 += predic==1
        print("processing type 1 image"+str(num_train_samples+i)+" is "+str(predic))


print("FINISHED TESTING SECOND SET OF IMAGES")

for i in range (num_test_samples):
    image = cv2.imread("../../fonts-dataset/Marhey/"+str(num_train_samples+i)+".jpeg")
    removed_noise = remove_noise(image)
    kp , descriptors= sift.detectAndCompute(removed_noise,None)
    if descriptors is not None:
        predic = predict_rf(descriptors, rf)
        num_right2 += predic==2
        print("processing type 2 image"+str(num_train_samples+i)+" is "+str(predic))

print("FINISHED TESTING THIRD SET OF IMAGES")

for i in range (num_test_samples):
    image = cv2.imread("../../fonts-dataset/Scheherazade New/"+str(num_train_samples+i)+".jpeg")
    removed_noise = remove_noise(image)
    kp , descriptors= sift.detectAndCompute(removed_noise,None)
    if descriptors is not None:
        predic = predict_rf(descriptors, rf)
        num_right3 += predic==3
        print("processing type 3 image"+str(num_train_samples+i)+" is "+str(predic))

print("FINISHED TESTING FOURTH SET OF IMAGES")
print(((num_right0+num_right1+num_right2+num_right3)/(4*num_test_samples))*100)

pickle.dump(rf , open('rf-model.pk1' , 'wb'))