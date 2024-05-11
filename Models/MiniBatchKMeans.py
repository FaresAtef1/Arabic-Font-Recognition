import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans
import random
from PIL import Image
from skimage.filters import unsharp_mask

num_of_centroids=4096
num_train_samples=500
num_test_samples=125
K = 11
all_image_idx=random.sample(range(0,1000),num_train_samples+num_test_samples)

all_image_idx=random.sample(range(0,999),num_train_samples+num_test_samples)

def remove_noise (image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    X = cv2.bilateralFilter(gray, 15, sigmaColor=10, sigmaSpace=10)
    median = cv2.medianBlur(X, 5)
    result_2 = unsharp_mask(median, radius=10, amount=4)*255
    result_2 = np.uint8(result_2)
    sharpen = cv2.Canny(result_2, 100,250)
    return sharpen

def predict(path,centroids,centroids_labels):
    image=cv2.imread(path)
    removed_noise = remove_noise(image)
    _ , descriptors= sift.detectAndCompute(removed_noise,None)

    predections=[0.0,0.0,0.0,0.0]
    if descriptors is not None:
        for des in descriptors:
            idx=kmeans.predict([des])
            dist=np.linalg.norm(des-centroids[idx])
            if dist == 0:
                dist = 0.0000001
            predections+=(centroids_labels[idx]/dist)
        return np.argmax(predections)
    else:
        return -1

all_des=np.empty((1,128))

sift = cv2.SIFT_create()

num_of_desc=[]

for i in range (num_train_samples):
    image = cv2.imread("../../fonts-dataset/IBM Plex Sans Arabic/"+str(all_image_idx[i])+".jpeg")
    removed_noise = remove_noise(image)
    kp , descriptors= sift.detectAndCompute(removed_noise,None)
    if descriptors is not None:
        all_des=np.vstack((all_des,descriptors))
    print("processing type 1 image"+str(i))

num_of_desc+=[all_des.shape[0]]
print("FINISHED READING FIRST SET OF IMAGES")

for i in range (num_train_samples):
    image = cv2.imread("../../fonts-dataset/Lemonada/"+str(all_image_idx[i])+".jpeg")
    removed_noise = remove_noise(image)
    kp , descriptors= sift.detectAndCompute(removed_noise,None)
    if descriptors is not None:
        all_des=np.vstack((all_des,descriptors))
    print("processing type 2 image"+str(i))

num_of_desc+=[all_des.shape[0]]
print("FINISHED READING SECOND SET OF IMAGES")

for i in range (num_train_samples):
    image = cv2.imread("../../fonts-dataset/Marhey/"+str(all_image_idx[i])+".jpeg")
    removed_noise = remove_noise(image)
    kp , descriptors= sift.detectAndCompute(removed_noise,None)
    if descriptors is not None:
        all_des=np.vstack((all_des,descriptors))
    print("processing type 3 image"+str(i))

num_of_desc+=[all_des.shape[0]]
print("FINISHED READING THIRD SET OF IMAGES")


for i in range (num_train_samples):
    image = cv2.imread("../../fonts-dataset/Scheherazade New/"+str(all_image_idx[i])+".jpeg")
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


# different options for kmeans
# - mini batches
# - limiting the number of descriptors
kmeans=MiniBatchKMeans(n_clusters=num_of_centroids,batch_size=num_of_centroids*1024,max_iter=20).fit(X=all_des)


# for each centroid, calculate how it is near to each label
centroids_labels=np.zeros((num_of_centroids,4))
for i in range(num_of_centroids):
    centroids_labels[i][0]=(np.sum(desc_labels[kmeans.labels_==i]==0))  ## kmeans.labels_ -> array to map each descriptor to a centroid
    centroids_labels[i][1]=(np.sum(desc_labels[kmeans.labels_==i]==1))
    centroids_labels[i][2]=(np.sum(desc_labels[kmeans.labels_==i]==2))
    centroids_labels[i][3]=(np.sum(desc_labels[kmeans.labels_==i]==3))
    centroids_labels[i]/=np.sum(centroids_labels[i])
print(centroids_labels[0][0], centroids_labels[0][1], centroids_labels[0][2], centroids_labels[0][3])

num_right0=0
num_right1=0
num_right2=0
num_right3=0

predictions=[]

for i in range (num_test_samples):
    rand_idx=str(all_image_idx[num_train_samples+i])
    path="../../fonts-dataset/IBM Plex Sans Arabic/"+rand_idx+".jpeg"
    predicted=predict(path,kmeans.cluster_centers_,centroids_labels)
    print("image",(rand_idx), "type0: ",predicted)
    num_right0+=predicted==0

for i in range (num_test_samples):
    rand_idx=str(all_image_idx[num_train_samples+i])
    path="../../fonts-dataset/Lemonada/"+rand_idx+".jpeg"
    predicted=predict(path,kmeans.cluster_centers_,centroids_labels)
    print("image",(rand_idx), "type1: ",predicted)
    num_right1+=predicted==1

for i in range (num_test_samples):
    rand_idx=str(all_image_idx[num_train_samples+i])
    path="../../fonts-dataset/Marhey/"+rand_idx+".jpeg"
    predicted=predict(path,kmeans.cluster_centers_,centroids_labels)
    print("image",(rand_idx), "type2: ",predicted)
    num_right2+=predicted==2

for i in range (num_test_samples):
    rand_idx=str(all_image_idx[num_train_samples+i])
    path="../../fonts-dataset/Scheherazade New/"+rand_idx+".jpeg"
    predicted=predict(path,kmeans.cluster_centers_,centroids_labels)
    print("image",(rand_idx), "type3: ",predicted)
    num_right3+=predicted==3

print(num_right0)
print(num_right1)
print(num_right2)
print(num_right3)


print((num_right0+num_right1+num_right2+num_right3)/(4*num_test_samples))