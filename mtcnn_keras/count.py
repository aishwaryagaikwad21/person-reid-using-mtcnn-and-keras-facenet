import os
import cv2
from PIL import Image
from keras.preprocessing import image
from numpy import asarray
from mtcnn.mtcnn import MTCNN
#For training set
files_train = folders_train = 0
path_train = 'archive/train'

for _, dirnames, filenames in os.walk(path_train):
  # ^ this idiom means "we won't be using this value"
    files_train += len(filenames)
    folders_train += len(dirnames)

directory = os.path.join(path_train,str(folders_train))
os.mkdir(directory)

count = 0
while count<11:
    file_train = 'archive/train/'+ str(folders_train) + '/' + str(count) + '.jpg'
    image = "joe.jpg"
    img = cv2.imread(image)
    print(file_train)
    cv2.imwrite(file_train, img)
    count+=1
    #print(img)


#For val set
files_val = folders_val = 0
path_val = 'archive/val'

for _, dirnames, filenames in os.walk(path_val):
  # ^ this idiom means "we won't be using this value"
    files_val += len(filenames)
    folders_val += len(dirnames)

directory = os.path.join(path_val,str(folders_val))
os.mkdir(directory)

count = 0
while count<6:
    file_val = 'archive/val/'+ str(folders_val) + '/' + str(count) + '.jpg'
    image = "joe.jpg"
    img = cv2.imread(image)
    print(file_val)
    cv2.imwrite(file_val, img)
    count+=1