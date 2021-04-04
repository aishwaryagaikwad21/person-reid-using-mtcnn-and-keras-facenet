#MAIN FILE.. Official for prediction
from os import listdir
from os.path import isdir
from PIL import Image
from matplotlib import pyplot
from numpy import savez_compressed
from numpy import asarray
from mtcnn.mtcnn import MTCNN
from numpy import expand_dims
from numpy import load
from keras.models import load_model
from random import choice
from numpy import load
from numpy import expand_dims
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
from matplotlib import pyplot
import os
import cv2
from PIL import Image
from keras.preprocessing import image
from numpy import asarray
from mtcnn.mtcnn import MTCNN
#load model
model = load_model('facenet_keras.h5')
#load numpy array dataset
data = load('faces-dataset.npz')
testX_faces = data['arr_2']
#image to array
input_image = 'images/geller.jpg'
required_size=(160,160)
image = Image.open(input_image)
image = image.convert('RGB')
pixels = asarray(image)
detector = MTCNN()
results = detector.detect_faces(pixels)
x1, y1, width, height = results[0]['box']
x1, y1 = abs(x1), abs(y1)
x2, y2 = x1 + width, y1 + height
	# extract the face
face = pixels[y1:y2, x1:x2]
	# resize pixels to the model size
image = Image.fromarray(face)
image = image.resize(required_size)
face_array = asarray(image) #imp
#Embedding
face_pixels = face_array.astype('float32')
	# standardize pixel values across channels (global)
mean, std = face_pixels.mean(), face_pixels.std()
face_pixels = (face_pixels - mean) / std
	# transform face into one sample
samples = expand_dims(face_pixels, axis=0)
	# make prediction to get embedding
yhat = model.predict(samples)
emb = yhat #imp
in_encoder = Normalizer(norm='l2')
emb = in_encoder.transform(emb)
# load face embeddings
data = load('faces-embeddings.npz')
trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
# normalize input vectors
in_encoder = Normalizer(norm='l2')
trainX = in_encoder.transform(trainX)
testX = in_encoder.transform(testX)
#print(trainX.shape[0])
# label encode targets0
out_encoder = LabelEncoder()
out_encoder.fit(trainy)
trainy = out_encoder.transform(trainy)
testy = out_encoder.transform(testy)
# fit model
model = SVC(kernel='linear', probability=True)
model.fit(trainX, trainy)
# test model on a random example from the test dataset
random_face_pixels = face_array
random_face_emb = emb[0]
# prediction for the face
samples = expand_dims(random_face_emb, axis=0)
yhat_class = model.predict(samples)
yhat_prob = model.predict_proba(samples)
# get name
#print(yhat_class)
class_index = yhat_class[0]
class_probability = yhat_prob[0,class_index] * 100
print(class_probability)
if(class_probability<40):
    print('Face not found')
    files_train = folders_train = 0
    path_train = 'archive/train'

    for _, dirnames, filenames in os.walk(path_train):
        files_train += len(filenames)
        folders_train += len(dirnames)

    directory = os.path.join(path_train,str(folders_train))
    os.mkdir(directory)

    count = 0
    while count<11:
        file_train = 'archive/train/'+ str(folders_train) + '/' + str(count) + '.jpg'
        img = cv2.imread(input_image)
        print(file_train)
        cv2.imwrite(file_train, img)
        count+=1
    #print(img)
    #For val set
    files_val = folders_val = 0
    path_val = 'archive/val'

    for _, dirnames, filenames in os.walk(path_val):
        files_val += len(filenames)
        folders_val += len(dirnames)

    directory = os.path.join(path_val,str(folders_val))
    os.mkdir(directory)

    count = 0
    while count<6:
        file_val = 'archive/val/'+ str(folders_val) + '/' + str(count) + '.jpg'
        img = cv2.imread(input_image)
        print(file_val)
        cv2.imwrite(file_val, img)
        count+=1
    
else:
    predict_names = out_encoder.inverse_transform(yhat_class)
    print('Predicted: %s (%.3f)' % (predict_names[0], class_probability))
    # print('Expected: %s' % random_face_name[0])
    pyplot.imshow(random_face_pixels)
    title = '%s (%.3f)' % (predict_names[0], class_probability)
    pyplot.title(title)
    pyplot.show()