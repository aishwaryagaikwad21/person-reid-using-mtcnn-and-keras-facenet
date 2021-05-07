
import cv2
from matplotlib import pyplot
from mtcnn.mtcnn import MTCNN
from numpy import asarray
from PIL import Image     
# Load the cascade  
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')  
# To capture video from existing video.   
cap = cv2.VideoCapture(0)  
count = 0     
while True:  
        # Read the frame
    required_size=(160,160)  
    _, img = cap.read()
    detector = MTCNN()
    results = detector.detect_faces(img)
    x1, y1, width, height = results[0]['box']
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
	# extract the face
    face = img[y1:y2, x1:x2]
	# resize pixels to the model size
    image = Image.fromarray(face)
    image = image.resize(required_size)
    face_array = asarray(image)
    print(face_array)
    # Convert to grayscale  
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)        
    # Detect the faces  
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    # Draw the rectangle around each face  
    for (x, y, w, h) in faces:  
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)  
    # Display  
    cv2.imshow('Video', img)  
    # Stop if escape key is pressed  
    k = cv2.waitKey(30) & 0xff  
    if k==27:  
        break  
# Release the VideoCapture object  
cap.release()  
