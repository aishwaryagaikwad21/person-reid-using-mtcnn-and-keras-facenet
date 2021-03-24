import cv2

# Load the cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# To capture video from webcam. 
cap = cv2.VideoCapture(0)
# To use a video file as input 
# cap = cv2.VideoCapture('filename.mp4')
count = 0
while True:
    # Read the frame
    _, img = cap.read()
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect the faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    # Draw the rectangle around each face
    for (x, y, w, h) in faces:
        count+=1
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

        face = img[y:y+h,x:x+w]
        file_path = './Dataset/Test/ele/'+ str(count) + '.jpg'
        cv2.imwrite(file_path,face)
    # Display
    cv2.imshow('img', img)
    # Stop if escape key is pressed
    k = cv2.waitKey(30) & 0xff
    if k==27 or count == 150:
        break
# Release the VideoCapture object
cap.release()