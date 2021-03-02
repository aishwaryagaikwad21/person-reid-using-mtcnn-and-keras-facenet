import cv2
import os
from mtcnn import MTCNN

capture = cv2.VideoCapture(0)
detector = MTCNN()
count = 0
path = './datasets/Test/Elena'
while True:
    files = 0
    for _, dirnames, filenames in os.walk(path):
        files += len(filenames)
        print(files)
      
    __, frame = capture.read()
    result = detector.detect_faces(frame)
    if result != []:
        count+=1
        face = cv2.resize(frame,(400,400))
        for person in result:
            t_r = person['box'] #traced rectangle
            key_points = person['keypoints']
            cv2.rectangle(frame,(t_r[0], t_r[1]),(t_r[0] + t_r[2],t_r[1] + t_r[3]),(0,155,255),2)
            cv2.circle(frame,(key_points['left_eye']), 2, (0,155,255), 2)
            cv2.circle(frame,(key_points['right_eye']), 2, (0,155,255), 2)
            cv2.circle(frame,(key_points['nose']), 2, (0,155,255), 2)
            cv2.circle(frame,(key_points['mouth_left']), 2, (0,155,255), 2)
            cv2.circle(frame,(key_points['mouth_right']), 2, (0,155,255), 2)
            cv2.imshow('frame',frame)
            c = files+1
            file_name_path = './datasets/Test/Elena/'+ str(c) + '.jpg'
            cv2.imwrite(file_name_path,face)

        if cv2.waitKey(1) &0xFF == ord('q') or count==100: #press q on keyboard
            break

capture.release()
cv2.destroyAllWindows()

