import cv2
from mtcnn import MTCNN

capture = cv2.VideoCapture(0)
detector = MTCNN()
while True:
      # Continuous capture of the video feed
        __, frame = capture.read()
        result = detector.detect_faces(frame)
        if result != []:
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
        if cv2.waitKey(1) &0xFF == ord('q'): #press q on keyboard
            break

capture.release()
cv2.destroyAllWindows()

