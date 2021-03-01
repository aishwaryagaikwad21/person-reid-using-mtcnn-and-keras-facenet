from mtcnn.mtcnn import MTCNN
import cv2
import numpy as np
from sorting import *
import sys
# from util import *
# cascPath = sys.argv[1]
vid = cv2.VideoCapture(0)
video_frame_cnt = int(vid.get(7))
video_width = int(vid.get(3))
video_height = int(vid.get(4))
video_fps = int(vid.get(5))
record_video = True
if record_video:
    out = cv2.VideoWriter('data/outvideo.avi', cv2.VideoWriter_fourcc('M', 'J',
                          'P', 'G'), video_fps, (video_width, video_height))  # for writing Video
face_detector = MTCNN()  # Initializing MTCNN detector object
# face_tracker  = Sort(max_age=50)   #Initializing SORT tracker object

ret, frame = vid.read()
conf_t = 0.99
while ret:
    ret, frame = vid.read()
    original_frame = frame.copy()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_detector.detect_faces(frame)
    print(result)

    box = []
    for res in result:
        x1, y1, width, height = res['box']
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height
            # print("Face Detectes: Box=",box_)
        confidence = res['confidence']
        if confidence < conf_t:
            continue
        key_points = res['keypoints'].values()
        cv2.imshow('Video', frame)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255, 0),2)

        # for point in key_points:
        #     cv2.circle(frame, point, 5, (0, 255, 0), thickness=-1)
            # box.append([box_[0],box_[1],box_[0]+box_[2],box_[1]+box_[3],result[i]["confidence"] ])

    #     dets = np.array(box)
    #     track_bbx,pts = face_tracker.update(dets)
    #     _MODEL_SIZE = [original_frame.shape[1], original_frame.shape[0]]
    #     original_frame = track_img(original_frame, track_bbx, _MODEL_SIZE, pts)

    #     if record_video:
    #         out.write(original_frame)

    #     cv2.imshow("out_frame", original_frame)
    # except Exception as e:
    #     print (getattr(e, 'message', repr(e)))

    keyPressed = cv2.waitKey(0) & 0xFF
    if keyPressed == 27: # ESC key
        break

# When everything is done, release the capture
vid.release()
cv2.destroyAllWindows()
