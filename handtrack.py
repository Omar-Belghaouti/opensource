import dlib
import cv2
import time
detector = dlib.simple_object_detector("myhanddetector2.svm")

import numpy as np
import cv2
cv2.namedWindow('frame', cv2.WINDOW_NORMAL)

cap = cv2.VideoCapture(0)
fps= 0
rscale=2.0
siz =0
while(True):
    start_time = time.time()
    ret, frame = cap.read()
    frame = cv2.flip( frame, 1 )
    cv2.putText(frame, 'FPS: {:.2f}'.format(fps), (20, 10),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=0.3,color=(0, 0, 255))
    cv2.putText(frame, 'size: {}'.format(siz), (100, 10),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=0.3,color=(0, 0, 255))

    ft = cv2.resize(frame, (int(frame.shape[1]/rscale), int(frame.shape[0]/rscale)))
    dets = detector(ft)
    #print(len(dets))    
    for d in (dets):    
        #cv2.rectangle(frame,(d.left(),d.top()),(d.right(), d.bottom()),(0,255,0),3)
        cv2.rectangle(frame,(int(d.left()*rscale),int(rscale*d.top())),(int(rscale*d.right()), int(rscale*d.bottom())),(0,255,0),3)
        siz = int(rscale*d.right())- int(d.left()*rscale) 
    
    cv2.imshow('frame',frame)
    fps= (1.0 / (time.time() - start_time))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
