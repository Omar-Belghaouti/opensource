# this code is somewhat modified version of Adrian's Hog pedesterain  example
import imutils
from imutils.object_detection import non_max_suppression
import numpy as np
import cv2
import time
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
fps =0
cap = cv2.VideoCapture('ped1.mp4')

cv2.namedWindow('image2', cv2.WINDOW_NORMAL)
bouncer = True
rsize= 280
while(True):
    start_time = time.time()

    ret, image = cap.read()
    if bouncer:

        if ret:

            #image = cv2.flip( frame, 1 ) 
            ratio = image.shape[1] /rsize

            cv2.putText(image, 'FPS: {:.2f}'.format(fps), (20, 20),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=0.5,color=(0, 0, 255))
            orig = image.copy()
            image = imutils.resize(image, width=min(rsize,image.shape[1]))
            (rects, weights) = hog.detectMultiScale(image, winStride=(8, 8), padding=(8, 8), scale=1.04)

            rects = np.array([[x, y, x + w, y + h]  for (x, y, w, h) in rects])
            pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

            for (x1, y1, x2, y2) in pick:
                cv2.rectangle(orig, (int(x1 * ratio), int(y1 *ratio)),( int(x2 * ratio), int(y2 * ratio)), (100, 205, 250), 2)
                cv2.putText(orig,'Pedesterain',(int(x1 * ratio),int(y1 *ratio)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 1, cv2.LINE_AA)

            cv2.imshow("image2", orig)
            fps= (1.0 / (time.time() - start_time))
            #bouncer = False   #ucomment  to run twice as fast on saved video
        else:
          break
    else: 
        bouncer = True
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()   
