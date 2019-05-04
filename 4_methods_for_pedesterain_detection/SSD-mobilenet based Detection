
import numpy as np
import cv2
import time
CLASSES = ['background','person']
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
conf= 0.50
basepath='object-detection-deep-learning'
modelarch= 'msdped.prototxt.txt'
modelweights ='MobileNetSSD_deploy10695.caffemodel'
net = cv2.dnn.readNetFromCaffe(modelarch, modelweights)

cap = cv2.VideoCapture('ped1.mp4')
cv2.namedWindow('frame',cv2.WINDOW_NORMAL)
fps = 0
while 1: 
    start_time = time.time()
    ret, image = cap.read()
    #image = cv2.flip( frame, 1 ) 
    if not ret:
        break
    
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 0.007843,(300, 300), 127.5)
    cv2.putText(image, 'FPS: {:.2f}'.format(fps), (20, 20),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=0.3,color=(0, 0, 255))
    net.setInput(blob)
    detections = net.forward()
    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > conf:
        
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            x1 = int(detections[0, 0, i, 3] * w)
            y1 = int(detections[0, 0, i, 4] * h)
            x2 = int(detections[0, 0, i, 5] * w)
            y2 = int(detections[0, 0, i, 6] * h)

            label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
            cv2.rectangle(image, (x1, y1), (x2, y2), COLORS[idx], 2)
            cv2.putText(image, label, (x1, y1-10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
    cv2.imshow('frame',image)
    fps= (1.0 / (time.time() - start_time))
    if cv2.waitKey(1) & 0xFF == ord('q'):
           break    
        
cap.release()
cv2.destroyAllWindows()            
