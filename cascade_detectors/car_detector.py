import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('carshaar.xml')
cap = cv2.VideoCapture('carsvid.wmv')
cv2.namedWindow('img', cv2.WINDOW_NORMAL)

while 1:
    
        ret, image = cap.read()
        if not ret:
             break
            #image = cv2.flip( image, 1 )
        imagec = cv2.resize(image, (0,0), fx=0.5, fy=0.5)
        ratio = 1/0.5
        gray = cv2.cvtColor(imagec, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray)
        for (x,y,w,h) in faces:
                x= int(x * ratio)
                y =int( y * ratio )
                w = int(w * ratio)
                h =int( h * ratio)
            
                cv2.rectangle(image,(x ,y),(x+w,y+h),(0,0,255),2)
                cv2.putText(image,'Car Detected',(x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,255,0), 1, cv2.LINE_AA)
                   
        cv2.imshow('img',image)
        k = cv2.waitKey(1) 
        if k == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
