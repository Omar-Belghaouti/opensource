import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalcatface.xml')
cap = cv2.VideoCapture('catrec.wmv')
#cv2.namedWindow('img', cv2.WINDOW_NORMAL)

while 1:
    
        ret, image = cap.read()
        if not ret:
             break
            #image = cv2.flip( image, 1 )
        image = cv2.resize(image, (0,0), fx=0.7, fy=0.7)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray)
        for (x,y,w,h) in faces:
            
                cv2.rectangle(image,(x ,y),(x+w,y+h),(0,0,255),2)
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(image,'Cat Detected',(x,y+h+15), font, 0.5, (0,0,255), 2, cv2.LINE_AA)   

                   
        cv2.imshow('img',image)
        k = cv2.waitKey(1) 
        if k == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
