import numpy as np
import cv2
fps=0
face_cascade = cv2.CascadeClassifier('haarcascade_fullbody.xml')
cap = cv2.VideoCapture('ped1.mp4')
resizer = 0.5
cv2.namedWindow('img', cv2.WINDOW_NORMAL)

while 1:
        start_time = time.time()

        ret, image = cap.read()
        if not ret:
             break
            #image = cv2.flip( image, 1 )
        imagec = cv2.resize(image, (0,0), fx=resizer, fy=resizer)
        ratio = 1 / resizer
        gray = cv2.cvtColor(imagec, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray)
        for (x,y,w,h) in faces:
                x= int(x * ratio)
                y =int( y * ratio )
                w = int(w * ratio)
                h =int( h * ratio)
            
                cv2.rectangle(image,(x ,y),(x+w,y+h),(0,0,255),2)
                cv2.putText(image,'Pedesterain',(x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 1, cv2.LINE_AA)

                
        cv2.putText(image, 'FPS: {:.2f}'.format(fps), (20, 20),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=0.5,color=(0, 0, 255))
        cv2.imshow('img',image)
        fps= (1.0 / (time.time() - start_time))

        k = cv2.waitKey(1)
        if k == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
