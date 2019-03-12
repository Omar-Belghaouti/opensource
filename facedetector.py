import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')  
cap = cv2.VideoCapture(0)

while 1:
    
        ret, img = cap.read()
        img = cv2.flip( img, 1 )
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x,y,w,h) in faces:
                cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
                cv2.putText(img,'Face Detected',(x,y+h+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2, cv2.LINE_AA)   

                
        cv2.imshow('img',img)
        if cv2.waitKey(1) == ord('q'):
            break
            
cap.release()
cv2.destroyAllWindows()
