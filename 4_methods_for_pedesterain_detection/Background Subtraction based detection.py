import numpy as np
import cv2 
fps=0
cap = cv2.VideoCapture('vtest.avi')

#cap = cv.VideoCapture(0)
kernel= None
foog = cv2.bgsegm.createBackgroundSubtractorMOG()
while(1):
    ret, frame = cap.read()
    if ret:
        frameorig = frame.copy()
        start_time = time.time()


        
        #frame = cv2.GaussianBlur(frame,(5,5),0)
        fgmask = foog.apply(frame)
        fgmask = cv2.erode(fgmask,kernel,iterations = 1)
        fgmask = cv2.dilate(fgmask,kernel,iterations = 3)
        _,contours, hierarchy = cv2.findContours(fgmask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        #cv2.drawContours(frame, contours, -1, (0,255,0), 3)
        for cnt in contours:
            if cv2.contourArea(cnt) > 200:
                
                x,y,w,h = cv2.boundingRect(cnt)
                cv2.rectangle(frameorig,(x ,y),(x+w,y+h),(0,0,255),2)
                cv2.putText(frameorig,'Pedesterain',(x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,255,0), 1, cv2.LINE_AA)
       
    
    
    
        cv2.putText(frameorig, 'FPS: {:.2f}'.format(fps), (20, 20),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=0.5,color=(0, 0, 255))   
       
    
        cv2.imshow('frame',fgmask)
        cv2.imshow('framre',frameorig)

        fps= (1.0 / (time.time() - start_time))

        k = cv2.waitKey(1) & 0xff
        if k == ord('q'):
            break
    else: 
        break
cap.release()
cv2.destroyAllWindows()
