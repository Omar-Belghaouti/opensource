import dlib
import cv2

detector = dlib.simple_object_detector("myhanddetector2.svm")

import numpy as np
import cv2
#cv2.namedWindow('frame', cv2.WINDOW_NORMAL)

cap = cv2.VideoCapture(0)
fps= 0
rscale=2.0
from pynput.keyboard import Key, Controller
keyboard = Controller()
size =0
mid=0
#import pyautogui
 
while(True):
    start_time = time.time()
    ret, frame = cap.read()
    frame = cv2.flip( frame, 1 )
    cv2.putText(frame, 'FPS: {:.2f}'.format(fps), (20, 10),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=0.3,color=(0, 0, 255))
    cv2.putText(frame, 'size: {} and the mid is {}'.format(size,mid), (100, 10),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=0.3,color=(0, 0, 255))

    ft = cv2.resize(frame, (int(frame.shape[1]/rscale), int(frame.shape[0]/rscale)))
    dets = detector(ft)
    #print(len(dets))    
    if len(dets) == 0:
            keyboard.release(Key.left)
            keyboard.release(Key.right)
            keyboard.release(Key.up)
            keyboard.release(Key.down)

    for d in (dets):    
        #cv2.rectangle(frame,(d.left(),d.top()),(d.right(), d.bottom()),(0,255,0),3)
        left = int(rscale*d.left())
        right=int(rscale*d.right())
        top = int(rscale*d.top())
        bottom= int(rscale*d.bottom())
        cv2.rectangle(frame,(left,top),(right, bottom),(0,255,0),3)
        size = right - left
        center = size/2 
        mid = left  +  center
        
        if mid < 280:
            #print('turn left')
            keyboard.release(Key.right)

            keyboard.press(Key.left)
            time.sleep(0.08)

            keyboard.release(Key.left)

            #pyautogui.press('left')
            pass

            
            
        elif mid > 360:
           #print(' turn right')
            keyboard.release(Key.left)

            keyboard.press(Key.right)
            time.sleep(0.08)

            keyboard.release(Key.right)

            #pyautogui.press('right')
            pass

        else:    
            #print('stop turning')
            keyboard.release(Key.left)
            keyboard.release(Key.right)
            pass

        
        
        if size > 150:
           # print('moveforward')
            keyboard.press(Key.up)
            time.sleep(0.06)

            keyboard.release(Key.up)

            #pyautogui.press('up')

            
        elif size < 130:
            #print('moveback')
            keyboard.press(Key.down)
            time.sleep(0.06)
            keyboard.release(Key.down)
            #pyautogui.press('down')

            
        else:
           # print('stop moving')
            keyboard.release(Key.up)
            keyboard.release(Key.down)


            
    #cv2.imshow('frame',frame)
    fps= (1.0 / (time.time() - start_time))

    #if cv2.waitKey(1) & 0xFF == ord('q'):
     #   break
#keyboard.release(Key.left)
#keyboard.release(Key.right)
keyboard.release(Key.up)
keyboard.release(Key.down)
cap.release()
cv2.destroyAllWindows()
