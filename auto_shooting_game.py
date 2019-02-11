#https://scratch.mit.edu/projects/14796524/       #link of the game

import pyautogui
import numpy as np
import cv2
import time

target1 = cv2.imread("melont2.png", 0)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(r'guns.mp4',fourcc, 2.0, (767,573))
midpadd = int(target1.shape[0]/2)
while True:
 
    image = pyautogui.screenshot()
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    imgc= img[145:718,299:1066] 
   
    allx=[]
    gray_img = cv2.cvtColor(imgc, cv2.COLOR_BGR2GRAY)

    result = cv2.matchTemplate(gray_img, target1 ,cv2.TM_CCOEFF_NORMED)
    loc = np.where(result >= 0.45)
    for pt in zip(*loc):
        pt= pt[::-1]
        cv2.rectangle(imgc, (pt[0] , pt[1] ), (pt[0] + target1.shape[1], pt[1] + target1.shape[0] ), (255, 145, 67), 2)  
        cv2.putText(imgc,'Hit here',(pt[0] , pt[1]  +90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2, cv2.LINE_AA)


        if pt:
            pyautogui.moveTo(pt[0]+ 299 +midpadd, pt[1] +145 +midpadd)
            time.sleep(0.2)
            pyautogui.click()
            #pyautogui.click(x=pt[0]+ 299 +midpadd, y=pt[1] +145 +midpadd)            
        break 
    cv2.putText(imgc,'After Image processing',(0,70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,25,155), 2, cv2.LINE_AA)

    cv2.imshow('imagerescc',cv2.resize(imgc, (0,0), fx=0.4, fy=0.4) )

    out.write(imgc)
    k = cv2.waitKey(1)
    if k == 27:   
        break
cv2.destroyAllWindows()
out.release()
