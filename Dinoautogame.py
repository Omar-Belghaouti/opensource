
import pyautogui
import numpy as np
import cv2
from pynput.keyboard import Key, Controller
keyboard = Controller()
gap=0

sp1 = cv2.imread("dinor/spike.PNG", 0)
sp2 = cv2.imread("dinor/spiked.PNG", 0)
sp3 = cv2.imread("dinor/spiket.PNG", 0)
sp4 = cv2.imread("dinor/spike4.PNG", 0)
sp5 = cv2.imread("dinor/spike2.PNG", 0)
sp6 = cv2.imread("dinor/bird1.PNG", 0)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(r'dino.mp4',fourcc, 3.0, (620,166))

thresh =446
obstacles = [sp4,sp3,sp2,sp1,sp5,sp6]
initval =147
bouncer = initval
orig =366
ext = thresh -orig
while True:
 
    image = pyautogui.screenshot()
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    imgc= img[115:281,thresh:986] 
    
    img = img[115:281,orig:986] 
    
   
    allx=[]
    gray_img = cv2.cvtColor(imgc, cv2.COLOR_BGR2GRAY)
   
    for temp in obstacles:
        result = cv2.matchTemplate(gray_img, temp,cv2.TM_CCOEFF_NORMED)
        loc = np.where(result >= 0.67)
        
        for pt in zip(*loc):
            pt= pt[::-1]
            cv2.rectangle(img, (pt[0] +ext, pt[1] ), (pt[0] + temp.shape[1] +ext, pt[1] + temp.shape[0] ), (255, 145, 67), 2)    
            #print(pt, pt[0],pt[1]) # this is the top left corner
            allx.append(pt[0]) #appending the x position only
            
            break 
    if len(allx) >1:      
        gap =  np.sort(allx)[1] - np.sort(allx)[0]
        if gap < 286 and gap > 5:
                bouncer = 165
        else:
            bouncer = initval
    else:
        bouncer = initval        
            
    if allx:   
        cv2.putText(img, '{}'.format(np.sort(allx)[0]), (10, 13),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=0.5,color=(0, 0, 255))    
        cv2.putText(img, '{}'.format(bouncer), (50, 13),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=0.5,color=(0, 0, 255))    
        cv2.putText(img, '{}'.format(gap), (100, 13),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=0.5,color=(0, 0, 255))    
        cv2.putText(img, 'Processed Image', (140, 13),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=0.5,color=(0, 0, 255))    

        if np.sort(allx)[0] <  bouncer:
              keyboard.press(Key.up)
              keyboard.release(Key.up)        
                 
    cv2.imshow('imageres',img)
    #cv2.imshow('imagerescc',imgc)

    out.write(img)
    k = cv2.waitKey(1)
    if k == 27:   
        break
cv2.destroyAllWindows()
out.release()
