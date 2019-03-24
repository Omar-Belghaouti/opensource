import cv2
import numpy as np
circle = cv2.imread('shapeimages/circle.png',0)
square =  cv2.imread('shapeimages/square.png',0)
rectangle = cv2.imread('shapeimages/rect1.jpeg',0)
triangle =  cv2.imread('shapeimages/triangle.jpg',0)
star =  cv2.imread('shapeimages/star.jpg',0)
polygon =  cv2.imread('shapeimages/polygon.jpg',0)

allshapes = [circle, square,  rectangle ,triangle ,star, polygon]
shapenames= ['Circle','Square', 'Rectangle', 'Triangle', 'Star', 'Polygon']

allcontours=[]

for shape in allshapes:
    ret, mask = cv2.threshold(shape, 220, 255, cv2.THRESH_BINARY_INV)
    _,contours, hierarchy = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    c = max(contours, key = cv2.contourArea)
    cv2.drawContours(shape, [c], 0, 0, 3)
    allcontours.append(c)

    cv2.imshow("mask",mask)
    cv2.imshow("Shape",shape)
    cv2.waitKey(0)
    
cv2.destroyAllWindows()



def imagedec(testimg1):
    testimg = cv2.cvtColor(testimg1, cv2.COLOR_BGR2GRAY)

    thresh = 0.3

    ret, mask = cv2.threshold(testimg, 220, 255, cv2.THRESH_BINARY_INV)
    _,contours, hierarchy = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    for testc in contours:

        x,y,w,h = cv2.boundingRect(testc)
        cv2.rectangle(testimg1,(x,y),(x+w,y+h),(0,255,0),3)

        scores=[]
        for cnt in allcontours:
           score = cv2.matchShapes(testc,cnt,1,0.0)
           scores.append(score)

        pos = np.argmin(np.array(scores))
        finalscore = min(scores)

        if finalscore < thresh:
            label = shapenames[pos]
        else:
            label = 'Unknown Shape'

        cv2.putText(testimg1,label,(x,y+h+16), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100,55,50), 2, cv2.LINE_AA)
    return testimg1
    
    
    
from tkinter import Tk
from tkinter.filedialog import askopenfilename
Tk().withdraw() 
filename = askopenfilename() 
#print(filename)
testimg1= cv2.imread(filename)  
testimg1 = imagedec(testimg1)
cv2.imshow("Shape",testimg1)
cv2.waitKey(0)
cv2.destroyAllWindows()
