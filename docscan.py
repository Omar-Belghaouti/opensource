import numpy as np
import cv2

def looper2(image,pts,scale):
    global holder,mover
    pts = pts/(1/scale)
    for key, p in zip(holder.keys(),pts):
        holder[key] = p

    cimage = image.copy()
    firstime= True
    while True:
          if mover | firstime :
              firstime = False
              cimage = image.copy()

    #drawing lines between points
              for i,xx in enumerate(holder.values()):
                    if i == 0:
                        prev = xx
                        first = xx
                    else:
                        cv2.line(cimage,tuple(prev),tuple(xx),(255,100,100),2)
                        if i == 3:
                            cv2.line(cimage,tuple(first),tuple(xx),(255,100,100),2)
                        prev = xx

    #just for the sake of the design i'm drawing the circles in a seperate loop to avoid connecting lines on top of the image

              for i,xx in enumerate(holder.values()):
                    cv2.circle(cimage,tuple(xx), 15, (200,50,90), -1)
                    cv2.circle(cimage,tuple(xx), 16, (255,0,255), 2)

                    cv2.putText(cimage,str(i),tuple(xx), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20,255,155), 2, cv2.LINE_AA)

         # cv2.putText(cimage,'Press q to extract and drag the /n balls to correct',(15,15), cv2.FONT_HERSHEY_SIMPLEX,
          #            0.5, (20,255,155), 1, cv2.LINE_AA)
          cv2.putText(cimage,'Press q to Extract',(1 ,15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20,255,155), 1, cv2.LINE_AA)
          cv2.putText(cimage,'Drag the circles to correct',(1 ,35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20,255,155), 1, cv2.LINE_AA)

          cv2.imshow('Detection',cimage)
          if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    dvals = holder.values()
    ptlist =[]
    for x in dvals:
       ptlist.append(x)
    pts1 = np.float32(np.array(ptlist).reshape(-1,2)*(1/scale))
    return pts1

def getshort(targetp,holder):
    shortestdist=None
    short={}
    for key,allpoints in holder.items():
      dist = np.linalg.norm(targetp-allpoints)
      if shortestdist is None:
            shortestdist = dist
            short[key] = targetp
      elif shortestdist > dist:
        shortestdist = dist
        short={}
        short[key] = targetp

    changedstr,changedval = list(short.items())[0]

    return changedstr,changedval

from tkinter import Tk
from tkinter.filedialog import askopenfilename
Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
file = askopenfilename() # show an "Open" dialog box and return the path to the selected file
#file = easygui.fileopenbox()
img = cv2.imread(file)
imgorg = img.copy()
htresh = 500
if img.shape[0] > htresh:
    scale = htresh / img.shape[0]
    img = cv2.resize(img, (0,0), fx=scale, fy=scale)
#img = cv2.resize(img, (0,0), fx=scale, fy=scale)
else:
    scale = 1
print(scale, img.shape)
mover =False
img2 = img.copy()
rows, cols, chan=imgorg.shape

imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

edges1 = cv2.Canny(imgray,219,390)

blurred = cv2.GaussianBlur(edges1,(5,5),0)

_,contours, hierarchy = cv2.findContours(blurred,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

c = max(contours, key = cv2.contourArea)

cv2.drawContours(img, [c], 0, (0,255,0), 3)#draw the 3rd contour
epsilon = 0.05*cv2.arcLength(c,True)
approx = cv2.approxPolyDP(c,epsilon,True)

pts1 = np.float32(approx.reshape(-1,2)*(1/scale))
pts2 = np.float32([[cols,0],[0,0],[0,rows],[cols,rows]]) # change this line according to structure

def draw_circle(event,x,y,flags,param):
       # if event == cv2.EVENT_LBUTTONDBLCLK:
        if event == cv2.EVENT_MOUSEMOVE:
            global holder,mover
            if mover:
                st,val=getshort((x,y),holder)
                holder[st]= np.array(val)

        if event == cv2.EVENT_LBUTTONDOWN:
               mover = True
        elif event == cv2.EVENT_LBUTTONUP:
               mover = False

holder = {'0th':None,'1st':None,'2nd':None ,'3rd':None}
cv2.namedWindow('Detection')
cv2.setMouseCallback('Detection',draw_circle)

#if you cant find the doc then approximate the doc is there in middle
if len(approx) != 4:
        pts1= np.float32([[cols-cols/4,rows/4],[cols-cols/1.3,rows/4],[cols-cols/1.3,rows/1.3],[cols-cols/4,rows/1.3]])

pts1 =  looper2(img2,pts1,scale)
#cv2.namedWindow('Extracted Document', cv2.WINDOW_NORMAL)
M = cv2.getPerspectiveTransform(pts1,pts2)
dst = cv2.warpPerspective(imgorg,M,(cols,rows))
cv2.imwrite('Saveddoc.jpg',dst)
if dst.shape[0]  > 500:
     dst = cv2.resize(dst, None, fx=scale, fy=scale)
     cv2.putText(dst,'Not orignal Size',(1 ,20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 1, cv2.LINE_AA)
     cv2.imshow('Extracted Document',dst)
else:
    cv2.imshow('Extracted Document',dst)

#Because puttext cant handle \n
cv2.putText(img,'Press q to exit and the image will',(1 ,15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20,255,155), 1, cv2.LINE_AA)
cv2.putText(img,'be saved in current directory',(1 ,35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20,255,155), 1, cv2.LINE_AA)

cv2.imshow('Detection',img)
cv2.waitKey(0)
cv2.destroyAllWindows()