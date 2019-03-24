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
    mask = cv2.Canny(shape,100,300)
    mask = cv2.dilate(mask,None,iterations = 2)
    _,contours, hierarchy = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    c = max(contours, key = cv2.contourArea)
    cv2.drawContours(shape, [c], 0, 0, 3)
    allcontours.append(c)

    cv2.imshow("mask",mask)
    cv2.imshow("Shape",shape)
    cv2.waitKey(0)
    
cv2.destroyAllWindows()





def imagedec2(testimg1):
    testimg = cv2.cvtColor(testimg1, cv2.COLOR_BGR2GRAY)

    thresh = 0.3

    mask = cv2.Canny(testimg,100,300)

    _,contours, hierarchy = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    for testc in contours:
        if cv2.contourArea(testc) > 300:

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
    return testimg1 , mask
    
    
 
 
 
import cv2
import numpy as np
drawing= False

cv2.namedWindow('image')
radius =10
x1 =None
x2 =0
y1 =0
y2=0

width = 1000
height = 600


def draw_line(event,x,y,flags,param):
    global x1,y1, x2,y2,drawing

    if event == cv2.EVENT_MOUSEMOVE:
        x2= x
        y2= y       

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        x1=None
        
img = np.zeros((height,width,3), np.uint8)
imga = np.zeros((height,width,3), np.uint8)
cv2.setMouseCallback('image',draw_line)

while True:
    if drawing:
        #print(x1)
   
        if x1 is not None:                  
            img = cv2.line(img, (x1,y1),(x2,y2), [255,255,255], 5)
            imga = cv2.line(imga, (x1,y1),(x2,y2), [255,255,255], 5)

            x1= x2
            y1 = y2
        else:
            x1,y1 =x2,y2
   
    cv2.imshow('image',img)
    
    k= cv2.waitKey(1)
    if k  == ord('c'):
        img = np.zeros((height,width,3), np.uint8)
        imga = np.zeros((height,width,3), np.uint8)

    elif k  == ord('p'):
        img,_ = imagedec2(imga.copy())  #if you do not add copy it modifies the image

    elif k & 0xFF == 27:
      #  cv2.imwrite('shapecheckk.jpg',img)
        break
cv2.destroyAllWindows()
    
    
    
    
    
    
    
#DETECTION ON PAPER
import cv2
cam = cv2.VideoCapture(0)      
cv2.namedWindow('img', cv2.WINDOW_NORMAL)
while (True):
  ret ,frame = cam.read()   
  if ret:

      #frame = cv2.flip( frame, 1 )
    
      img ,mask = imagedec2(frame)
      cv2.imshow("img",img)     

      if cv2.waitKey(1)  & 0xFF ==ord('a'):  
          break
      if cv2.waitKey(1)  & 0xFF ==ord('s'):
          cv2.imwrite('contshapereal.jpg',frame)
          break
#release capture when all done      
cam.release()
cv2.destroyAllWindows()    
