one = cv2.imread('backhandone.jpg',0)
two = cv2.imread('backhandtwo.jpg',0)
three = cv2.imread('backhandthree.jpg',0)
four = cv2.imread('backhandfour.jpg',0)
five = cv2.imread('backhandfive.jpg',0)

allshapes = [one, two,  three ,four ,five ]
shapenames= ['1','2', '3', '4', '5']

allcontours=[]

for shape in allshapes:
    ret, mask = cv2.threshold(shape, 10, 255, cv2.THRESH_BINARY)
    _,contours, hierarchy = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    c = max(contours, key = cv2.contourArea)
    cv2.drawContours(shape, [c], 0, 0, 3)
    allcontours.append(c)

    cv2.imshow("contour mask",mask)
    cv2.imshow("Shape",shape)
    cv2.waitKey(0)
    
cv2.destroyAllWindows()



def find_object(frame, object_hist):
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # apply back projection to image using object_hist as
    # the model histogram
    object_segment = cv2.calcBackProject(
        [hsv_frame], [0, 1], object_hist, [0, 180, 0, 256], 1)

    _, segment_thresh = cv2.threshold(object_segment, 90, 255, cv2.THRESH_BINARY)

    # apply some image operations to enhance image
    kernel = None
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    filtered = cv2.filter2D(segment_thresh, -1, disc)

    eroded = cv2.erode(filtered, kernel, iterations=1)
    dilated = cv2.dilate(eroded, kernel, iterations=1)
    closing = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)

    # masking
    masked = cv2.bitwise_and(frame, frame, mask=closing)

    return closing, masked, segment_thresh , object_segment, filtered
    
    
    
    
 def find_object2(frame, object_hist):
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    object_segment = cv2.calcBackProject([hsv_frame], [0, 1], object_hist, [0, 180, 0, 256], 1)
    _, segment_thresh = cv2.threshold(object_segment, 90, 255, cv2.THRESH_BINARY)

    kernel = None
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    filtered = cv2.filter2D(segment_thresh, -1, disc)

    eroded = cv2.erode(filtered, kernel, iterations=1)
    dilated = cv2.dilate(eroded, kernel, iterations=1)
    closing = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)
    masked = cv2.bitwise_and(frame, frame, mask=closing)
    
    return closing
    
    
    
def find_object_backsub(imb, imf,filtert):
    
        filterthresh = filtert
        imf = np.int64(imf)
        imb = np.int64(imb)
        img = imf - imb

        img[img < -filterthresh] = 255
        img[img < filterthresh] = 0
        img[img > filterthresh] = 255

        img = np.uint8(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)
        img = cv2.erode(img,None,iterations = 1)
        imgmask = cv2.dilate(img,None,iterations = 3)
        
        return imgmask
        
        
        
def find_object_hsv(frame,vals):
    useload =True
    handval = vals
    kernel = np.ones((5,5),np.uint8)

    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    if useload:
            lower_red = handval[0]
            upper_red = handval[1]
    else:        
    # define range of target color in HSV
      lower_red = np.array([0, 51, 105]) 
      upper_red = np.array([13, 149, 255])


    #Threshold the HSV image to get only target colors
    mask = cv2.inRange(hsv, lower_red, upper_red)
    #mask = cv2.erode(mask,kernel,iterations = 1)
    mask = cv2.dilate(mask,kernel,iterations = 1)
    #mask = cv2.erode(mask,kernel,iterations = 1)

    
    return mask        
    
    
# returns position of finger on template and modified closing2
def checkroi(closing2):
    position = None
    _,contours, hierarchy = cv2.findContours(closing2,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if contours and cv2.contourArea(max(contours, key = cv2.contourArea)) > 200:
        c = max(contours, key = cv2.contourArea)
        hull = cv2.convexHull(c)
       # cv2.drawContours(frame, [hull], -1, 255, 3)
        cv2.drawContours(closing2, [hull], -1, 255, 3) 
        M = cv2.moments(c)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        cv2.circle(closing2,(cx,cy),4,(0,255,0),2)
        position = (cx,cy)
    return position, closing2
    
    
import numpy as np
import cv2
import time
from statistics import mode
from statistics import mode, StatisticsError

#since we are gonna be doing background subtraction so you dont need to worry about different methods for now
method ='back' #'hist'  #'back'  #'hsv'

grabhist = False
if grabhist:
   gethist = capture_histogram()

handval = np.load('handextt.npy')
filterthresh = 30    # backgorund subtraction threshold  #tuneable param
thresh = 1         # threshold for shape detector
tester = []
cap = cv2.VideoCapture(0)

kernel = None
mostocc = 'N/A'  # storage value unit
imb = None       # background storage
title ='None'    # just the window title
res_text='N/A'   # to store all the text
space= False   # to check if i took my hand out of the box
newval= False   # to check if a fresh vlue was created
check_cell= -1  # check previous if its the same value , if i added a opperator then this becomes  -2
nums = set(['1','2','3','4','5'])
just_deleted= False     # to tell if a value was just deleted
oplist= ['+','-','*','/','=','X','Cl']   #list of all operators
imbtemp = None      # the background of the operators template
#optext = 
pointercount= []   # to keep count of all n finger points on template
finalpointertemp = None    # intermediate pointer value
finalpointer = None        # final point value after preprocessing
executed = False       # if all were the values were computed
outsideofrect= True     # is the pointer of template out of template
cv2.namedWindow('frame', cv2.WINDOW_NORMAL)

while(True):
    ret, frame = cap.read()

    frame = cv2.flip( frame, 1 ) 
    if not ret:
       break
    cv2.rectangle(frame, (430, 0), (635, 210), (0, 200, 10), 2)
    boxf = frame[5:205, 435:630]
    canvas = np.zeros(boxf.shape[:2],dtype= np.uint8)
    tempframe = frame[10:50, 5:71*(len(oplist)-1)]

    
    if method == 'hsv':
        title = 'Using Hsv Color Extraction'
        closing = find_object_hsv(boxf,handval)
        
    elif method == 'hist':
        title = 'Using Histogram backprojection'
        closing = find_object2(boxf, gethist)
        
    else:
            if imb is None:
                 time.sleep(2)
                 imb = boxf
                 imbtemp = tempframe
                 continue
            title = 'Using Custom Background Subtraction'        
            closing = find_object_backsub(imb, boxf, filterthresh)
            closing2 = find_object_backsub(imbtemp, tempframe, filterthresh)
            
    position, closing2 = checkroi(closing2)
    
    if position is not None:
        cv2.circle(frame,position,4,(0,0,0),-1)
        cv2.circle(frame,position,13,(100,25,110),4)
        for i,op in enumerate(oplist):
            #cv2.putText(frame,str(op),(i*65 +5,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)
            if position[0]  >  (i*65 +5) and position[0]  <  ((i+1)*65 +5):
                finalpointertemp= op
                pointercount.append(op)
    else:
        finalpointertemp = None
        pointercount = []
        outsideofrect = True
    for i,op in enumerate(oplist):
        if finalpointertemp == op:
            width = 1.3
            col = (0,0,255)
        else:
            width =0.9
            col= (255,0,0)
        cv2.putText(frame,str(op),(i*65 +5,40), cv2.FONT_HERSHEY_SIMPLEX, width, col, 2, cv2.LINE_AA)
        

        
    if len(pointercount) > 10:
        try:
            finalpointer = str(mode(pointercount))
        except StatisticsError:
            pointercount = []            
        if finalpointer == 'Cl':
            res_text= 'N/A'
            executed = False
        if res_text[-1] != finalpointer and not executed:            
            check_cell= -2
            if (finalpointer == 'X' and not just_deleted ) or (finalpointer == 'X' and outsideofrect):
               if len(res_text) > 1: 
                   res_text =  res_text[:-1]
                   newval = False
                   if res_text[-1]  in nums:
                        check_cell= -1
                        pass
                   else:
                        check_cell= -2
                        pass
                   just_deleted = True  

            elif finalpointer == '=':
                res_text = '=' + str( eval(res_text) )
                executed = True
                just_deleted = False 


            elif finalpointer in oplist and finalpointer is not "Cl" and finalpointer is not 'X':
                res_text +=   str(finalpointer)
                just_deleted = False 

            outsideofrect = False
        
    #cv2.putText(frame,str(finalpointer),(10,180), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2, cv2.LINE_AA)
    
   # cv2.rectangle(frame,(5,10),((65+5 +1)*(len(oplist) -1),50),(0,100,200),1)

    _,contours, hierarchy = cv2.findContours(closing,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if contours and cv2.contourArea(max(contours, key = cv2.contourArea)) > 1200:
        c = max(contours, key = cv2.contourArea)
        hull = cv2.convexHull(c)
       # cv2.drawContours(frame, [hull], -1, 255, 3)
        cv2.drawContours(canvas, [hull], -1, 255, 3) 

        scores=[]

        for cnt in allcontours:
           score = cv2.matchShapes(hull,cnt,1,0.0)
           scores.append(score)

        pos = np.argmin(np.array(scores))
        finalscore = min(scores)

        if finalscore < thresh:
            label = shapenames[pos]
        else:
            label = 'Unknown Shape'
        if len(tester) < 15:
            tester.append(label)
        else:
            try:
              if str(mode(tester)) == 'Unkown Shape':
                tester =[]
                continue
              mostocc = str(mode(tester))
              newval = True
            except:
                tester = []
            tester =[]

        if res_text == 'N/A' and newval:
            res_text= str(mostocc)
            check_cell=-1
            space = False
        elif (res_text != 'N/A' and not just_deleted and (res_text[check_cell] != mostocc) or (space and newval  )):
            res_text +=   str(mostocc)
            check_cell=-1
            space = False
        elif just_deleted and newval:
             res_text +=   str(mostocc)
             just_deleted = False
             check_cell=-1
             space = False


                    
        else:
            pass
        
        newval= False

    
    else:
        tester =[]  
        space = True
        
    cv2.putText(frame,res_text,(10,280), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2, cv2.LINE_AA)
    #cv2.putText(frame,title,(1,30), cv2.FONT_HERSHEY_SIMPLEX, .7, (100,25,200), 2, cv2.LINE_AA)
    cv2.imshow('closing',closing )
    cv2.imshow('canvas',canvas)
    cv2.imshow('c2',closing2)
    cv2.imshow('frame',frame)
    

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imwrite('backhandone.jpg',canvas)
        break  

cap.release()
cap2.release()
cv2.destroyAllWindows()    
    
