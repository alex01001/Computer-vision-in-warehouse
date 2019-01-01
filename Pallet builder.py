# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 13:15:59 2018

"""

import math
import numpy as np
import cv2

#dictionary of all contours
contours = {}
#array of edges of polygon
approx = []
#scale of the text
scale = 2
#camera
cap = cv2.VideoCapture(0)
print("press ESC to exit")

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))

palletNumbers=[]
numFrames=40
frameCounter = 0

maxRolls = []
maxRollsCurrent = []

# Helper function to return a given contour
def c(index):
    global contours
    return contours[index]

# Count the number of real children
def count_children(index, h_, contour):
    # No children
    if h_[0][index][2] < 0:
        return 0
    else:
        #If the first child is a contour we care about
        # then count it, otherwise don't
        if keep(c(h_[0][index][2])):
            count = 1
        else:
            count = 0

            # Also count all of the child's siblings and their children
        count += count_siblings(h_[0][index][2], h_, contour, True)
        return count


def keep(contour):
    approx = cv2.approxPolyDP(contour,cv2.arcLength(contour,True)*0.03,True)
    if(abs(cv2.contourArea(contour))<100 or not(cv2.isContourConvex(approx))):
        return False
    return True

# Count the number of relevant siblings of a contour
def count_siblings(index, h_, contour, inc_children=False):
    # Include the children if necessary
    count = 0
    # Look ahead
    p_ = h_[0][index][0]
    while p_ > 0:

        if keep(c(p_)):
            count += 1
        p_ = h_[0][p_][0]

    # Look behind
    n = h_[0][index][1]
    while n > 0:
        if keep(c(n)):
            count += 1
        if inc_children:
            count += count_children(n, h_, contour)
        n = h_[0][n][1]
    return count

#calculate angle
def angle(pt1,pt2,pt0):
    dx1 = pt1[0][0] - pt0[0][0]
    dy1 = pt1[0][1] - pt0[0][1]
    dx2 = pt2[0][0] - pt0[0][0]
    dy2 = pt2[0][1] - pt0[0][1]
    return float((dx1*dx2 + dy1*dy2))/math.sqrt(float((dx1*dx1 + dy1*dy1))*(dx2*dx2 + dy2*dy2) + 1e-10)

while(cap.isOpened()):
    #Capture frame-by-frame
    ret, frame = cap.read()
    palletNumbers=[]
    palletCount = 0
    
    frameCounter+=1

    if ret==True:
        #grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #Canny
        canny = cv2.Canny(frame,80,240,3)
        
        canny2, contours, hierarchy = cv2.findContours(canny,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        for i in range(0,len(contours)):
            #approximate the contour with accuracy proportional to
            #the contour perimeter
            approx = cv2.approxPolyDP(contours[i],cv2.arcLength(contours[i],True)*0.03,True)

            #Skip small or non-convex objects
            if(abs(cv2.contourArea(contours[i]))<100 or not(cv2.isContourConvex(approx))):
                continue

            #triangle
            if(len(approx) == 3):
                x,y,w,h = cv2.boundingRect(contours[i])
#                cv2.putText(frame,'TRI',(x,y),cv2.FONT_HERSHEY_SIMPLEX,scale,(255,255,255),2,cv2.LINE_AA)
            elif(len(approx)>=4 and len(approx)<=4):
                #nb vertices of a polygonal curve
                vtc = len(approx)
                #get cos of all corners
                cos = []
                for j in range(2,vtc+1):
                    cos.append(angle(approx[j%vtc],approx[j-2],approx[j-1]))
                #sort ascending cos
                cos.sort()
                #get lowest and highest
                mincos = cos[0]
                maxcos = cos[-1]

                #Use the degrees obtained above and the number of vertices
                #to determine the shape of the contour
                x,y,w,h = cv2.boundingRect(contours[i])
                if(vtc==4):
                    rect = cv2.minAreaRect(contours[i])
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)
                    cv2.drawContours(frame,[box],0,(255,0,0),2)
                    a = np.array(hierarchy)
                    parentNum = hierarchy[0][i][3]
                    if(parentNum>-1):
                        chi = count_children(i, hierarchy, contours[i])
                        palletCount+=1
                        
                        try:
                            mr = maxRollsCurrent[palletCount-1]
                            if(chi>mr):
                                maxRollsCurrent[palletCount-1]=chi

                            if(frameCounter>numFrames):
                                frameCounter=0
                                for j in range(0,len(maxRollsCurrent)):
                                    maxRolls[j]=maxRollsCurrent[j]
                                for j in range(0,len(maxRollsCurrent)):
                                    maxRollsCurrent[j]=0
                        except:
                            maxRollsCurrent.append(chi)
                            maxRolls.append(chi)

                        cv2.putText(frame,'pallet '+ str(palletCount) +' (' + str(maxRolls[palletCount-1]) +' rolls)',(x,y),cv2.FONT_HERSHEY_DUPLEX,1,(250,250,250),1,cv2.LINE_AA)
                        
                elif(vtc==5):
                    cv2.putText(frame,'PENTA',(x,y),cv2.FONT_HERSHEY_SIMPLEX,scale,(255,255,255),2,cv2.LINE_AA)
                elif(vtc==6):
                    cv2.putText(frame,'HEXA',(x,y),cv2.FONT_HERSHEY_SIMPLEX,scale,(255,255,255),2,cv2.LINE_AA)
            else:
                #detect and label circle
                area = cv2.contourArea(contours[i])
                x,y,w,h = cv2.boundingRect(contours[i])
                roi_color = frame[y:y+h, x:x+w]

                radius = w/2
                if(abs(1 - (float(w)/h))<=2 and abs(1-(area/(math.pi*radius*radius)))<=0.2):
                    cv2.rectangle(roi_color,(x,y),(x+w,y+h),(0,255,0),2)
                    (x,y),radius = cv2.minEnclosingCircle(contours[i])
                    center1 = (int(x),int(y))
                    radius1 = int(radius)
                    cv2.circle(frame,center1,radius1,(0,255,0),2)

        #Display the resulting frame
        imS = cv2.resize(frame, (960, 780)) 
        cv2.imshow('frame',imS)
        cv2.imshow('canny',canny)

        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
        
        if cv2.waitKey(1) == 1048689: #if q is pressed
            break

#When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
