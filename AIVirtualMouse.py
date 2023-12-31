import cv2
import time
import HandTracking as ht
import numpy as np
import pyautogui as pg
import math

################################

pg.FAILSAFE = False
wCam,hCam = 640,480
frameR = 100
smoothening = 2

################################

cap = cv2.VideoCapture(0)
cap.set(3,wCam)
cap.set(4,hCam)
pTime = 0
plocX,plocY = 0,0
clocX,clocY = 0,0
detector = ht.HandDetector()
wScr,hScr = pg.size()

while True:
    success,img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img)

    if len(lmList) != 0:
        x1,y1 = lmList[8][1:]
        x2,y2 = lmList[12][1:]
        
        fingers = detector.fingersUp()
        # print(fingers)
        cv2.rectangle(img,(frameR,frameR),(wCam-frameR,hCam-frameR),(255,0,255),2)

        if fingers[1] and fingers[2]==0:
            x3 = np.interp(x1,(frameR,wCam-frameR),(0,wScr))
            y3 = np.interp(y1,(frameR,hCam-frameR),(0,hScr))
            
            clocX = plocX + (x3-plocX) / smoothening
            clocY = plocY + (y3-plocY) / smoothening

            pg.moveTo(wScr-clocX,clocY)
            cv2.circle(img,(x1,y1),15,(255,0,255),cv2.FILLED)
            plocX,plocY = clocX,clocY

        if fingers[1] and fingers[2]:
            x1,y1 = lmList[8][1] , lmList[8][2]
            x2,y2 = lmList[12][1] , lmList[12][2]
            length = math.hypot(x2-x1,y2-y1)

            if length < 40:
                cv2.circle(img,(x1,y1),15,(0,255,0),cv2.FILLED)
                pg.click()

    cTime = time.time() 
    fps = 1/(cTime-pTime)
    pTime = cTime
    
    cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
    cv2.imshow("Image",img)
    cv2.waitKey(10)
