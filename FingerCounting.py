import os
import time
import cv2
import HandTracking as ht

wCam,hCam = 640,480
cap = cv2.VideoCapture(0)
cap.set(3,wCam)
cap.set(4,hCam)
pTime = 0

detector = ht.HandDetector(detectionCon=int(0.7))
tipId = [4,8,12,16,20]


while True:
    success,img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img,draw=False)
    # print(lmList)
    if len(lmList) != 0:
        finger = []
        if lmList[tipId[0]][1] > lmList[tipId[0]-1][1]: 
                finger.append(1)
        else:
            finger.append(0)
        for id in range(1,5):
            if lmList[tipId[id]][2] < lmList[tipId[id]-2][2]:
                finger.append(1)
            else:
                finger.append(0)
        # print(finger)
        totalFingers = finger.count(1)
        print(totalFingers)
        cv2.rectangle(img,(20,255),(170,425),(0,255,0),cv2.FILLED)
        cv2.putText(img,str(totalFingers),(45,375),cv2.FONT_HERSHEY_SCRIPT_COMPLEX,3,(255,0,0),3)
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img,(f"FPS:{str(int(fps))}"),(400,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),3)
    cv2.imshow("Image",img)
    cv2.waitKey(10)

