import cv2
import mediapipe as mp
import time

class HandDetector():
    def __init__(self,mode=False,maxHands=1,detectionCon=int(0.5),trackCon=int(0.5)):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode,self.maxHands,self.detectionCon,self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils 
        self.tips = [4,8,12,16,20]

    
    def findHands(self,img,draw = True): 
        imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for self.handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img,self.handLms,self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self,img,handNo=0,draw=True):
        self.lmList = []
        if self.results.multi_hand_landmarks:
            for id,lm in enumerate(self.handLms.landmark):
                h , w, c = img.shape
                cx , cy = int(lm.x * w) , int(lm.y * h)
                self.lmList.append([id,cx,cy])
                # if id == 9  or id == 10 or id == 11 or id == 12 and draw:
                #     cv2.circle(img,(cx,cy),10,(255,0,0),cv2.FILLED)
        return self.lmList
    
    def fingersUp(self):
        finger = []
        
        if self.lmList[self.tips[0]][1] > self.lmList[self.tips[0]-1][1]: 
                finger.append(1)
        else:
            finger.append(0)
            
        for id in range(1,5):
            if self.lmList[self.tips[id]][2] < self.lmList[self.tips[id]-2][2]:
                finger.append(1)
            else:
                finger.append(0)
                
        return finger
def main():
    cap = cv2.VideoCapture(0)
    pTime = 0
    detector = HandDetector()
    while True:
        success , img = cap.read()
        img = detector.findHands(img)
        lmList = detector.findPosition(img)
        if len(lmList) != 0:
            print(lmList[4])
            
        cTime = time.time()
        fps = 1/(cTime - pTime)
        pTime = cTime

        cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
        cv2.imshow("Image",img)
        cv2.waitKey(10)
    
if __name__ == "__main__": 
    main()
    
