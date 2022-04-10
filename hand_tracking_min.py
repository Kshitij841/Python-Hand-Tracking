from calendar import c
import cv2
import mediapipe as mp
import time

#getting image source from webcam
cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
#Can change the parameter for hands by checking hands.py in python folder
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

#For FPS counter
pTime = 0
cTime = 0

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    #print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            #checking that bottom is zero..... top is highest
            for id, lm in enumerate(handLms.landmark):
                #print(id, lm) #to print in decimal values
                #height, width, channel
                h, w, c = img.shape
                #multiplying landmark x and y co-ordinates with the hegiht and width decimal to convert them to pixels
                cx, cy = int(lm.x*w), int(lm.y*h)
                print(id, cx, cy)
                if id == 0:
                    cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
    
    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10, 25), cv2.FONT_HERSHEY_PLAIN, 2 ,(255, 0, 255), 3)

    cv2.imshow("image", img)
    cv2.waitKey(1)