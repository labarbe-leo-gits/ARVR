import cv2
import mediapipe as mp
from google.protobuf.json_format import MessageToDict
import time
import math
import numpy as np

cam = cv2.VideoCapture(0) # 0 = caméra par défaut
cam.set(3, 200)
cam.set(4, 200)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
width = int(cam.get(3))
height = int(cam.get(4))
out = cv2.VideoWriter('AR_Demo.avi', fourcc, 20.0, (width, height))

# Modèle main
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.75,
    max_num_hands=2
)

mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

currentStroke = []
strokes = []
deletedStroke = []
drawing = False
eraser = False
ERASER_RADIUS = 20
eraser_point = None

while True:
    rect, frame = cam.read()
    frame = cv2.flip(frame, 1) # Miroir
    
    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(imgRGB)

    #print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:

            thumb = None
            index = None
            ringFinger = None
            middleFinger = None

            for id, lm in enumerate(handLms.landmark):
                #print(id, lm)
                h, w, c = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                #print(id, cx, cy)

                if id == 4:
                    thumb = (cx, cy)
                elif id == 8:
                    index = (cx, cy)
                elif id == 12:
                    middleFinger = (cx, cy)
                elif id == 16:
                    ringFinger = (cx, cy)

            if thumb and index:
                pinch_distance = math.hypot(index[0] - thumb[0], index[1] - thumb[1])
                if pinch_distance < 15:
                    cv2.putText(frame, "DRAWING", (10, 120), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                    if not drawing:
                        drawing = True
                        currentStroke = []
                    currentStroke.append(index)
                else:
                    if drawing:
                        drawing = False
                        if currentStroke:
                            strokes.append(currentStroke)
                            currentStroke = []

            if thumb and middleFinger:
                for i in results.multi_handedness:
                    label = MessageToDict(i)['classification'][0]['label']
                    if label == 'Right':
                        pinch_distance = math.hypot(middleFinger[0] - thumb[0], middleFinger[1] - thumb[1])
                        if pinch_distance < 15:
                            cv2.putText(frame, "ERASER", (10, 120), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2) 
                            cv2.circle(frame, middleFinger, 15, (0, 255, 0), 2)
                            eraser = True
                            eraser_point = middleFinger
                        else:
                            eraser = False

            if thumb and ringFinger:
                for i in results.multi_handedness:
                    label = MessageToDict(i)['classification'][0]['label']
                    if label == 'Left':
                        pinch_distance = math.hypot(ringFinger[0] - thumb[0], ringFinger[1] - thumb[1])
                        if pinch_distance < 25:
                            cv2.putText(frame, "CLEAR", (10, 120), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                            strokes.clear()

            if eraser and eraser_point is not None:
                newStrokes = []
                for stroke in strokes:
                    newStroke = []
                    for point in stroke:
                        dist = math.hypot(eraser_point[0] - point[0], eraser_point[1] - point[1])
                    
                        if dist >= ERASER_RADIUS:
                            newStroke.append(point)
                    
                    if newStroke:
                        newStrokes.append(newStroke)
                strokes = newStrokes

            if drawing and currentStroke:
                cv2.polylines(frame, [np.array(currentStroke, dtype=np.int32)], False, (0, 255, 255), 4)

            """ for stroke in deletedStroke:
                if len(stroke) > 1:
                    strokes.remove(stroke)
                    deletedStroke.clear() """

            for stroke in strokes:
                if len(stroke) > 1:
                    cv2.polylines(frame, [np.array(stroke, dtype=np.int32)], False, (0, 255, 255), 4)

            mpDraw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)
    
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(frame, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 3)

    out.write(frame)

    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

out.release()
cam.release()
cv2.destroyAllWindows()