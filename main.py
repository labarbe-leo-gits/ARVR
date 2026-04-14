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
    min_detection_confidence=0.9,
    min_tracking_confidence=0.9,
    max_num_hands=2
)

mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

currentStroke = []
strokes = []
deletedStroke = []
strokeColors = []
drawing = False
eraser = False
ERASER_RADIUS = 20
eraser_point = None
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255), (0, 0, 0), (255, 255, 255)]
currentColor = 0
maxColor = len(colors) - 1
colorChangeDelay = 2.0 # 2 secondes
lastColorChangeTime = 0.0

while True:

    now = time.time()

    rect, frame = cam.read()
    frame = cv2.flip(frame, 1) # Miroir
    
    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(imgRGB)
    eraser = False
    eraser_point = None

    #print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks and results.multi_handedness:

        for handLms, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):

            label = MessageToDict(handedness)['classification'][0]['label']

            thumb = None
            index = None
            ringFinger = None
            middleFinger = None
            pinkie = None

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
                elif id == 20:
                    pinkie = (cx, cy)

            if label == 'Left' and thumb and index:
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
                            strokeColors.append(currentColor)
                            currentStroke = []

            if label == 'Right' and thumb and middleFinger:
                pinch_distance = math.hypot(middleFinger[0] - thumb[0], middleFinger[1] - thumb[1])
                if pinch_distance < 20:
                    cv2.putText(frame, "ERASER", (10, 120), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                    cv2.circle(frame, middleFinger, 15, (0, 255, 0), 2)
                    eraser = True
                    eraser_point = middleFinger

            if label == 'Right' and thumb and ringFinger:
                pinch_distance = math.hypot(ringFinger[0] - thumb[0], ringFinger[1] - thumb[1])
                if pinch_distance < 25:
                    cv2.putText(frame, "CLEAR", (10, 120), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                    strokes.clear()
                    strokeColors.clear()

            if label == 'Left' and thumb and pinkie and not drawing:
                pinch_distance = math.hypot(pinkie[0] - thumb[0], pinkie[1] - thumb[1])
                if pinch_distance < 12 and (now - lastColorChangeTime) >= colorChangeDelay:
                    cv2.putText(frame, "NEXT COLOR", (10, 120), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                    if currentColor < maxColor:
                        currentColor = currentColor + 1
                    else:
                        currentColor = 0
                    lastColorChangeTime = now

            if eraser and eraser_point is not None:
                newStrokes = []
                newStrokeColors = []
                for stroke, color in zip(strokes, strokeColors):
                    newStroke = []
                    for point in stroke:
                        dist = math.hypot(eraser_point[0] - point[0], eraser_point[1] - point[1])
                    
                        if dist >= ERASER_RADIUS:
                            newStroke.append(point)
                    
                    if newStroke:
                        newStrokes.append(newStroke)
                        newStrokeColors.append(color)
                strokes = newStrokes
                strokeColors = newStrokeColors

            if drawing and currentStroke:
                cv2.polylines(frame, [np.array(currentStroke, dtype=np.int32)], False, colors[currentColor], 4)

            """ for stroke in deletedStroke:
                if len(stroke) > 1:
                    strokes.remove(stroke)
                    deletedStroke.clear() """

            for idx, stroke in enumerate(strokes):
                if len(stroke) > 1:
                    color = colors[strokeColors[idx]]
                    cv2.polylines(frame, [np.array(stroke, dtype=np.int32)], False, color, 4)

            mpDraw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)
    
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    remaining = max(0.0, colorChangeDelay - (now - lastColorChangeTime))
    colorReady = remaining <= 0.0

    cv2.putText(frame, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 3)
    cv2.putText(frame, "CURRENT COLOR", (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, colors[currentColor], 3)

    if colorReady:
        countdownText = "NEXT COLOR READY"
        countdownColor = (0, 255, 0)
    else:
        countdownText = f"NEXT COLOR IN : {remaining:.1f}s"
        countdownColor = (0, 255, 255)

    cv2.putText(frame, countdownText, (100, 70), cv2.FONT_HERSHEY_COMPLEX, 0.8, countdownColor, 2)

    #out.write(frame)

    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

out.release()
cam.release()
cv2.destroyAllWindows()