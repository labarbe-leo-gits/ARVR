import cv2
import mediapipe as mp
from google.protobuf.json_format import MessageToDict
import time
import math
import numpy as np

cam = cv2.VideoCapture(0) # 0 = caméra par défaut
cam.set(3, 200)
cam.set(4, 200)

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
drawing = False

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

            for id, lm in enumerate(handLms.landmark):
                #print(id, lm)
                h, w, c = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                #print(id, cx, cy)

                if id == 4:
                    thumb = (cx, cy)
                elif id == 8:
                    index = (cx, cy)

            if thumb and index:
                pinch_distance = math.hypot(index[0] - thumb[0], index[1] - thumb[1])
                if pinch_distance < 20:
                    cv2.putText(frame, "PINCH", (10, 120), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
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

            if drawing and currentStroke:
                cv2.polylines(frame, [np.array(currentStroke, dtype=np.int32)], False, (0, 255, 255), 4)

            for stroke in strokes:
                if len(stroke) > 1:
                    cv2.polylines(frame, [np.array(stroke, dtype=np.int32)], False, (0, 255, 255), 4)

            mpDraw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)
    
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(frame, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 3)

    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()