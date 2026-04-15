import cv2
import mediapipe as mp
from google.protobuf.json_format import MessageToDict
import time
import math
import numpy as np
import screeninfo

primary = next((m for m in screeninfo.get_monitors() if m.is_primary), None)

if primary is None:
    primary = screeninfo.get_monitors()[0]

cam = cv2.VideoCapture(0) # 0 = caméra par défaut
cam.set(3, 0.75*(primary.width))
cam.set(4, 0.75*(primary.height))

#fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#width = int(cam.get(3))
#height = int(cam.get(4))
#out = cv2.VideoWriter('AR_Demo.avi', fourcc, 20.0, (width, height))

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

def dist2D(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

# Ratios pour distance dynamique
DRAW_RATIO_THR = 0.45
ERASER_RATIO_THR = 0.40
CLEAR_RATIO_THR = 0.45
NEXT_COLOR_RATIO_THR = 0.30
ERASER_RADIUS_RATIO = 0.35
RECORDING_RATIO_THR = 0.20

currentStroke = []
strokes = []
deletedStroke = []
strokeColors = []
drawing = False
eraser = False
recording = False
out = None
record_path = None
ERASER_RADIUS = 20
eraser_point = None
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255), (0, 0, 0), (255, 255, 255)]
currentColor = 0
maxColor = len(colors) - 1
colorChangeDelay = 2.0 # 2 secondes
lastColorChangeTime = 0.0
lastRecordToggleTime = 0.0
recordToggleDelay = 2.0
recordStartTime = 0.0

def startRecording(frame):
    global out, recording, record_path, recordStartTime
    if recording:
        return
    
    h, w = frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps_out = cam.get(cv2.CAP_PROP_FPS)
    if fps_out <= 1:
        fps_out = 20.0
    
    ts = time.strftime("%Y%m%d_%H%M%S")
    record_path = f"AR_Demo_{ts}.avi"
    out = cv2.VideoWriter(record_path, fourcc, fps_out, (w, h))
    recording = out.isOpened()
    recordStartTime = time.time()

def stopRecording():
    global out, recording, recordStartTime
    if out is not None:
        out.release()
        out = None
    recording = False
    recordStartTime = 0.0

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

            lms = handLms.landmark
            thumb_n = (lms[4].x, lms[4].y)
            index_n = (lms[8].x, lms[8].y)
            middle_n = (lms[12].x, lms[12].y)
            ring_n = (lms[16].x, lms[16].y)
            pinky_n = (lms[20].x, lms[20].y)

            hand_ref_n = dist2D((lms[5].x, lms[5].y), (lms[17].x, lms[17].y))
            if hand_ref_n < 1e-6:
                continue

            h, w = frame.shape[:2]
            p5 = (int(lms[5].x * w), int(lms[5].y * h))
            p17 = (int(lms[17].x * w), int(lms[17].y * h))

            hand_ref_px = dist2D(p5, p17)
            eraser_radius_px = max(10, int(hand_ref_px * ERASER_RADIUS_RATIO))

            if label == 'Left' and thumb and index:
                #pinch_ratio = math.hypot(index[0] - thumb[0], index[1] - thumb[1])
                pinch_ratio = dist2D(thumb_n, index_n) / hand_ref_n
                if pinch_ratio < DRAW_RATIO_THR:
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
                #pinch_ratio = math.hypot(middleFinger[0] - thumb[0], middleFinger[1] - thumb[1])
                pinch_ratio = dist2D(thumb_n, middle_n) / hand_ref_n
                if pinch_ratio < ERASER_RATIO_THR:
                    cv2.putText(frame, "ERASER", (10, 120), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                    cv2.circle(frame, middleFinger, eraser_radius_px, (0, 255, 0), 2)
                    eraser = True
                    eraser_point = middleFinger

            if label == 'Left' and thumb and middleFinger:
                pinch_ratio = dist2D(thumb_n, middle_n) / hand_ref_n
                if pinch_ratio < RECORDING_RATIO_THR and (now - lastRecordToggleTime) >= recordToggleDelay:
                    if recording:
                        stopRecording()
                    else:
                        startRecording(frame)
                    lastRecordToggleTime = now

            if label == 'Right' and thumb and ringFinger:
                #pinch_ratio = math.hypot(ringFinger[0] - thumb[0], ringFinger[1] - thumb[1])
                pinch_ratio = dist2D(thumb_n, ring_n) / hand_ref_n
                if pinch_ratio < CLEAR_RATIO_THR:
                    cv2.putText(frame, "CLEAR", (10, 120), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                    strokes.clear()
                    strokeColors.clear()

            if label == 'Left' and thumb and pinkie and not drawing:
                #pinch_ratio = math.hypot(pinkie[0] - thumb[0], pinkie[1] - thumb[1])
                pinch_ratio = dist2D(thumb_n, pinky_n) / hand_ref_n
                if pinch_ratio < NEXT_COLOR_RATIO_THR and (now - lastColorChangeTime) >= colorChangeDelay:
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
                    
                        if dist >= eraser_radius_px:
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

    if recording and out is not None:
        out.write(frame)
        if recordStartTime > 0:
            elapsed = now - recordStartTime
            cv2.putText(frame, f"REC {int(elapsed)}s", (10, 160), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

    #out.write(frame)

    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

if out is not None:
    out.release()
cam.release()
cv2.destroyAllWindows()