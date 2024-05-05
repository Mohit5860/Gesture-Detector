import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import os

mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

model = tf.keras.models.load_model('lb.keras')

labels = []
path = "landmarks"
for i in os.listdir(path) :
    labels.append(i)

cap = cv2.VideoCapture("http://192.168.29.14:4747/video")

while True:

    _, frame = cap.read()

    x, y, c = frame.shape

    frame = cv2.flip(frame, 1)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(framergb)

    
    className = ''

    if result.multi_hand_landmarks:
        landmarks = []
        for handslms in result.multi_hand_landmarks:
            for lm in handslms.landmark:
                lmx = int(lm.x * x)
                lmy = int(lm.y * y)

                landmarks.append([lmx, lmy])

            mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)
            landmarks = np.array(landmarks)
            flattened = landmarks.reshape(-1, landmarks.shape[0] * landmarks.shape[1])
            prediction = model.predict(flattened)

            classID = np.argmax(prediction)
            className = labels[classID]

    cv2.putText(frame, className, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (0,0,255), 2, cv2.LINE_AA)

    cv2.imshow("Output", frame) 

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()

cv2.destroyAllWindows()