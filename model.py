import mediapipe as mp
import cv2
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Flatten
import numpy as np
import os

mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

image_data_dir = "hagrid-sample-30k-384p/hagrid_30k"
landmark_data_dir = "landmarks"

def extract_landmarks(image_path, landmark_save_path):
  image = cv2.imread(image_path)
  results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

  if results.multi_hand_landmarks:
    for hand_landmarks in results.multi_hand_landmarks:
      landmarks = []
      for lm in hand_landmarks.landmark:
        lmx = int(lm.x * image.shape[1])
        lmy = int(lm.y * image.shape[0])
        landmarks.append([lmx, lmy])

      with open(landmark_save_path, 'w') as f:
        f.write(','.join(str(x) for x in landmarks))
  else:
    print(f"No hands detected in image: {image_path}")

for i in os.listdir(image_data_dir):
    path = os.path.join(image_data_dir, i)
    landmark_dir = os.path.join(landmark_data_dir, i)
    os.makedirs(landmark_dir, exist_ok=True)

    for j in os.listdir(path):
      image_path = os.path.join(path, j)
      landmark_path = os.path.join(landmark_dir, os.path.splitext(j)[0] + '.csv')
      if not os.path.exists(landmark_path):
        extract_landmarks(image_path, landmark_path)

x_data = []
y_data = []
datacount = 0
for class_name in os.listdir(landmark_data_dir):
  class_path = os.path.join(landmark_data_dir, class_name)
  class_index = os.listdir(image_data_dir).index(class_name)

  for landmark_file in os.listdir(class_path):
    landmark_path = os.path.join(class_path, landmark_file)
    with open(landmark_path, 'r') as f:
      landmarks = np.array([float(x.strip('[]')) for x in f.read().split(',')]).reshape(-1, 2)
      x_data.append(landmarks)
      y_data.append(class_index)
      datacount+=1
x_data = np.array(x_data)
x_data = x_data/255
y_data = np.array(y_data)
y_data = y_data.reshape(datacount,1)
y_data = to_categorical(y_data)
flattened_x_data = x_data.reshape(-1, x_data.shape[1] * x_data.shape[2])
x_train, x_, y_train, y_ = train_test_split(flattened_x_data, y_data, train_size = 0.7, random_state=42)
x_val, x_test, y_val, y_test = train_test_split(x_, y_, train_size=0.5, random_state=42)

model = Sequential([
  Flatten(input_shape=(x_train.shape[1],)),
  Dense(512, activation='relu'),
  Dense(512, activation='relu'),
  Dense(256, activation='relu'),
  Dense(18, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(x_train, y_train , batch_size= 64, epochs= 50, verbose= 1, validation_data=(x_val, y_val))