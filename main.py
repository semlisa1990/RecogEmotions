#!usr/bin/env python3
from mss import mss
import cv2
from PIL import Image
import numpy as np
from fer import FER
import datetime
import matplotlib.pyplot as plt
import os

file_prefix = 'emotion_data'
file_suffix = 1
file_name = f'{file_prefix}{file_suffix}.txt'

# Генерируем имя файла для сохранения данных
while os.path.exists(file_name):
    file_suffix += 1
    file_name = f'{file_prefix}{file_suffix}.txt'

file = open(file_name, 'w')
if file:
    print(f"Файл {file_name} успешно создан")
else:
    print("Ошибка создания файла")
file.write("time,person_id,angry,disgust,fear,happy,sad,surprise,neutral\n")

emotion_data = {}
fig, ax = plt.subplots()
ax.set_title('Emotion Graph')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Emotion')

x_data = []
y_data = []

mon = {'top': 0, 'left': 0, 'width': 700, 'height': 700}
sct = mss()

emo_detector = FER(mtcnn=True)
start = datetime.datetime.now()
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    sct_img = sct.grab(mon)
    img = Image.frombytes('RGB', (sct_img.size.width, sct_img.size.height), sct_img.rgb)
    img_np = np.array(img)
    gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)

    # Находим лица на изображении
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    emotions = emo_detector.detect_emotions(img_np)
    now = datetime.datetime.now()
    count = now - start

    # Проверяем обнаруженные эмоции для каждого лица
    for idx, (x, y, w, h) in enumerate(faces):
        cv2.rectangle(img_np, (x, y), (x + w, y + h), (255, 255, 0), 2)

        # Идентификатор лица (можно использовать индекс в списке лиц)
        person_id = idx

        if idx < len(emotions):
            person_emotions = emotions[idx]['emotions']
            dominant_emotion = max(person_emotions, key=person_emotions.get)
            emotion_score = person_emotions[dominant_emotion]

            # Записываем данные эмоций в файл
            values_str = ','.join([f"{score:.2f}" for score in person_emotions.values()])
            file.write(f"{str(count)[:-7]},{person_id},{values_str}\n")

            # Отображаем эмоции на экране
            y_pos = y + h + 20
            for emotion, score in person_emotions.items():
                text = f"{emotion}: {score:.2f}"
                cv2.putText(img_np, text, (x, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                y_pos += 20
        else:
            dominant_emotion = 'none'

        cv2.putText(img_np, dominant_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

    # Визуализация данных на графике
    x_data.append(count.seconds)
    y_data.append(emotion_data.get(str(count)[:-7], 'none'))

    ax.clear()
    ax.set_title('Emotion Graph')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Emotion')
    ax.plot(x_data, y_data)

    plt.pause(0.01)
    cv2.imshow('Emotion Detection', img_np)

    # Остановка программы при нажатии 'q'
    if cv2.waitKey(25) == ord('q'):
        file.close()
        print('Файл записан')
        cv2.destroyAllWindows()
        break
