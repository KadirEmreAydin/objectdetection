import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO

model = YOLO('yolov8s.pt')

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:  
        colorsBGR = [x, y]
        print(colorsBGR)

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

cap = cv2.VideoCapture('objectdetection.mp4')

my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")
print(class_list)

count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    count += 1
    if count % 3 != 0:
        continue
    frame = cv2.resize(frame, (1280, 720))

    results = model.predict(frame)

    # Yeni: boxes, conf ve cls özelliklerini kullanarak verileri alın
    boxes = results[0].boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2 formatında kutular
    scores = results[0].boxes.conf.cpu().numpy()  # Güven puanları
    classes = results[0].boxes.cls.cpu().numpy()  # Sınıf etiketleri

    # Yeni: DataFrame'e dönüştürme
    px = pd.DataFrame(boxes, columns=['x1', 'y1', 'x2', 'y2'])
    px['conf'] = scores
    px['class'] = classes.astype(int)

    for index, row in px.iterrows():
        x1 = int(row['x1'])
        y1 = int(row['y1'])
        x2 = int(row['x2'])
        y2 = int(row['x2'])
        d = int(row['class'])
        c = class_list[d]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(frame, str(c), (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0), 2) 

    cv2.imshow("RGB", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
