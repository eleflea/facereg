import cv2
from time import sleep
import recognize
import numpy as np
from PIL import Image

cap = cv2.VideoCapture(0)
faceCascade = cv2.CascadeClassifier(
    "G:\\T_T\\exe\\facereg\\haarcascade_frontalface_default.xml")

PATH = 'G:\\T_T\\sth\\face\\xl5.bmp'

f = 0
rate = [0 for _ in range(10)]
x, y, w, h = 0, 0, 0, 0
while True:
    # 读取一帧
    ret, frame = cap.read()

    # Detect faces in the image
    faces = faceCascade.detectMultiScale(
        frame,
        scaleFactor=1.05,
        minNeighbors=20,
        minSize=(80, 80),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # print("Found {0} faces!".format(len(faces)))
    if cv2.waitKey(1) & 0xFF == ord('c'):
        face = Image.fromarray(np.uint8(frame))
        face = face.crop((faces[0][0], faces[0][1], faces[0][0] + faces[0][2], faces[0][1] + faces[0][3]))
        face.save("G:\\T_T\\sth\\face\\temp.bmp")
        PATH = "G:\\T_T\\sth\\face\\temp.bmp"
        cv2.putText(frame, 'Capture face successfully.', (100, 100),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))
    # Draw a rectangle around the faces
    i = 0
    for x, y, w, h in faces:
        if f % 20 == 0:
            # rate = [0 for _ in range(10)]
            # print(len(faces))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            height, width = gray.shape
            # face = Image.new('L', (width, height))
            face = Image.fromarray(np.uint8(gray))
            face = face.crop((x, y, x + w, y + h))
            face = face.resize((64, 64))
            face.save("G:\\T_T\\sth\\face\\00{0}.bmp".format(i))
            # face.show()
            rate[i] = (recognize.compare(
                face.getdata(), PATH, recognize.vector_cos))
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, str(rate[i]), (x + 5, y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))
        i += 1

    # 显示帧
    cv2.imshow('faceReg', frame)
    f += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放VideoCapture对象
cap.release()
cv2.destroyAllWindows()
