import face_recognition as frec
import cv2
import numpy as np
from xpinyin import Pinyin
from lib import ismale


class VideoCamera(object):
    def __init__(self, data):
        self.video = cv2.VideoCapture(2)
        self.gender_model, self.face_encoding_list, self.face_name_list = data
        self.face_locations = []
        self.face_encodings = []
        self.face_names = []
        self.face_gender = []
        self.p = Pinyin()
        self.num_this_frame = 0

    def __del__(self):
        self.video.release()

    def analysis(self):
        ret, frame = self.video.read()
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        if self.num_this_frame % 4 == 0:
            self.face_locations = frec.face_locations(small_frame)
            self.face_encodings = frec.face_encodings(small_frame, self.face_locations)

            self.face_names = []
            if self.face_encodings:
                self.face_gender = ['male' if b else 'female' for b in ismale(self.gender_model, self.face_encodings)]
            for face_encoding in self.face_encodings:
                match = frec.compare_faces(
                    self.face_encoding_list, face_encoding, tolerance=0.5)
                name = ''

                try:
                    i = match.index(True)
                except ValueError:
                    name = "Unknown"
                if not name:
                    name = self.face_name_list[i]
                self.face_names.append(self.p.get_pinyin(name))
        self.num_this_frame += 1

        for (top, right, bottom, left), name, gender in zip(self.face_locations, self.face_names, self.face_gender):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.rectangle(frame, (left, bottom),
                          (right, bottom + 50), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 5, bottom + 20),
                        font, 0.8, (255, 255, 255), 1)
            cv2.putText(frame, gender, (left + 5, bottom + 45),
                        font, 0.8, (255, 255, 255), 1)
        return frame

    def push_frame(self):
        image = self.analysis()
        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()
