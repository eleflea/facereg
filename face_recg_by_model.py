import json
import face_recognition as frec
import cv2
import numpy as np
import bulid_db
from xpinyin import Pinyin
from lib import ismale, init_model

p = Pinyin()

video_capture = cv2.VideoCapture(0)

face_db = json.load(open(bulid_db.SAVE_PATH))
face_encoding_list = [np.array(face['feature']) for face in face_db['face']]
face_name_list = [face['name'] for face in face_db['face']]
print('Load {0} faces from database ({1}).'.format(face_db['length'], bulid_db.SAVE_PATH))

# Initialize some variables
gender_model = init_model()
face_locations = []
face_encodings = []
face_names = []
face_gender = []
num_this_frame = 0

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Only process every other frame of video to save time
    if num_this_frame % 4 == 0:
        # Find all the faces and face encodings in the current frame of video
        face_locations = frec.face_locations(small_frame)
        face_encodings = frec.face_encodings(small_frame, face_locations)

        face_names = []
        if face_encodings:
            face_gender = ['male' if b else 'female' for b in ismale(gender_model, face_encodings)]
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            match = frec.compare_faces(face_encoding_list, face_encoding, tolerance=0.45)
            name = ''

            try:
                i = match.index(True)
            except ValueError:
                name = "Unknown"
            if not name:
                name = face_name_list[i]

            face_names.append(p.get_pinyin(name, ''))

    num_this_frame += 1


    # Display the results
    for (top, right, bottom, left), name, gender in zip(face_locations, face_names, face_gender):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom), (right, bottom + 50), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 5, bottom + 20), font, 0.8, (255, 255, 255), 1)
        cv2.putText(frame, gender, (left + 5, bottom + 45), font, 0.8, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Face Recognize', frame)
    # print(list(zip(face_names, face_gender)))

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
