import json
import os

import face_recognition as frec

D = path.__file__

PATH = "G:\\T_T\\sth\\face\\know_face"
SAVE_PATH = os.path.join(D, "know_face.json")


def bulid_from_file():
    know_face_info = []
    for file in os.listdir(PATH):
        img = frec.load_image_file(os.path.join(PATH, file))
        face_array = frec.face_encodings(img)[0]
        know_face_info.append({"name": file.split('.')[0],
                               "feature": list(face_array)})

    json.dump({"length": len(know_face_info),
               "face": know_face_info}, open(SAVE_PATH, 'w'))
    print('Save {0} faces successfully.'.format(len(know_face_info)))


if __name__ == '__main__':
    bulid_from_file()
