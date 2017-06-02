import tflearn
import json
import numpy as np
from os import path

D = path.dirname(__file__)

SAVE_PATH = path.join(D, "know_face.json")

def init_model():
    net = tflearn.input_data(shape=[None, 128])
    net = tflearn.fully_connected(net, 64, activation='tanh')
    net = tflearn.fully_connected(net, 32, activation='tanh')
    net = tflearn.dropout(net, 0.8)
    net = tflearn.fully_connected(net, 2, activation='softmax')
    net = tflearn.regression(net, optimizer='adam', learning_rate=0.001,
                             metric=None)
    model = tflearn.DNN(net, tensorboard_verbose=0)
    model.load("G:\\T_T\\exe\\facereg\\model\\gender4.model")
    return model

def ismale(model, encoding):
    return [data[0] > data[1] for data in model.predict(encoding)]

def load_encodings():
    face_db = json.load(open(SAVE_PATH))
    face_encoding_list = [np.array(face['feature']) for face in face_db['face']]
    face_name_list = [face['name'] for face in face_db['face']]
    print('Load {0} faces from database ({1}).'.format(face_db['length'], SAVE_PATH))
    return face_encoding_list, face_name_list
