import face_recognition as frec
import cv2
import os
import numpy as np
import tflearn
import tflearn.datasets.mnist as mnist
import matplotlib.pyplot as plt
import json
import random

def ing(gender):
    return (np.array(gender)+2)**2+random.random()

def intilize(gender):
    if gender == '男':
        return [1, 0]
    else:
        return [0, 1]

def init_model():
    net = tflearn.input_data(shape=[None, 128])
    net = tflearn.fully_connected(net, 64, activation='tanh')
    net = tflearn.fully_connected(net, 32, activation='tanh')
    net = tflearn.dropout(net, 0.5)
    net = tflearn.fully_connected(net, 2, activation='softmax')
    net = tflearn.regression(net, optimizer='adam', learning_rate=0.001,
                             metric=None)
    model = tflearn.DNN(net, tensorboard_verbose=0)
    model.load("G:\\T_T\\exe\\facereg\\model\\gender4.model")
    return model

def ismale(model, encoding):
    return model.predict(encoding)
    # return [data[0] > data[1] for data in model.predict(encoding)]

PATH = "G:\\T_T\\exe\\neu_captcha\\pic_2015"
'''
i = 0
X = []
Y = []
X_TEST = []
Y_TEST = []

with open("G:\\T_T\\exe\\neu_captcha\\info_2015.json") as fr:
    stu_info = json.load(fr)

for file in os.listdir(PATH):
    file_path = os.path.join(PATH, file)
    if os.path.getsize(file_path) > 2 << 12:
        img = frec.load_image_file(file_path)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        box = frec.face_locations(img)
        encoding = frec.face_encodings(img, box)[0]
        info = stu_info.get(file.split('.')[0])
        if info:
            gender = intilize(info[2])
        if box:
            # box = box[0]
            # img = img[box[0]:box[2], box[3]:box[1]]
            # img = cv2.resize(img, (32, 32))
            # img = np.reshape(img, 1024)
            # img = list(img)
            # img[-1], img[-2] = ing(gender)
            if i < 1100:
                X.append(encoding)
                Y.append(np.array(gender))
            else:
                X_TEST.append(encoding)
                Y_TEST.append(np.array(gender))
        
        # while True:
        #     cv2.imshow('faceReg', img)
        #     if cv2.waitKey(1) & 0xFF == ord('q'):
        #         break
        # print(img)

    i += 1
    if i > 1600:
        break

X = np.array(X)
Y = np.array(Y)
X_TEST = np.array(X_TEST)
Y_TEST = np.array(Y_TEST)

np.save("G:\\T_T\\exe\\neu_captcha\\X.npy", X)
np.save("G:\\T_T\\exe\\neu_captcha\\Y.npy", Y)
np.save("G:\\T_T\\exe\\neu_captcha\\X_TEST.npy", X_TEST)
np.save("G:\\T_T\\exe\\neu_captcha\\Y_TEST.npy", Y_TEST)
'''
'''
X = np.load("G:\\T_T\\exe\\neu_captcha\\X.npy")
Y = np.load("G:\\T_T\\exe\\neu_captcha\\Y.npy")
X_TEST = np.load("G:\\T_T\\exe\\neu_captcha\\X_TEST.npy")
Y_TEST = np.load("G:\\T_T\\exe\\neu_captcha\\Y_TEST.npy")

# Building the encoder
# encoder = tflearn.input_data(shape=[None, 64, 64, 1])
encoder = tflearn.input_data(shape=[None, 128])
# encoder = tflearn.conv_2d(encoder, 64, 3, activation='relu', regularizer="L2")
# encoder = tflearn.max_pool_2d(encoder, 2)
# encoder = tflearn.local_response_normalization(encoder)
# encoder = tflearn.conv_2d(encoder, 128, 3, activation='relu', regularizer="L2")
# encoder = tflearn.max_pool_2d(encoder, 2)
# encoder = tflearn.local_response_normalization(encoder)
encoder = tflearn.fully_connected(encoder, 64, activation='tanh')
encoder = tflearn.fully_connected(encoder, 32, activation='tanh')
encoder = tflearn.dropout(encoder, 0.8)
encoder = tflearn.fully_connected(encoder, 2, activation='softmax')

# Building the decoder
# decoder = tflearn.fully_connected(encoder, 256)
# decoder = tflearn.fully_connected(encoder, 1024)
# decoder = tflearn.fully_connected(decoder, 4096, activation='sigmoid')

# Regression, with mean square error
net = tflearn.regression(encoder, optimizer='adam', learning_rate=0.001,
                         metric=None)

# Training the auto encoder
model = tflearn.DNN(net, tensorboard_verbose=0)
model.fit(X, Y, n_epoch=15, validation_set=(X_TEST, Y_TEST),
          run_id="auto_encoder", batch_size=64)

model.save("G:\\T_T\\exe\\facereg\\model\\gender4.model")
# Encoding X[0] for test
print("\nTest encoding of X[0]:")
# New model, re-using the same session, for weights sharing
# encoding_model = tflearn.DNN(encoder, session=model.session)
# print(encoding_model.predict([X[0]]))
# decoding_model = tflearn.DNN(decoder, session=model.session)
# print(decoding_model.predict(t))
encode_decode = model.predict(X_TEST)
print([p[1]>p[0] for p in encode_decode].count(True))
print([p[1]>p[0] for p in Y_TEST].count(True))

# # Testing the image reconstruction on new data (test set)
# print("\nVisualizing results after being encoded and decoded:")
# X_TEST = tflearn.data_utils.shuffle(X_TEST)[0]
# # Applying encode and decode over test set
# encode_decode = model.predict(X_TEST)
# # Compare original images with their reconstructions
# f, a = plt.subplots(2, 10, figsize=(10, 2))
# for i in range(10):
#     temp = [[ii, ii, ii] for ii in list(X_TEST[i])]
#     a[0][i].imshow(np.reshape(temp, (28, 28, 3)))
#     temp = [[ii, ii, ii] for ii in list(encode_decode[i])]
#     a[1][i].imshow(np.reshape(temp, (28, 28, 3)))
# f.show()
# plt.draw()
# plt.waitforbuttonpress()
'''

net = init_model()
result = []
wrong = []
nb = 0
with open("G:\\T_T\\exe\\neu_captcha\\info_2016.json") as fr:
    stu_info = json.load(fr)
for file in os.listdir("G:\\T_T\\exe\\neu_captcha\\pic_2016"):
    file_path = os.path.join("G:\\T_T\\exe\\neu_captcha\\pic_2016", file)
    if os.path.getsize(file_path) > 2 << 12:
        img = frec.load_image_file(file_path)
        box = frec.face_locations(img)
        encoding = frec.face_encodings(img, box)
        info = stu_info.get(file.split('.')[0])
        if info and encoding:
            nb += 1
            gender = True if info[2] == '男' else False
            data = ismale(net, encoding)[0]
            gender_predict = data[0] > data[1] 
            bo = gender == gender_predict
            if not bo:
                wrong.append([info[0], info[2], data])
            result.append(bo)
print(nb, result.count(False))
print(wrong)
# img = frec.load_image_file("G:\\T_T\\exe\\neu_captcha\\pic_2016\\tmp9xrl28yl.png")
# box = frec.face_locations(img)
# encoding = frec.face_encodings(img, box)
# print(ismale(net, encoding))
