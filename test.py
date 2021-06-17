from keras.models import load_model
import cv2
import numpy as np
import os




# data_dir = "dataset/archive/Images/Images"
# classes = []
# folder = os.listdir(data_dir)
# folder = sorted(folder)
# for folder_name in folder:
#     try:
#         class_name = folder_name.split("_")[2]
#         classes.append(class_name)
#     except:
#         class_name = folder_name.split("_")[1]
#         classes.append(class_name)
# print(classes)

classes = ['ka', 'kha', 'ga', 'gha', 'kna', 'cha', 
    'chha', 'ja', 'jha', 'yna', 'taamatar', 'thaa', 
    'daa', 'dhaa', 'adna', 'tabala', 'tha', 'da', 'dha', 
    'na', 'pa', 'pha', 'ba', 'bha', 'ma', 'yaw', 'ra', 'la',
    'waw', 'motosaw', 'petchiryakha', 'patalosaw', 'ha', 
    'chhya', 'tra', 'gya', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']


model = load_model('model/new_model.h5')
model.summary()
for img in os.listdir('test_image'):
    image = cv2.imread(os.path.join('test_image', img))
    image = cv2.GaussianBlur(image,(3,3),5)
    image = cv2.resize(image,(32,32))
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    _,thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY_INV)

    dilate = cv2.dilate(thresh,(3,3))

    result= model.predict(np.expand_dims(dilate, axis=0))
    print(result)
    index = np.argmax(result[0])
    print(f"index: {index} prob: {result[0][index]}")
    # cv2.imshow('image',image)
    # cv2.imshow('thresh',thresh)
    image = cv2.resize(image, (200, 200))
    cv2.imshow(f'{classes[index]}',image)


    cv2.waitKey(0)
