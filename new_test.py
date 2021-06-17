from keras import models
from keras.models import load_model
import cv2
from keras.preprocessing.image import img_to_array
import numpy as np

'''

    क,ख,ग,घ,ङ,च,छ,ज,झ,ञ,ट,ठ,ड,ढ,ण,त,थ,द,ध,न,प,फ,ब,भ,म,य,र,ल,व,श,ष,स,ह,क्ष,त्र,ज्ञ
'''

MODEL = load_model('model/new_DevaModel.h5')

LABELS = [u'\u091E',u'\u091F',u'\u0920',u'\u0921',u'\u0922',u'\u0923',u'\u0924',u'\u0925',u'\u0926',u'\u0927',u'\u0915',u'\u0928',u'\u092A',u'\u092B',u'\u092c',u'\u092d',u'\u092e',u'\u092f',u'\u0930',u'\u0932',u'\u0935',u'\u0916',u'\u0936',u'\u0937',u'\u0938',u'\u0939','chya','tra','gya',u'\u0917',u'\u0918',u'\u0919',u'\u091a',u'\u091b',u'\u091c',u'\u091d',u'\u0966',u'\u0967',u'\u0968',u'\u0969',u'\u096a',u'\u096b',u'\u096c',u'\u096d',u'\u096e',u'\u096f']

IMG_NAME = 'test_image/ka.png'
image = cv2.imread(IMG_NAME)

# If text is white and bg is white
image = 255 - image

image = cv2.resize(image, (32, 32))
image = image.astype('float')/255.0
image = img_to_array(image)
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

image = np.expand_dims(image, axis=0)
image = np.expand_dims(image, axis=3)
pred_array = MODEL.predict(image)[0]
pred = LABELS[np.argmax(pred_array)]
print(pred)
