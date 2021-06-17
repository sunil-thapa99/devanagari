import io
import base64
import re

from flask import Flask, render_template, request, jsonify
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import cv2
import numpy as np
from PIL import Image

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 


MODEL = load_model('model/new_model.h5')
LABELS = [u'\u091E',u'\u091F',u'\u0920',u'\u0921',u'\u0922',u'\u0923',u'\u0924',u'\u0925',u'\u0926',u'\u0927',u'\u0915',u'\u0928',u'\u092A',u'\u092B',u'\u092c',u'\u092d',u'\u092e',u'\u092f',u'\u0930',u'\u0932',u'\u0935',u'\u0916',u'\u0936',u'\u0937',u'\u0938',u'\u0939',u'\u0915\u094D\u0937',u'\u0924\u094D\u0930',u'\u091c\u094D\u091e',u'\u0917',u'\u0918',u'\u0919',u'\u091a',u'\u091b',u'\u091c',u'\u091d',u'\u0966',u'\u0967',u'\u0968',u'\u0969',u'\u096a',u'\u096b',u'\u096c',u'\u096d',u'\u096e',u'\u096f']

app = Flask(__name__)


def convert_image_base64(image):
    detect_img = Image.fromarray(image.astype("uint8"))
    rawBytes = io.BytesIO()
    detect_img.save(rawBytes, "JPEG")
    rawBytes.seek(0)
    img_base64 = base64.b64encode(rawBytes.read())

    return img_base64

def decode_base64(data, altchars=b'+/'):
    """Decode base64, padding being optional.

    :param data: Base64 data as an ASCII byte string
    :returns: The decoded byte string.

    """
    data = re.sub(rb'[^a-zA-Z0-9%s]+' % altchars, b'', data)  # normalize
    missing_padding = len(data) % 4
    if missing_padding:
        data += b'='* (4 - missing_padding)
    return base64.b64decode(data, altchars)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/solve', methods=['POST'])
def solve():
    img_file = request.files['file']
    img = Image.open(img_file.stream)
    # img = base64.decodebytes(img_file)


    # print(request.form, request.get_json(), request.get_data())
    # img_file = request.form.get('file')

    # img_file = request.get_data()
    # img_file = img_file.decode("utf-8") 
    # print(img_file)
    # img = decode_base64(img_file)
    # img = Image.open(io.BytesIO(img))
    # img.show()

    # img_file = img_file.split(',')[1]
    # img_file = base64.b64decode(img_file)
    # img = Image.open(io.BytesIO(img))
    # img = np.array(img)
    # img_file = img_file.split(',')[1]
    # npimg = np.fromstring(img, np.uint8)
    # npimg = np.fromstring(img_file, np.uint8)
    img = np.array(img)
    # img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    img = 255 - img
    kernel = np.ones((3, 3), np.uint8)
    img = cv2.dilate(img,kernel,iterations = 1)
    cv2.imwrite('test.jpg', img)
    # img = cv2.imdecode(np.frombuffer(img, np.uint8), -1)

    # cv2.imshow('Test.png', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    img = cv2.resize(img, (32, 32))
    img = img.astype('float')/255.0
    img = img_to_array(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    cv2.imshow('Test.png', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=3)

    predictions = MODEL.predict(img)
    pred = LABELS[np.argmax(predictions)]
    # pred = 0

    return jsonify({'status': 'Success', 'predictions': pred})



if __name__ == '__main__':
    app.run('0.0.0.0', port=6006, debug=True)
