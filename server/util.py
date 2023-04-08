import joblib
import json
import numpy as np
import base64
import cv2
from wavelet import w2d
_class_number_to_name={}
class_name_to_number={}
_model=None

def get_cv2_image_from_base64_string(b64str):
    '''
    credit: https://stackoverflow.com/questions/33754935/read-a-base-64-encoded-image-from-memory-using-opencv-python-library
    :param uri:
    :return:
    '''
    load_artefacts();
    # print(b64str[2:])
    # encoded_data = b64str.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(b64str), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    # print(img)
    return img

def image_classify(b64str):

    img=get_cv2_image_from_base64_string(b64str)
    faces=cropped_img_if_2_eyes(img)
    # print("hello")
    result=[]
    # print(faces)
    for face in faces:
        scalled_raw_img = cv2.resize(face, (32, 32))
        img_har = w2d(img, 'db1', 5)
        scalled_img_har = cv2.resize(img_har, (32, 32))
        combined_img = np.vstack((scalled_raw_img.reshape(32 * 32 * 3, 1), scalled_img_har.reshape(32 * 32, 1)))

        len_image_array = 32 * 32 * 3 + 32 * 32

        final = combined_img.reshape(1, len_image_array).astype(float)
        # print(final)
        if _model is not None:
            # print(_model.predict_proba(final))
            celeb_no= _model.predict(final)[0]
            probab_dict=np.around(_model.predict_proba(final)*100,3).tolist()[0]
            result.append({
                'celeb': class_number_to_name(celeb_no),
                 'probability': probab_dict[celeb_no]
            })
        # print(result)
        return result

def class_number_to_name(class_num):
    return _class_number_to_name[class_num]

def load_artefacts():
    print("loading saved artifacts...start")
    global class_name_to_number
    global _class_number_to_name

    with open("./artefacts/celeb_mapping.json", "r") as f:
        class_name_to_number = json.load(f)
        _class_number_to_name = {v: k for k, v in class_name_to_number.items()}
    global _model
    if _model is None:
        with open("artefacts/saved_svm_piped_model.pkl","rb") as f:
            _model= joblib.load(f)


def cropped_img_if_2_eyes(img):
    face_cascade = cv2.CascadeClassifier('../opencv/haarcascades/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('../opencv/haarcascades/haarcascade_eye.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    cropped_faces = []
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        if len(eyes) >= 2:
            cropped_faces.append(roi_color)
    return cropped_faces


def get_b64_test_image_for_virat():
    with open("b64.txt") as f:
        return f.read()

if __name__=='__main__':
    load_artefacts()
    print(image_classify(get_b64_test_image_for_virat()))
