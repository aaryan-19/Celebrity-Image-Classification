
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import cv2
import os
import json
import joblib
import shutil
%matplotlib inline

face_cascade = cv2.CascadeClassifier('./opencv/haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('./opencv/haarcascades/haarcascade_eye.xml')

def get_cropped_images_if_2_eyes(imagePath):
    img=cv2.imread(imagePath)
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY);
    face=face_cascade.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in face:
        roi_gray=gray[y:y+h,x:x+h]
        roi_color=img[y:y+h,x:x+h]
        eyes=eye_cascade.detectMultiScale(roi_gray)
        if(len(eyes)>=2):
            return roi_color


image_dirs=[]
path_data="./dataset/"
path_to_face="./dataset/cropped/"
for entry in os.scandir(path_data):
    if(entry.is_dir() and entry.path!="./dataset/cropped"):
        image_dirs.append(entry.path)

if (os.path.exists(path_to_face)==False):
    os.mkdir(path_to_face)

cropped_image_dir=[]
celebrity_file_names={}
for imgD in image_dirs:
    #going throigh each celeb web mined pics
    count=0
    #getting name
    celebrity_name=imgD.split('/')[-1]
    celebrity_file_names[celebrity_name]=[]
    temp_path=path_to_face+celebrity_name
    if not os.path.exists(temp_path):
        #making folder to stored cleaned images
        os.mkdir(temp_path)
        ## to cropped_image_dir
        cropped_image_dir.append(temp_path) ##change location in future if needed outside if block
    for entry in os.scandir(imgD):
        #iterating through all images in original
        roi_color=get_cropped_images_if_2_eyes(entry.path)
        if(roi_color is not None):
            #if image meets cleaning parameter store to folder
            cropped_file_name=celebrity_name+str(count)+".png"
            cropped_file_path=temp_path+"/"+cropped_file_name
            cv2.imwrite(cropped_file_path,roi_color)
            celebrity_file_names[celebrity_name].append(cropped_file_path)
            count=count+1
    
    
celebrity_file_names.keys()
celeb_map={}
count=0;
for i in celebrity_file_names.keys():
    celeb_map[i]=count;
    count=count+1;


import numpy as np
import pywt
import cv2    
## doing haar transform to do feature extraction

def w2d(img, mode='haar', level=1):
    imArray = img
    #Datatype conversions
    #convert to grayscale
    imArray = cv2.cvtColor( imArray,cv2.COLOR_RGB2GRAY )
    #convert to float
    imArray =  np.float32(imArray)   
    imArray /= 255;
    # compute coefficients 
    coeffs=pywt.wavedec2(imArray, mode, level=level)

    #Process Coefficients
    coeffs_H=list(coeffs)  
    coeffs_H[0] *= 0;  

    # reconstruction
    imArray_H=pywt.waverec2(coeffs_H, mode);
    imArray_H *= 255;
    imArray_H =  np.uint8(imArray_H)

    return imArray_H

X, y = [], []
#creating dataset by appending images vertical with har transofromed images with 
for celebrity_name, training_files in celebrity_file_names.items():
    for training_image in training_files:
        img = cv2.imread(training_image)
        if(img is None):
            continue
        scalled_raw_img = cv2.resize(img, (32, 32))
        img_har = w2d(img,'db1',5)
        scalled_img_har = cv2.resize(img_har, (32, 32))
        combined_img = np.vstack((scalled_raw_img.reshape(32*32*3,1),scalled_img_har.reshape(32*32,1)))
        X.append(combined_img)
        y.append(celeb_map[celebrity_name])     

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

X_train, X_test,y_train, y_test = train_test_split(X,y ,random_state=104,test_size=0.25,shuffle=True)

pipe=Pipeline([('Scale',StandardScaler()),("svc",SVC(kernel="rbf",C=100, probability=True)) ])
pipe.fit(X_train, y_train)
pipe.score(X_test, y_test)
print(classification_report(y_test, pipe.predict(X_test)))

joblib.dump(pipe,"saved_svm_piped_model.pkl")

with open("celeb_mapping.json","w") as f:
    f.write(json.dumps(celeb_map))

gg={
    "lionel_messi": ["https://twitter.com/imessi","https://www.instagram.com/leomessi/"],
    "maria_sharapova": ["https://twitter.com/MariaSharapova","https://www.instagram.com/mariasharapova/?hl=en"],
    "roger_federer": ["https://twitter.com/rogerfederer","https://www.instagram.com/rogerfederer/"],
    "serena_williams": ["https://twitter.com/serenawilliams","https://www.instagram.com/stories/highlights/17916329126375618/"],
    "virat_kohli":["https://twitter.com/imVkohli","https://www.instagram.com/virat.kohli/?hl=en"],
    "Cristiano_Ronaldo":["https://twitter.com/Cristiano","https://www.instagram.com/cristiano/?hl=en"]
    
}

with open("D:\\Vishesh\\Desktop\\study\\WEB MINING\\project\\server\\artefacts\\celeb_handles.json","w") as f:
    f.write(json.dumps(gg))