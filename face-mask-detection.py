# Face Mask Detection Project
# https://thecleverprogrammer.com/2020/11/17/face-mask-detection-with-machine-learning/

#%%
# 라이브러리 호출
import pandas as pd                 # 데이터 분석, 처리
import numpy as np                  # 다차원배열 처리
import cv2                          # python 영상 처리
import json                         # pthon 객체를 json 데이터로 변환
import os                           # 운영체제 접근
import matplotlib.pyplot as plt     # 그래프 생성 함수
import random                       # 랜덤 관련 함수
import seaborn as sb                # 데이터 분포 시각화

from keras.models import Sequential
from keras import optimizers
from keras import backend as krBackend
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator

annotationsPath = "/workspaces/Face-Mask-Detection/dataset/Medical mask/Medical mask/Medical Mask/annotations"
imagesPath = "/workspaces/Face-Mask-Detection/dataset/Medical mask/Medical mask/Medical Mask/images"

architectirePath = "/workspaces/Face-Mask-Detection/dataset/architecture.txt"
weightPath = "/workspaces/Face-Mask-Detection/dataset/weights.caffemodel"

df = pd.read_csv("/workspaces/Face-Mask-Detection/dataset/train.csv")
df_test = pd.read_csv("/workspaces/Face-Mask-Detection/dataset/submission.csv")

# 필요 함수 생성
cvNet = cv2.dnn.readNetFromCaffe(architectirePath, weightPath)
# cvNet = cv2.dnn.readNetFromCaffe('weights.caffemodel')

def getJSON(filepathName):
    with open(filepathName, 'r') as f:
        return json.load(f)

def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    tabel = np.array([ ((i / 255) ** invGamma) * 255 for i in np.zeros((1400, 1400), dtype=int) ])
    return cv2.LUT(image.astype(np.uint32), tabel.astype(np.uint32))

"""
invGamma = 1.0 / gamma
tabel = np.array([ ((i / 255) ** invGamma) * 255 for i in np.zeros((1400, 1400), dtype=int) ], dtype=int)
print(tabel)
print(tabel.dtype)
print(np.zeros(256, dtype=int))
"""
#%%
# 데이터 처리 / 훈련 JSON데이터 탐색
jsonfiles = []
for i in os.listdir(annotationsPath):
    jsonfiles.append(getJSON(os.path.join(annotationsPath, i)))
jsonfiles[0]

#%%
# 훈련된 데이터를 출력
df = pd.read_csv("/workspaces/Face-Mask-Detection/dataset/train.csv")
df.head()

#%%
data = []
img_size = 124
mask = ['face_with_mask']
non_mask =['face_no_mask']
labels = {'mask':0, 'without mask':1}

for i in df["name"].unique():
    f = i+".json"
    for j in getJSON(os.path.join(annotationsPath, f)).get("Annotations"):
        if j["classname"] in mask:
            x, y, w, h = j["BoundingBox"]
            img = cv2.imread(os.path.join(imagesPath, i), 1)#1 = cv2.IMREAD_COLOR
            img = img[y:h, x:w]
            img = cv2.resize(img, (img_size, img_size))
            data.append([img, labels["mask"]])

        if j["classname"] in non_mask:
            x,y,w,h = j["BoundingBox"]
            img = cv2.imread(os.path.join(imagesPath, i), 1)
            img = img[y:h, x:w]
            img = cv2.resize(img,(img_size, img_size))    
            data.append([img, labels["without mask"]])
random.shuffle(data)

print("lender data :", len(data))

#%%
X = []
Y = []
for features, label in data:
    X.append(features)
    Y.append(label)

X = np.array(X) / 255.0
Y = X.reshape(-1, 124, 124, 3)
Y = np.array(Y)

#%%
# 마스크 감지 신경망 훈련
model = Sequential()

model.add(Conv2D(32, (3, 3), padding = "same", activation='relu', input_shape=(124, 124, 3)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
xTrain, xVal, yTrain, yVal = train_test_split(X, Y, train_size=0.8, rando_state=0)
datagen = ImageDataGenerator(
    featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=False)
datagen.fit(xTrain)

history = model.fit_generator(datagen.flow(xTrain, yTrain, batch_size=32), 
                              step_per_epoch=xTrain.shape[0] // 32, 
                              epoch=5, 
                              verbose=1, 
                              validation_data=(xVal, yVal))

#%%
# 테스트 모델 구현
test_images = ['1114.png','1504.jpg', '0072.jpg','0012.jpg','0353.jpg','1374.jpg']

gamma = 2.0
fig = plt.figure(figsize = (14,14))
rows = 3
cols = 2
axes = []
assign = {'0':'Mask','1':"No Mask"}
for j,im in enumerate(test_images):
    image = cv2.imread(os.path.join(imagesPath,im),1)
    image = adjust_gamma(image, gamma=gamma)
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300,300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    cvNet.setInput(blob)
    detections = cvNet.forward()
    for i in range(0, detections.shape[2]):
        try:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            frame = image[startY:endY, startX:endX]
            confidence = detections[0, 0, i, 2]
            if confidence > 0.2:
                im = cv2.resize(frame,(img_size,img_size))
                im = np.array(im)/255.0
                im = im.reshape(1,124,124,3)
                result = model.predict(im)
                if result>0.5:
                    label_Y = 1
                else:
                    label_Y = 0
                cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
                cv2.putText(image,assign[str(label_Y)] , (startX, startY-10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (36,255,12), 2)
        
        except:pass
    axes.append(fig.add_subplot(rows, cols, j+1))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.show()
# %%
