# Face-Mask-Detection ✅
### *Step 1* : Extract face data for training.
### *Step 2* : Train the classifier to classify faces in mask or labels without a mask.
### *Step 3* : Detect faces while testing data using pre-trained face detector.
### *Step 4* : Using the trained classifier, classify the detected faces.

# Error Fix List
**1. ImportError: libGL.so.1: cannot open shared object file: No such file or directory**

#### cv2의 라이브러리 중 일부가 설치가 안되었거나 문제가 생겨 발생한 에러.
    sudo apt-get install --reinstall libgl1-mesa-glx
로 해당 라이브러리 재설치

**2. OpenCV(4.8.0) /io/opencv/modules/dnn/src/caffe/caffe_io.cpp:1126: error: (-2:Unspecified error) FAILED: fs.is_open(). Can't open "weights.caffemodel" in function 'ReadProtoFromTextFile'**
#### 
#%%
p = []
for face in data:
    if(face[1] == 0):
        p.append("Mask")
    else:
        p.append("NoMask")
sb.countplot(p)
