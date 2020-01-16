import cv2
import os
import numpy as np

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

model_dir = './models'

protoPath = os.path.sep.join([model_dir, "deploy.prototxt"])
modelPath = os.path.sep.join([model_dir, "res10_300x300_ssd_iter_140000.caffemodel"])
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

def exctract_faces(frame, methode="dnn"):
    if methode == "dnn":
        return exctract_faces_dnn(frame)
    else:
        return exctract_faces_harr_cascade(frame)

def exctract_faces_harr_cascade(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    faces_data = []
    for (x,y,w,h) in faces:
        startX, endX = x, x+w
        startY, endY = y, y+h
        face_img = frame[startY:endY, startX:endX]
        faces_data.append({
            "startX": startX,
            "startY": startY,
            "endX": endX,
            "endY": endY,
            "image": face_img
        })
    return faces_data

def exctract_faces_dnn(frame, min_confidence=0.5):
    imageBlob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300),(104.0, 177.0, 123.0), swapRB=False, crop=False)
    detector.setInput(imageBlob)
    detections = detector.forward()
    (h, w) = frame.shape[:2]

    faces_data = []

    for j in range(len(detections)):
        i = np.argmax(detections[j, 0, :, 2])
        confidence = detections[j, 0, i, 2]
 
        if confidence > min_confidence:
            box = detections[j, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            face = frame[startY:endY, startX:endX]
            (fH, fW) = face.shape[:2]

            if fW < 20 or fH < 20:
                continue

            faces_data.append({
                "startX": startX,
                "startY": startY,
                "endX": endX,
                "endY": endY,
                "image": face
            })

    return faces_data