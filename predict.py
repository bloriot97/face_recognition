import numpy as np
import argparse
import imutils
import pickle
import cv2
import os
from core import exctract_faces

embedder = cv2.dnn.readNetFromTorch("./openface_nn4.small2.v1.t7")

recognizer = pickle.loads(open("./output/recognizer.pickle", "rb").read())
le = pickle.loads(open("./output/le.pickle", "rb").read())


def identify_face(face):
    faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
    embedder.setInput(faceBlob)
    vec = embedder.forward()
 
    # perform classification to recognize the face
    preds = recognizer.predict_proba(vec)[0]
    j = np.argmax(preds)
    proba = preds[j]
    name = le.classes_[j]

    return name, proba

def label_image(frame):
    faces = exctract_faces(frame)
    for face in faces:
        name, proba = identify_face(face["image"])
        frame = cv2.rectangle(frame,(face["startX"],face["startY"]),(face["endX"],face["endY"]),(255,0,0),2)
        y = face["startY"] - 10 if face["startY"] - 10 > 10 else face["startY"] + 10
        cv2.putText(frame, f"{name} ({proba:.2f})", (face["startX"], y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
    return frame

def live_prediction():
    cap = cv2.VideoCapture(0)

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        labeled_frame = label_image(frame)

        cv2.imshow('img',labeled_frame)

        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

live_prediction()