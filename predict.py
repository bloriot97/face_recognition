import numpy as np
import argparse
import imutils
import pickle
import cv2
import os
from core import exctract_faces
from dataset import test_dir
from sklearn.metrics import confusion_matrix, accuracy_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

    return name, proba, j

def label_image(frame):
    faces = exctract_faces(frame)
    predictions = []
    for face in faces:
        name, proba, j = identify_face(face["image"])
        predictions.append((name, proba, j))
        frame = cv2.rectangle(frame,(face["startX"],face["startY"]),(face["endX"],face["endY"]),(255,0,0),2)
        y = face["startY"] - 10 if face["startY"] - 10 > 10 else face["startY"] + 10
        cv2.putText(frame, f"{name} ({proba:.2f})", (face["startX"], y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
    return frame,predictions

def live_prediction():
    cap = cv2.VideoCapture(0)

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        labeled_frame, _ = label_image(frame)

        cv2.imshow('img',labeled_frame)

        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def load_and_label(img_path):
    frame = cv2.imread(img_path)
    labeled_frame = label_image(frame)
    cv2.imshow('labeled_frame',labeled_frame)
    cv2.waitKey(0)

def eval(test_dir):
    real = []
    pred = []

    for name in os.listdir(test_dir):
        if name != '.DS_Store':
            folder = os.path.join(test_dir, name)
            for image_name in os.listdir(folder):
                if image_name != '.DS_Store':
                    image_path = os.path.join(folder, image_name)
                    image = cv2.imread(image_path)
                    _, predictions = label_image(image)
                    if len(predictions) == 1:
                        #print(f"{name} : {predictions[0][0]}({predictions[0][1]})")
                        real.append(name)
                        pred.append(predictions[0][0])
                    else:
                        pass
                        #print("")
    accuracy=  accuracy_score(real, pred)
    print(f"accuracy : {accuracy}")
    conf = confusion_matrix(real, pred, labels=le.classes_) 
    conf_df = pd.DataFrame(data=conf, index=le.classes_, columns=le.classes_)
    ax = sns.heatmap(conf_df, annot=True, fmt="d")
    plt.show()

live_prediction()
#eval(test_dir)

#load_and_label('./images/trisha_adrian.jpg')
