import numpy as np
import cv2
import os
from tqdm import tqdm
import pickle
from core import exctract_faces

embedder = cv2.dnn.readNetFromTorch("./openface_nn4.small2.v1.t7")

def embed_face(face, embedder):
    faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
				(96, 96), (0, 0, 0), swapRB=True, crop=False)
    embedder.setInput(faceBlob)
    vec = embedder.forward()
    return vec

def embed_dataset(dataset_dir):
    people = [f for f in os.listdir(dataset_dir) if f != '.DS_Store']
    knownNames = []
    knownEmbeddings = []
    for person_name in tqdm(people):
        person_images_dir = os.path.join(dataset_dir, person_name)
        person_images_path = [
            os.path.join(person_images_dir, file_name) 
            for file_name in os.listdir(person_images_dir) 
            if file_name != '.DS_Store'
        ]
        person_images = [
            cv2.imread(imagePath)
            for imagePath in person_images_path
        ]
        for image in person_images:
            faces = exctract_faces(image)
            if len(faces) == 1:
                face_image = faces[0]["image"]
                vec = embed_face(face_image, embedder)
                knownNames.append(person_name)
                knownEmbeddings.append(vec.flatten())
    return knownNames, knownEmbeddings

def gen_embedded_dataset(dataset_dir):
    knownNames, knownEmbeddings = embed_dataset(dataset_dir)
    print(f"We have been able to extract {len(knownNames)} faces")
    data = {"embeddings": knownEmbeddings, "names": knownNames}
    f = open('./output/embeddings.pickle', "wb")
    f.write(pickle.dumps(data))
    f.close()


gen_embedded_dataset('./perso/dataset')