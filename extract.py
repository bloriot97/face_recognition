import numpy as np
import cv2
import os
from tqdm import tqdm
import pickle
from core import exctract_faces
import imutils

embedder = cv2.dnn.readNetFromTorch("./openface_nn4.small2.v1.t7")

def embed_face(face, embedder):
    faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
				(96, 96), (0, 0, 0), swapRB=True, crop=False)
    embedder.setInput(faceBlob)
    vec = embedder.forward()
    return vec

def increase_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += np.uint8(value)

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img

angles_agmentation = [-20, -10, -5, 0, 5, 10, 20]
brightness_agmentation = [-50, -25, -10, 0, 10, 25, 50]

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
        for img in tqdm(person_images):
            image = imutils.resize(img, width=600)
            if True:
                for angle in angles_agmentation:
                    num_rows, num_cols = image.shape[:2]
                    rotation_matrix = cv2.getRotationMatrix2D((num_cols/2, num_rows/2), angle, 1)
                    image_rotation = cv2.warpAffine(image, rotation_matrix, (num_cols, num_rows))
                    for brightness in brightness_agmentation:
                        image_brightness = imutils.adjust_brightness_contrast(image_rotation, brightness=brightness)
                        faces = exctract_faces(image_brightness)
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

def display_image_grid(img_path):
    face_images = []
    image = cv2.imread(img_path)
    image = imutils.resize(image, width=600)
    output = None
    for angle in angles_agmentation:
        num_rows, num_cols = image.shape[:2]
        rotation_matrix = cv2.getRotationMatrix2D((num_cols/2, num_rows/2), angle, 1)
        image_rotation = cv2.warpAffine(image, rotation_matrix, (num_cols, num_rows))
        line = None 
        for brightness in brightness_agmentation:
            image_brightness = imutils.adjust_brightness_contrast(image_rotation, brightness=brightness)
            faces = exctract_faces(image_brightness)
            if len(faces) == 1:
                face_image = faces[0]["image"]
                face_image = cv2.resize(face_image, (50, 50))
                face_images.append(face_image)
                if line is None:
                    line = face_image
                else:
                    line = np.concatenate((line,face_image),axis=1)
        if output is None:
            output = line
        else:
            output = np.concatenate((output,line),axis=0)
    return output


#gen_embedded_dataset('./perso/dataset')


img_grid = display_image_grid('./perso/photos/benjamin/IMG_20200116_155204.jpg')
cv2.imwrite('augmentation.png', img_grid)
cv2.imshow('img_grid',img_grid)
cv2.waitKey(0)
cv2.destroyAllWindows()