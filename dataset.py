import sys
import cv2
import random
import os
from shutil import copyfile, rmtree

perso_dir = './perso'

photos_dir = os.path.join(perso_dir, 'photos')
dataset_dir = os.path.join(perso_dir, 'dataset')
test_dir = os.path.join(perso_dir, 'test')

def clear_dir(directory):
    if os.path.exists(directory):
        rmtree(directory)
    os.mkdir(directory)

if __name__ == "__main__":
    clear_dir(dataset_dir)
    clear_dir(test_dir)

    train_split = 0.6

    def copy_images(images, target_dir):
        for i in range(len(images)):
            from_file = images[i]
            to_file = os.path.join(target_dir, f"{i}.jpg")
            print(f"{from_file}->{to_file}")
            copyfile(from_file, to_file)

    for name in os.listdir(photos_dir):
        if name != '.DS_Store':
            personal_photos_dir = os.path.join(photos_dir, name)

            personal_dataset_dir = os.path.join(dataset_dir, name)
            personal_test_dir = os.path.join(test_dir, name)

            os.mkdir(personal_dataset_dir)
            os.mkdir(personal_test_dir)

            images = [
                os.path.join(personal_photos_dir, image)
                for image in os.listdir(personal_photos_dir)
                if image != '.DS_Store'
            ]

            random.shuffle(images)

            n_images = len(images)

            dataset_images = images[:int(n_images * train_split)]
            test_images = images[int(n_images * train_split):]

            copy_images(dataset_images, personal_dataset_dir) 
            copy_images(test_images, personal_test_dir) 
