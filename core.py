import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def exctract_faces(frame):
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