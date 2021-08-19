from neural_network import neural_network
import cv2
import numpy as np
from PIL import Image

model = neural_network(784, 100, 100, 0.2)

name = "Jaskirat"
haar_file = 'haarcascade_frontalface_default.xml'


result = np.zeros((1,100))
result[-1] = 1

for i in range(1,501):

    data = cv2.imread(r"datasets/Jaskirat/{}.png".format(i))
    data = data[:,:,0]

    data = cv2.resize(data, dsize=(28,28))
    data = data.flatten()

    model.train(data, result)







webcam = cv2.VideoCapture(0) 
face_cascade = cv2.CascadeClassifier(haar_file) 



(width, height) = (28, 28)

while True: 
    (_, im) = webcam.read() 
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) 
    faces = face_cascade.detectMultiScale(gray, 1.3, 5) 
    for (x, y, w, h) in faces: 
        cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2) 
        face = gray[y:y + h, x:x + w] 
        face_resize = cv2.resize(face, (width, height))
        face_resize = face_resize.flatten() 
        prediction = model.query(face_resize) 
        cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 3)

        pred_list = prediction.tolist()

        cv2.putText(im, '% s - %.0f' % (name, pred_list.index(max(pred_list))), (x-10, y-10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0)) 

    cv2.imshow('OpenCV', im) 


    key = cv2.waitKey(10) 
    if key == 27: 
        break