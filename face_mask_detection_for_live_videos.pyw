from tensorflow import keras
import numpy as np
import cv2

# Loading the saved model
model = keras.models.load_model(r'face_mask detection model\face_mask')
model.load_weights(r'face_mask detection model\face_mask_w')
model.compile(optimizer='adam', loss=keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])

source = cv2.VideoCapture(0)

clf = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

a = {
    0: 'Mask is weared incorrectly',
    1: 'Mask',
    2: 'No Mask'
}

# Detecting the face_mask from live video
while True:
    ret, img = source.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = clf.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for face in faces:
        x, y, w, h = face
        image = img[y:y+h, x:x+w]

        # Resizing the image into the shape that is compatible with our model
        image = cv2.resize(image, (224, 224))

        # Predicting the image using the model
        prediction = model.predict(image.reshape(1, 224, 224, 3))
        pred = np.argmax(prediction)

        # Drawing a Green rectangle for Mask, Red rectangle for No Mask and Blue rectangle for Mask is weared incorrectly
        if a[pred] == 'Mask':
            cv2.rectangle(img, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=1)
        elif a[pred] == 'No Mask':
            cv2.rectangle(img, (x, y), (x + w, y + h), color=(0, 0, 255), thickness=1)
        else:
            cv2.rectangle(img, (x, y), (x + w, y + h), color=(255, 0, 0), thickness=1)

        cv2.putText(img, a[pred], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow('LIVE', img)
    key = cv2.waitKey(1)
    if key == 27:
        break

source.release()
cv2.destroyAllWindows()

