import cv2
import numpy as np
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model('my_model.h5')

# Start video capture
cap = cv2.VideoCapture(0)

# Load the Haar Cascade for face detection
cascade_path = r"C:\Users\Admin\Downloads\haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)

while True:
    ret, frame = cap.read()
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        face_roi = frame[y:y+h, x:x+w]
        face_roi = cv2.resize(face_roi, (128, 128))  # Resize to the expected input size of the model
        face_roi = face_roi / 255.0  # Normalize pixel values if needed
        face_roi = np.expand_dims(face_roi, axis=0)  # Add the batch dimension

        prediction = model.predict(face_roi)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        threshold = 0.4  
        if prediction > threshold:
            cv2.putText(frame, 'Mask', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        else:
            cv2.putText(frame, 'No Mask', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()