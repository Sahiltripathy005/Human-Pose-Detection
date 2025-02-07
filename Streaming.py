import cv2 as cv
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model("pose_model.h5")

BODY_PARTS = { "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
               "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
               "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
               "LEye": 15, "REar": 16, "LEar": 17 }

cap = cv.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    input_frame = cv.resize(frame, (128, 128))
    input_frame = input_frame.astype("float32") / 255.0
    input_frame = np.expand_dims(input_frame, axis=0)

    keypoints = model.predict(input_frame)[0].reshape(-1, 2) * [frame.shape[1], frame.shape[0]]

    for i, point in enumerate(keypoints):
        x, y = int(point[0]), int(point[1])
        cv.circle(frame, (x, y), 4, (0, 255, 0), -1)

    cv.imshow("Pose Estimation", frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
