# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import os

def detect_and_predict_mask(frame, faceNet, maskNet):
    # grab the dimensions of the frame and construct a blob
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224), (104.0, 177.0, 123.0))

    # pass the blob through the network to obtain face detections
    faceNet.setInput(blob)
    detections = faceNet.forward()

    # initialize lists for faces, locations, and predictions
    faces = []
    locs = []
    preds = []

    # loop over detections
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # filter weak detections
        if confidence > 0.5:
            # compute bounding box coordinates
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # ensure the box falls within the frame dimensions
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # extract, preprocess the face ROI
            face = frame[startY:endY, startX:endX]
            if face.shape[0] > 0 and face.shape[1] > 0:  # check for valid ROI
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.resize(face, (224, 224))
                face = img_to_array(face)
                face = preprocess_input(face)

                faces.append(face)
                locs.append((startX, startY, endX, endY))

    # make batch predictions if faces are detected
    if len(faces) > 0:
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)

    return (locs, preds)

# paths for the face detector model
base_dir = "/Users/tharindusumanarathna/Desktop/Face Mask Detection/Face-mask-detection"
prototxtPath = os.path.join(base_dir, "face_detector/deploy.prototxt")
weightsPath = os.path.join(base_dir, "face_detector/res10_300x300_ssd_iter_140000.caffemodel")

# check if paths exist
if not os.path.exists(prototxtPath) or not os.path.exists(weightsPath):
    raise FileNotFoundError("Face detection model files not found. Please check the paths.")

# load the face detector model
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model
mask_model_path = os.path.join(base_dir, "/Users/tharindusumanarathna/Desktop/Face Mask Detection/Face-mask-detection/mask_detector.h5")
if not os.path.exists(mask_model_path):
    raise FileNotFoundError("Mask detector model file 'mask_detector.h5' not found. Please check the path.")
maskNet = load_model(mask_model_path)

# initialize the video stream
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)  # allow camera to warm up

try:
    while True:
        # grab and resize the frame
        frame = vs.read()
        frame = imutils.resize(frame, width=400)

        # detect faces and predict mask usage
        (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

        # loop over face locations and predictions
        for (box, pred) in zip(locs, preds):
            (startX, startY, endX, endY) = box
            (mask, withoutMask) = pred

            # determine label and bounding box color
            label = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

            # display label and bounding box
            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
            cv2.putText(frame, label, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

        # show the frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # break the loop on 'q' key press
        if key == ord("q"):
            break

except Exception as e:
    print(f"[ERROR] {e}")

finally:
    # clean up resources
    cv2.destroyAllWindows()
    vs.stop()
