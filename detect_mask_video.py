# Import necessary libraries and modules
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
    # Get the frame dimensions and create a blob for processing
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224), (104.0, 177.0, 123.0))

    # Feed the blob into the network to detect faces
    faceNet.setInput(blob)
    detections = faceNet.forward()

    # Initialize lists to store face data, their coordinates, and predictions
    faces = []
    locs = []
    preds = []

    # Iterate over detected faces
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # Ignore detections below the confidence threshold
        if confidence > 0.5:
            # Calculate bounding box coordinates for the face
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Ensure the bounding box fits within the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # Extract and preprocess the face region of interest (ROI)
            face = frame[startY:endY, startX:endX]
            if face.shape[0] > 0 and face.shape[1] > 0:  # Validate the ROI
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.resize(face, (224, 224))
                face = img_to_array(face)
                face = preprocess_input(face)

                faces.append(face)
                locs.append((startX, startY, endX, endY))

    # Perform mask predictions if any faces are detected
    if len(faces) > 0:
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)

    return (locs, preds)

# Define paths for the face detection model
base_dir = "/Users/tharindusumanarathna/Desktop/Face Mask Detection/Face-mask-detection"
prototxtPath = os.path.join(base_dir, "face_detector/deploy.prototxt")
weightsPath = os.path.join(base_dir, "face_detector/res10_300x300_ssd_iter_140000.caffemodel")

# Validate the existence of model files
if not os.path.exists(prototxtPath) or not os.path.exists(weightsPath):
    raise FileNotFoundError("Face detection model files not found. Please verify the paths.")

# Load the pre-trained face detection model
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# Load the face mask detection model
mask_model_path = os.path.join(base_dir, "/Users/tharindusumanarathna/Desktop/Face Mask Detection/Face-mask-detection/mask_detector.h5")
if not os.path.exists(mask_model_path):
    raise FileNotFoundError("Mask detector model file 'mask_detector.h5' not found. Please verify the path.")
maskNet = load_model(mask_model_path)

# Initialize the webcam video stream
print("[INFO] Starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)  # Allow the camera to initialize

try:
    while True:
        # Capture a frame and resize it
        frame = vs.read()
        frame = imutils.resize(frame, width=400)

        # Detect faces and predict mask usage for each
        (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

        # Loop through detected faces and predictions
        for (box, pred) in zip(locs, preds):
            (startX, startY, endX, endY) = box
            (mask, withoutMask) = pred

            # Set label and bounding box color based on the prediction
            label = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

            # Display the label and bounding box on the frame
            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
            cv2.putText(frame, label, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

        # Show the processed video stream
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # Exit the loop if 'q' is pressed
        if key == ord("q"):
            break

except Exception as e:
    print(f"[ERROR] {e}")

finally:
    # Release resources and close windows
    cv2.destroyAllWindows()
    vs.stop()
