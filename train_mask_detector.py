# Import the necessary libraries for building and training the face mask detector
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import os

# Set the initial learning rate, number of training epochs, and batch size
INIT_LR = 1e-4  # Initial learning rate for the Adam optimizer
EPOCHS = 20     # Number of epochs to train the model
BS = 32         # Batch size for training

# Define the dataset directory and the two categories: with_mask and without_mask
DIRECTORY = r"/Users/tharindusumanarathna/Desktop/Face Mask Detection/Face-mask-detection/dataset"
CATEGORIES = ["with_mask", "without_mask"]

# Load images and prepare the dataset
print("[INFO] loading images...")

data = []   # List to store image data
labels = [] # List to store corresponding labels

# Loop through each category directory (with_mask and without_mask)
for category in CATEGORIES:
    path = os.path.join(DIRECTORY, category)
    for img in os.listdir(path):
        # Load each image, resize it to 224x224 pixels, and preprocess it
        img_path = os.path.join(path, img)
        image = load_img(img_path, target_size=(224, 224))  # Resize image
        image = img_to_array(image)                        # Convert image to array
        image = preprocess_input(image)                    # Preprocess for MobileNetV2

        data.append(image)                                 # Add image to the dataset
        labels.append(category)                            # Add corresponding label

# Perform one-hot encoding on the labels
lb = LabelBinarizer()                     # Initialize label binarizer
labels = lb.fit_transform(labels)         # Convert text labels to binary
labels = to_categorical(labels)           # Convert binary to one-hot encoding

# Convert data and labels to NumPy arrays for easier processing
data = np.array(data, dtype="float32")
labels = np.array(labels)

# Split the dataset into training and testing sets (80% train, 20% test)
(trainX, testX, trainY, testY) = train_test_split(
    data, labels, test_size=0.20, stratify=labels, random_state=42
)

# Set up image data augmentation to enhance training performance
aug = ImageDataGenerator(
    rotation_range=20,      # Randomly rotate images by up to 20 degrees
    zoom_range=0.15,        # Randomly zoom images by up to 15%
    width_shift_range=0.2,  # Randomly shift images horizontally by up to 20%
    height_shift_range=0.2, # Randomly shift images vertically by up to 20%
    shear_range=0.15,       # Apply random shearing transformations
    horizontal_flip=True,   # Randomly flip images horizontally
    fill_mode="nearest"     # Fill empty pixels with the nearest value
)

# Load the MobileNetV2 model pre-trained on ImageNet, excluding the top layers
baseModel = MobileNetV2(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))

# Build the custom head of the model for the face mask classification task
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)  # Add an average pooling layer
headModel = Flatten(name="flatten")(headModel)            # Flatten the pooled output
headModel = Dense(128, activation="relu")(headModel)      # Add a dense layer with 128 neurons
headModel = Dropout(0.5)(headModel)                       # Add dropout for regularization
headModel = Dense(2, activation="softmax")(headModel)     # Add a final dense layer for binary classification

# Combine the base model with the custom head to form the complete model
model = Model(inputs=baseModel.input, outputs=headModel)

# Freeze the base model layers to retain pre-trained weights during initial training
for layer in baseModel.layers:
    layer.trainable = False

# Compile the model with the Adam optimizer and binary cross-entropy loss function
print("[INFO] compiling model...")
opt = Adam(learning_rate=INIT_LR)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

# Train the model on the augmented dataset
print("[INFO] training head...")
H = model.fit(
    aug.flow(trainX, trainY, batch_size=BS),
    steps_per_epoch=len(trainX) // BS,
    validation_data=(testX, testY),
    validation_steps=len(testX) // BS,
    epochs=EPOCHS
)

# Evaluate the model on the testing set
print("[INFO] evaluating network...")
predIdxs = model.predict(testX, batch_size=BS)

# Determine the predicted class for each test image
predIdxs = np.argmax(predIdxs, axis=1)

# Generate and display a classification report
print(classification_report(testY.argmax(axis=1), predIdxs, target_names=lb.classes_))

# Save the trained model to disk
print("[INFO] saving mask detector model...")
model.save("mask_detector.h5")

# Plot the training loss and accuracy over epochs
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot.png")
