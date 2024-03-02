import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import os

# Path to dataset
dataset_path = "output"

# Preprocessing the images
# This will scale pixel values to the range [0, 1]
datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,  # Rotates the image within 20 degrees range
    width_shift_range=0.2,  # Shifts the image horizontally by 20%
    height_shift_range=0.2,  # Shifts the image vertically by 20%
    shear_range=0.2,  # Shear angle in counter-clockwise direction
    zoom_range=0.2,  # Zoom in by 20% max
    horizontal_flip=True,  # Allows horizontal flipping
    fill_mode="nearest",  # Strategy to fill newly created pixels
    validation_split=0.2,  # Splits dataset into training and validation sets
)

train_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(200, 200),
    batch_size=32,
    class_mode="binary",
    subset="training",
    samples=1000,
)

validation_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(200, 200),
    batch_size=32,
    class_mode="binary",
    subset="validation",
)
# Prepare data generators
train_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(200, 200),  # Resize images to 200x200
    batch_size=32,
    class_mode="categorical",  # Assuming multi-class classification
    subset="training",
)

validation_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(200, 200),
    batch_size=32,
    class_mode="categorical",
    subset="validation",
)

# Define a simple CNN model
model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Conv2D(
            32, (3, 3), activation="relu", input_shape=(200, 200, 3)
        ),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(train_generator.num_classes, activation="softmax"),
    ]
)

# Compile the model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Train the model
model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    epochs=5,
)

# After training
model.save("cnn_model_v2.h5")


# import cv2
# import os
# import numpy as np
# from skimage.feature import hog
# from sklearn.model_selection import train_test_split
# from sklearn.svm import SVC
# from sklearn.metrics import classification_report
# from sklearn.externals import joblib


# def load_images_and_extract_features(directory):
#     labels = []
#     features = []
#     valid_extensions = {
#         ".jpg",
#         ".jpeg",
#         ".png",
#         ".bmp",
#     }  # Add other extensions if needed

#     print(f"Processing directory: {directory}")  # Diagnostic print

#     for label in os.listdir(directory):
#         label_dir = os.path.join(directory, label)
#         if os.path.isdir(label_dir):
#             print(f"Processing label directory: {label_dir}")  # Diagnostic print

#             for sub_label in os.listdir(label_dir):
#                 sub_label_dir = os.path.join(label_dir, sub_label)
#                 if os.path.isdir(sub_label_dir):
#                     print(
#                         f"Processing sub-label directory: {sub_label_dir}"
#                     )  # Diagnostic print

#                     for image_file in os.listdir(sub_label_dir):
#                         if not any(
#                             image_file.lower().endswith(ext) for ext in valid_extensions
#                         ):
#                             print(
#                                 f"Skipping non-image file: {image_file}"
#                             )  # Diagnostic print
#                             continue

#                         image_path = os.path.join(sub_label_dir, image_file)
#                         image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

#                         if image is None:
#                             print(f"Failed to load image: {image_path}")
#                             continue

#                         resized_image = cv2.resize(image, (128, 128))
#                         hog_features = hog(
#                             resized_image,
#                             pixels_per_cell=(8, 8),
#                             cells_per_block=(2, 2),
#                         )
#                         features.append(hog_features)
#                         labels.append(label)  # Using the main label for classification

#     if not features:
#         print("No features extracted. Please check your dataset.")  # Diagnostic print

#     return np.array(features), np.array(labels)


# # Load your dataset
# dataset_path = "/Users/yskakshiyap/Desktop/ai-motion-tracker/output"
# features, labels = load_images_and_extract_features(dataset_path)

# # Split the dataset into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(
#     features, labels, test_size=0.2, random_state=42
# )

# # Train a Support Vector Machine
# svm_model = SVC(kernel="linear")
# svm_model.fit(X_train, y_train)

# # Evaluate the model
# y_pred = svm_model.predict(X_test)
# print(classification_report(y_test, y_pred))

# # Save your trained SVM model
# joblib.dump(svm_model, "svm_model.pkl")
