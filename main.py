import os
import cv2
import xml.etree.ElementTree as ET
import pytesseract
import numpy as np
from sklearn.model_selection import train_test_split
from utils import get_labels, load_dataset, preprocess_data
import tensorflow as tf
import keras
from keras import models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, BatchNormalization, Input, AveragePooling2D
from tensorflow.keras.optimizers import Adam
from keras_cv.losses import IoULoss
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


xml_path = './archive/annotations/'
img_path = './archive/images/'
dataset = {}

#Loop over all the xml files and get the data
for xml_file in os.listdir(xml_path):
    x = os.path.join(xml_path, xml_file)
    extracted = get_labels(x)
    filename = extracted['filename']
    dataset[filename] = extracted

# Get all the data from images and xml files
images, boxes = load_dataset(img_path, xml_path)

# Preprocess the images and boxes (resize images and adjust boxes with scale)
imgs_preprocessed, boxes_preprocessed = preprocess_data(images, boxes)

# Splitting the data to train and test sets
X_train, X_test, y_train, y_test = train_test_split(imgs_preprocessed, boxes_preprocessed, test_size=0.15)

def build_custom_model(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        Conv2D(32, (3,3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(3,3),

        Conv2D(64, (3,3), activation='relu', padding='same'),
        BatchNormalization(),
        AveragePooling2D(16),

        Flatten(),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(8, activation='relu'),
        #Dropout(0.5),
        Dense(4, activation='sigmoid')
    ])
    optimizer = Adam(learning_rate=0.01)  # Adjusted learning rate
    model.compile(optimizer='adam', loss='mean_squared_error')
    ##model.compile(optimizer=optimizer, loss=IoULoss(bounding_box_format='xyxy', mode='linear'))
    return model

model = build_custom_model(input_shape=(500,500,3))

# Got idea for callbacks from chatgpt
callbacks = [
    ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, min_lr=1e-5),
    EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)
]

history = model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test), batch_size=4, callbacks=callbacks)

def load_and_predict(image_path):
    image = cv2.imread(image_path)
    og_image = image.copy()

    image = cv2.resize(image, (500,500))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)

    predicted = model.predict(image)[0]

    original_h, original_w = og_image.shape[:2]

    xmin, ymin, xmax, ymax = predicted
    xmin, xmax = int(xmin * original_w), int(xmax * original_w)
    ymin, ymax = int(ymin * original_h), int(ymax * original_h)

    #Draws a rectangle over the detected registration plate
    cv2.rectangle(og_image, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)

    # Get some margin, transform it to gray and read the text
    img2 = og_image[ymin-10:ymax+10, xmin-10:xmax+10]
    grey_img = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(grey_img)
    print(f"Plate detected: {text}")
    # Display the text top right of the rectangle
    cv2.putText(og_image, text, (xmin-5, ymin-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    cv2.imshow("Detected plate " + text, og_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


load_and_predict('./img/testImg.jpg')