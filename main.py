import os
import cv2
import xml.etree.ElementTree as ET
import pytesseract
import numpy as np
from sklearn.model_selection import train_test_split
from utils import get_labels, load_dataset, preprocess_data
import tensorflow
import keras
from keras import models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D, Conv2D, MaxPooling2D, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
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

#print(X_test)

def build_custom_model(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        Conv2D(32, (3,3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(2,2),
        Conv2D(64, (3,3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(2,2),
        Conv2D(128, (3,3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(2,2),
        Conv2D(256, (3,3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(2,2),
        Conv2D(512, (3,3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(2,2),
        Conv2D(256, (3,3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(2,2),
        Conv2D(128, (3,3), activation='relu', padding='same'),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(1024, activation='relu'),
        Dropout(0.5),
        Dense(4, activation='sigmoid')
    ])
    optimizer = Adam(learning_rate=0.001)  # Adjusted learning rate
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model

#input_shape = imgs_preprocessed[0].shape
input_shape = (224,224,3)
model = build_custom_model(input_shape=(224,224,3))

# Got idea for callbacks from chatgpt
callbacks = [
    ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-6),
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
]

history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), batch_size=32, callbacks=callbacks)

#predictions = model.predict(X_test)

#print(predictions)

def load_and_predict(image_path):
    image = cv2.imread(image_path)
    og_image = image.copy()

    image = cv2.resize(image, (224,224))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)

    predicted = model.predict(image)[0]

    original_h, original_w = og_image.shape[:2]
    scale_x, scale_y = original_w / 224, original_h / 224

    xmin, ymin, xmax, ymax = predicted
    xmin, xmax = int(xmin * 224), int(xmax *224)
    ymin, ymax = int(ymin * 224), int(ymax * 224)
    print("Xmin: " + str(xmin) + " | Xmax: "  + str(xmax) + " | Ymin: " + str(ymin) + " | Ymax: " + str(ymax))

    cv2.rectangle(og_image, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)

    cv2.imshow("Predicted", og_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


load_and_predict('./testImg.jpg')



### /////////////////////////////////////////////////////////
### DRAWING RECTANGLE ON THE IMAGE - for now not relavant
### /////////////////////////////////////////////////////////

# image_path = './archive/images/Cars175.png'
# image = cv2.imread(image_path)

# # Read from the dictionary
# extracted = dataset['Cars175.png']

# # Get the values needed to draw a rectangle
# x = extracted['xmin']
# x2 = extracted['xmax']
# y = extracted['ymin']
# y2 = extracted['ymax']
# # Draw a rectangle 
# cv2.rectangle(image, (x,y), (x2, y2), (255, 0 , 0), 2)

# # Cuts the image, makes it black and white and tries to read the text
# img2 = image[y-5:y2+5, x-5:x2+5]
# grey_img = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
# text = pytesseract.image_to_string(grey_img)

# # Draw a text next to the recvtangle including registration
# cv2.putText(image, text, (x-5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

# # Displays the image in a new window
# cv2.imshow('Test', image);
# cv2.waitKey(0)
# cv2.destroyAllWindows()