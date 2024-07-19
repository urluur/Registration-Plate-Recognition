import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from utils import get_labels, load_dataset, preprocess_data
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Paths to dataset
xml_path = './archive/annotations/'
img_path = './archive/images/'

# Get all the data from images and xml files
images, boxes = load_dataset(img_path, xml_path)

# Preprocess the images and boxes (resize images and adjust boxes with scale)
imgs_preprocessed, boxes_preprocessed = preprocess_data(images, boxes)

# Splitting the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(imgs_preprocessed, boxes_preprocessed, test_size=0.15)

def build_model(input_shape):
    base_model = MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet')
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    output = Dense(5, activation='sigmoid')(x)  # 4 for bounding box coordinates + 1 for confidence score
    model = Model(inputs=base_model.input, outputs=output)
    for layer in base_model.layers:
        layer.trainable = False
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model

input_shape = (224, 224, 3)
model = build_model(input_shape)

callbacks = [
    ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-6),
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
]

history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), batch_size=32, callbacks=callbacks)

predictions = model.predict(X_test)
print(predictions)

# Real-time video stream integration
capture = cv2.VideoCapture(1) # for me 1=webcam, 0=external camera

if not capture.isOpened():
    print("Error: Could not open webcam.")
    exit()

confidence_threshold = 0.5  # Set your confidence threshold here

while True:
    ret, frame = capture.read()
    if not ret:
        break
    
    # Preprocess the frame
    frame_resized = cv2.resize(frame, (224, 224))
    frame_normalized = frame_resized / 255.0
    frame_expanded = np.expand_dims(frame_normalized, axis=0)

    # Predict bounding box and confidence score
    pred = model.predict(frame_expanded)[0]
    pred_box = pred[:4]  # Bounding box coordinates
    confidence = pred[4]  # Confidence score

    if confidence > confidence_threshold:
        h, w, _ = frame.shape
        xmin = int(pred_box[0] * w)
        ymin = int(pred_box[1] * h)
        xmax = int(pred_box[2] * w)
        ymax = int(pred_box[3] * h)

        # Draw bounding box on the frame
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
    
    cv2.imshow('License Plate Detection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
