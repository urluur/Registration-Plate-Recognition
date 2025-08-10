import tensorflow as tf
import math
import os
import numpy as np
import cv2


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, image_paths, labels, batch_size=8, target_size=(500,500)):
        self.image_paths = image_paths
        self.labels = labels
        self.batch_size = batch_size
        self.target_size = target_size
        self.on_epoch_end()
    
    def __len__(self):
        return math.ceil(len(self.image_paths) / self.batch_size)

    def __getitem__(self, index):
        batch_paths = self.image_paths[index*self.batch_size:(index+1)*self.batch_size]
        batch_labels = [self.labels[os.path.basename(p)] for p in batch_paths]

        X = np.empty((len(batch_paths), *self.target_size, 3), dtype=np.float32)
        Y = np.empty((len(batch_paths), 4), dtype=np.float32)

        for i, path in enumerate(batch_paths):
            img = cv2.imread(path)
            if img is None:
                # Handle case where image couldn't be loaded
                print(f"Warning: Could not load image {path}")
                img = np.zeros((*self.target_size, 3), dtype=np.uint8)
            
            img = cv2.resize(img, self.target_size)
            # Normalize and ensure float32
            img = img.astype(np.float32) / 255.0
            X[i] = img
            
            # Get the label dictionary for the current image
            label_info = batch_labels[i]
           
            # Get the original image dimensions for normalization
            orig_h = float(label_info['h'])
            orig_w = float(label_info['w'])
            
            # Construct the normalized bounding box from the individual keys
            # Ensure all operations result in float32
            box = np.array([
                float(label_info['xmin']) / orig_w,
                float(label_info['ymin']) / orig_h,
                float(label_info['xmax']) / orig_w,
                float(label_info['ymax']) / orig_h
            ], dtype=np.float32)
            
            # Clip values to [0, 1] range to prevent invalid coordinates
            box = np.clip(box, 0.0, 1.0)
            
            Y[i] = box

        return X, Y
    
    def on_epoch_end(self):
        pass