import xml.etree.ElementTree as ET
import os
import cv2
import numpy as np

# Function for extracting data from xml file
def get_labels(xml_file):
    t = ET.parse(xml_file) #Build xml tree
    r = t.getroot() # Get the root element
    labels = {}

    # Get the filename of an image
    filename = r.find('filename')
    labels['filename'] = filename.text

    # Find width and height labels from inside the size label
    size = r.find('size')
    width = size.find('width')
    height = size.find('height')

    labels['w'] = int(width.text)
    labels['h'] = int(height.text)

    # For each plate in the picture get plate locations
    for o in r.findall('object'):
        bndbox = o.find('bndbox')
        xmin = bndbox.find('xmin')
        ymin = bndbox.find('ymin')
        xmax = bndbox.find('xmax')
        ymax = bndbox.find('ymax')
        labels['xmin'] = int(xmin.text)
        labels['ymin'] = int(ymin.text)
        labels['xmax'] = int(xmax.text)
        labels['ymax'] = int(ymax.text)
    return labels

# Function to load the dataset from the specified path and 
def load_dataset(img_path, xml_path):
    images = []
    boxes = []
    for xml_file in os.listdir(xml_path):
        if xml_file.endswith('.xml'):
            data = get_labels(os.path.join(xml_path, xml_file))
            image_path = os.path.join(img_path, data['filename'])
            if os.path.exists(image_path):
                image = cv2.imread(image_path)
                images.append(image)
                boxes.append([data['xmin'], data['ymin'], data['xmax'], data['ymax']])
    return images, boxes



# Function that resizes the images and adjusts the bounding boxes relative to resize
def preprocess_data(images, boxes, target_size=(224, 224)):
    processed_imgs = []
    processed_boxes = []

    for img, box in zip(images, boxes):
        # Resizing the image
        h, w = img.shape[:2]
        img_resized = cv2.resize(img, target_size)

        # Normalization of the image
        img_normalized = img_resized / 255.0

        # Adjust the box since the image size was changed
        scale_x = target_size[0] / w
        scale_y = target_size[1] / h

        box_resized = [box[0] * scale_x / target_size[0],    #xmin
                       box[1] * scale_y / target_size[1],    #ymin
                       box[2] * scale_x / target_size[0],    #xmax
                       box[3] * scale_y/ target_size[1]]    #ymax
        
        processed_imgs.append(img_normalized)
        processed_boxes.append(box_resized)

    # Change them to numpy array
    np_imgs = np.array(processed_imgs, dtype=np.float32)
    np_boxes = np.array(processed_boxes, dtype=np.float32)
    return np_imgs, np_boxes