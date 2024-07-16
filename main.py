import os
import cv2
import xml.etree.ElementTree as ET
import pytesseract

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


xml_path = './archive/annotations/'
img_path = './archive/images/'
dataset = {}

for xml_file in os.listdir(xml_path):
    x = os.path.join(xml_path, xml_file)
    extracted = get_labels(x)
    filename = extracted['filename']
    dataset[filename] = extracted
#print(dataset)

def load_dataset(img_path, xml_path):
    images = []
    boxes = []
    for xml_file in os.listdir(xml_path):
        data = get_labels(os.path.join(xml_path, xml_file))
        image_path = os.path.join(img_path, data['filename'])
        if os.path.exists(image_path):
            image = cv2.imread(image_path)
            images.append(image)
            boxes.append([data['xmin'], data['ymin'], data['xmax'], data['ymax']])
    return images, boxes


from sklearn.model_selection import train_test_split
import numpy as np

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

        box_resized = [box[0] * scale_x,    #xmin
                       box[1] * scale_y,    #ymin
                       box[2] * scale_x,    #xmax
                       box[3] * scale_y]    #ymax
        
        processed_imgs.append(img_normalized)
        processed_boxes.append(box_resized)

    # Change them to numpy array
    np_imgs = np.array(processed_imgs)
    np_boxes = np.array(processed_boxes)
    return np_imgs, np_boxes

# Get all the data from images and xml files
images, boxes = load_dataset(img_path, xml_path)

# Preprocess the images and boxes (resize images and adjust boxes with scale)
imgs_preprocessed, boxes_preprocessed = preprocess_data(images, boxes)

# Splitting the data to train and test sets
X_train, X_test, y_train, y_test = train_test_split(imgs_preprocessed, boxes_preprocessed)





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