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

print(dataset)

### DRAWING RECTANGLE ON THE IMAGE

image_path = './archive/images/Cars17.png'
image = cv2.imread(image_path)

# Read from the dictionary
extracted = dataset['Cars17.png']

# Get the values needed to draw a rectangle
x = extracted['xmin']
x2 = extracted['xmax']
y = extracted['ymin']
y2 = extracted['ymax']
# Draw a rectangle 
cv2.rectangle(image, (x,y), (x2, y2), (255, 0 , 0), 2)

# Cuts the image and tries to read the text
img2 = image[y:y2, x:x2]
text = pytesseract.image_to_string(img2)

# Draw a text next to the recvtangle including registration
cv2.putText(image, text, (x-5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

# Displays the image in a new window
cv2.imshow('Test', image);
cv2.waitKey(0)
cv2.destroyAllWindows()