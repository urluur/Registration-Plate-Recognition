import os
import xml.etree.ElementTree as ET

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

