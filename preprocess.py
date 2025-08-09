import os
import shutil
import glob
from PIL import Image
import xml.etree.ElementTree as ET

def create_voc_element(filename, width, height, xmin, ymin, xmax, ymax):
    root = ET.Element("annotation")
    ET.SubElement(root, "filename").text = filename
    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = str(width)
    ET.SubElement(size, "height").text = str(height)
    ET.SubElement(size, "depth").text = "3"
    obj = ET.SubElement(root, "object")
    ET.SubElement(obj, "name").text = "license_plate"
    bndbox = ET.SubElement(obj, "bndbox")
    ET.SubElement(bndbox, "xmin").text = str(xmin)
    ET.SubElement(bndbox, "ymin").text = str(ymin)
    ET.SubElement(bndbox, "xmax").text = str(xmax)
    ET.SubElement(bndbox, "ymax").text = str(ymax)
    return root

def convert_ccpd_to_voc(ccpd_dir, output_dir):
    # Make folders
    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'annotations'), exist_ok=True)

    #Read and split
    train_files = []
    val_files = []
    with open(os.path.join(ccpd_dir, 'splits/train.txt'), 'r') as f:
        train_files = [os.path.basename(line.strip()) for line in f]
    with open(os.path.join(ccpd_dir, 'splits/val.txt'), 'r') as f:
        val_files = [os.path.basename(line.strip()) for line in f]

    # Process 
    # IMPORTANT: FIRST ONLY 1500 FILES FOR TESTING, DATASET HAS 250000 images (250k)
    image_files = glob.glob(os.path.join(ccpd_dir, 'ccpd_base', '*.jpg'))[:1500]
    for img_path in image_files:
        filename = os.path.basename(img_path)
        parts = filename.split('-')
        if len(parts) < 3:
            continue #skip files that have wrong name
        bbox_part = parts[2]
        try:
            xmin, ymin = map(int, bbox_part.split('_')[0].split('&'))
            xmax, ymax = map(int, bbox_part.split('_')[1].split('&'))
        except ValueError:
            continue # Skip parsing if something wrong
        
        # Load image to get the dimentions
        try:
            with Image.open(img_path) as img:
                width, height = img.size
        except Exception:
            continue # Skip it if it goes wrong

        #create xml 
        xml_tree = create_voc_element(filename, width, height, xmin, ymin, xmax, ymax)
        xml_filename = filename.replace('.jpg', '.xml')
        xml_path = os.path.join(output_dir, 'annotations', xml_filename)

        tree= ET.ElementTree(xml_tree)
        tree.write(xml_path, encoding='utf-8', xml_declaration=True)
        
        shutil.copy(img_path, os.path.join(output_dir, 'images', filename))
    
    # create train and val txt files
    with open(os.path.join(output_dir, 'train.txt'), 'w') as f:
        for fname in train_files:
            if os.path.exists(os.path.join(output_dir, 'annotations', fname.replace('.jpg', '.xml'))):
                f.write(f"images/{fname}")
    
    with open(os.path.join(output_dir, 'val.txt'), 'w') as f:
        for fname in val_files:
            if os.path.exists(os.path.join(output_dir, 'annotations', fname.replace('.jpg', '.xml'))):
                f.write(f"images/{fname}")



if __name__ == "__main__":
    ccpd_dir = "./archive/CCPD2019"
    output_dir = "./archive/CCPD2019_converted"
    convert_ccpd_to_voc(ccpd_dir, output_dir)