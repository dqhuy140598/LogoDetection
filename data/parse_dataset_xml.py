import xml.etree.ElementTree as ET
import os
import sys
import argparse

working_dir = os.getcwd()
sys.path.append(working_dir)
from config.config import YOLO_CONFIG

import glob

def parse_xml(path):
    tree = ET.parse(path)
    img_name = path.split('/')[-1][:-4]

    height = tree.findtext("./size/height")
    width = tree.findtext("./size/width")

    objects = [img_name, width, height]

    for obj in tree.findall('object'):
        difficult = obj.find('difficult').text
        if difficult == '1':
            continue
        name = obj.find('name').text
        bbox = obj.find('bndbox')
        xmin = bbox.find('xmin').text
        ymin = bbox.find('ymin').text
        xmax = bbox.find('xmax').text
        ymax = bbox.find('ymax').text

        objects.extend(['0', xmin, ymin, xmax, ymax])
    if len(objects) > 1:
        return objects
    else:
        return None

def generate_annotation(xml_directory, train_path, val_path):
    with open(train_path, 'w+') as f:

        with open(val_path, 'w+') as fv:

            list_file_xml = glob.glob(xml_directory + '/*.xml')
            train_count = 0
            val_count = 0

            for i, path in enumerate(list_file_xml):

                object = parse_xml(path)
                image_name = object[0] + '.jpg'

                image_name = image_name.strip()

                image_path = os.path.join(os.path.dirname(xml_directory), 'JPEGImages', image_name)

                if os.path.exists(image_path):
                    if i % 4 != 0:
                        object.insert(0, str(train_count))
                        object = ' '.join(object) + '\n'
                        train_count += 1
                        f.write(object)
                    else:
                        object.insert(0, str(val_count))
                        object = ' '.join(object) + '\n'
                        val_count += 1
                        fv.write(object)
                else:
                    print('error')
                    break

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some parameters')

    parser.add_argument('--data',help='Xml Directory Path')

    args = parser.parse_args()
    xml_directory = args.data

    train_path = YOLO_CONFIG.TRAIN_PATH
    val_path = YOLO_CONFIG.VAL_PATH

    generate_annotation(xml_directory,train_path,val_path)


