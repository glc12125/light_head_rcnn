import os
import sys
import json
import glob
import PIL.Image
import numpy as np


from lxml import etree

debug = False #only load 10 images
shuffle = True # shuffle dataset

def get_training_examples(dir):
  examples_list = []
  annotation_file = os.path.join(dir, "gt.txt")
  print(annotation_file)
  examples_list = examples_list + read_examples_list(annotation_file)
  #print(*examples_list, sep='\n')
  return examples_list

def read_examples_list(path):
  with open(path, 'r') as f:
    lines = f.readlines()
    print(*lines)
  return [line.strip().split(' ')[0] for line in lines]

def get_id_class_mapping(dir):
    mapping = {}
    class_file = os.path.join(dir, "gtsdb_classes.txt")
    print(class_file)
    class_names = read_classes_list(class_file)
    class_id = 0
    for class_name in class_names:
        mapping[class_id] = class_name
        class_id = class_id + 1
    return mapping

def read_classes_list(path):
  with open(path, 'r') as f:
    lines = f.readlines()
  return [line.strip().split('\n')[0] for line in lines]

text = []
image_labels = []
#root_dir = '/Users/liangchuangu/Development/machine_learning/darknet-traffic/gtsdb'
root_dir = '/home/robok/Development/darknet-traffic/gtsdb'
#od_dir = '/Users/liangchuangu/Development/machine_learning/light_head_rcnn/data/gtsdb'
#data_dir = '/Users/liangchuangu/Development/machine_learning/darknet-traffic/gtsdb/images'
od_dir = '/home/robok/Development/light_head_rcnn/data/gtsdb'
data_dir = '/home/robok/Development/darknet-traffic/gtsdb/images'

training_list = get_training_examples(root_dir)
print("There are " + str(len(training_list)) + " labels\n")
unique_set = set([])
for idx, example in enumerate(training_list):
    unique_set.add(os.path.join(data_dir, example.split(';')[0].split('.')[0] + '.jpg'))
print("There are " + str(len(unique_set)) + " images containing labels\n")

class_id_name_map = get_id_class_mapping(od_dir)

name_to_index_map = {}
index = -1
for idx, example in enumerate(training_list):
    #print("exmaple: " + example)
    training_data = example.split(';')
    if training_data[0] in name_to_index_map:
        index = name_to_index_map[training_data[0]]
    else:
        index = index + 1
        name_to_index_map[training_data[0]] = index
        if index >= len(image_labels):
            image_labels.append([])
            image_labels[index].append([data_dir, training_data[0].split('.')[0] + '.jpg']) # for path components array
    #print("index: " + str(index) + ", len(image_labels): " + str(len(image_labels)))

    # for labels
    boxConfig = []
    boxConfig.append(int(training_data[5]))
    boxConfig.append(float(training_data[1]))
    boxConfig.append(float(training_data[2]))
    boxConfig.append(float(training_data[3]))
    boxConfig.append(float(training_data[4]))
    image_labels[index].append(boxConfig)
    if index % 100 == 0:
        print('On image %d of %d', index, len(training_list))


#shuffle dataset
if shuffle:
    np.random.shuffle(image_labels)

trainSeparator = int(len(image_labels) * 0.7)

with open(od_dir + '/odformat/gtsdb_train.odgt', 'w') as od_file:
    for labels in image_labels[0:trainSeparator] :
        print("Image labels: " + str(labels))
        img = np.array(PIL.Image.open(os.path.join(labels[0][0], labels[0][1])), dtype=np.uint8)
        print(img.shape)
        width = img.shape[1]
        height = img.shape[0]
        data = {}
        data["dbName"] = 'gtsdb'
        data["width"] = width
        data["height"] = height
        data["ID"] = labels[0][1]
        gtboxes = data["gtboxes"] = []
        data["fpath"] = labels[0][0] + "/" + labels[0][1]
        for label in labels[1:]:
            info = {}
            info["box"] = [int(label[1]),  int(label[2]), int(label[3] - label[1]), int(label[4] - label[2])]
            info["occ"] = 0
            info["tag"] = class_id_name_map[label[0]]
            gtboxes.append(info)
        json.dump(data, od_file)
        od_file.write("\n")

with open(od_dir + '/odformat/gtsdb_val.odgt', 'w') as od_file:
    for labels in image_labels[trainSeparator:] :
        print("Image labels: " + str(labels))
        img = np.array(PIL.Image.open(os.path.join(labels[0][0], labels[0][1])), dtype=np.uint8)
        print(img.shape)
        width = img.shape[1]
        height = img.shape[0]
        data = {}
        data["dbName"] = 'gtsdb'
        data["width"] = width
        data["height"] = height
        data["ID"] = labels[0][1]
        gtboxes = data["gtboxes"] = []
        data["fpath"] = labels[0][0] + "/" + labels[0][1]
        for label in labels[1:]:
            info = {}
            info["box"] = [label[1],  label[2], label[3], label[4]]
            info["occ"] = 0
            info["tag"] = class_id_name_map[label[0]]
            gtboxes.append(info)
        json.dump(data, od_file)
        od_file.write("\n")
