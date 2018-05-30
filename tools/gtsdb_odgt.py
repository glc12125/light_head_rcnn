import os
import sys
import json
import glob
import PIL.Image
import numpy as np


from lxml import etree

debug = False #only load 10 images
shuffle = False # shuffle dataset

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

text = []
image_labels = []
root_dir = '/Users/liangchuangu/Development/machine_learning/darknet-traffic/gtsdb'
data_dir = '/Users/liangchuangu/Development/machine_learning/darknet-traffic/gtsdb/images'

training_list = get_training_examples(root_dir)
print("There are " + str(len(training_list)) + " labels\n")
unique_set = set([])
for idx, example in enumerate(training_list):
    unique_set.add(os.path.join(data_dir, example.split(';')[0].split('.')[0] + '.jpg'))
print("There are " + str(len(unique_set)) + " images containing labels\n")

with open(root_dir + '/train.txt', 'w') as train_file:
    for path in unique_set:
        train_file.write(path + "\n")


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

# save labels for each image
j = 0
for labels in image_labels :
    print("Image label #" + str(j) + ", labels: " + str(labels))
    img = np.array(PIL.Image.open(os.path.join(labels[0][0], labels[0][1])), dtype=np.uint8)
    print(img.shape)
    width = img.shape[1]
    height = img.shape[0]
    with open(root_dir + '/labels/' + '{:05}'.format(j) + '.txt', 'w') as label_file:
        for label in labels[1:]:
            label_file.write(str(label[0]) + " " + str(label[1]/width) + " " + str(label[2]/height) + " " + str((label[3] - label[1])/width) + " " + str((label[4] - label[2])/height) + "\n")
    j = j + 1
