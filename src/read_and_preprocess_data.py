import os
import random
import tensorflow as tf
import json


def get_data_points_list(source_dir):
    image_file = []
    metadata_file = []
    for file in os.listdir(source_dir):
        if os.path.isfile(os.path.join(source_dir, file)) and file.endswith('.png'):
            filename_image = os.path.join(source_dir, file)
            filename = os.path.splitext(file)[0]
            filename_metadata = os.path.join(source_dir, filename + '.json')
            if os.path.isfile(filename_metadata):
                image_file.append(filename_image)
                metadata_file.append(filename_metadata)
    #return list(zip(image_file, metadata_file))[:200]
    return list(zip(image_file, metadata_file))


def read_image(file):
    image_file = tf.io.read_file(file)
    image_data = tf.image.decode_image(image_file)
    return image_data


def read_label(file):
    no_classes = 3
    one_hot_encoder = tf.one_hot(range(no_classes), no_classes)
    with open(file) as f:
        json_content = json.load(f)
        label_int = json_content['labels']['flow_regime']['value']
        label = one_hot_encoder[label_int]
        return label


def data_generator(list_data_points, repeats):
    for repeat in range(repeats):
        for data_point in random.sample(list_data_points, len(list_data_points)):
            image_file = data_point[0]
            image_original = read_image(image_file)
            image_grayscaled = tf.image.rgb_to_grayscale(image_original)
            image_resized = tf.image.resize(image_grayscaled, [54, 96], method='bicubic')
            image_data = image_resized

            label_file = data_point[1]
            label_data = read_label(label_file)
            #print(image_file)
            yield image_data, label_data
