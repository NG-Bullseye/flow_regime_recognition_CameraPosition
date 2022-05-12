import os
import random
import tensorflow as tf
import json
import cv2
import numpy as np


class DatasetGenerator:
    def __init__(self, source_dir, img_shape=(256, 256, 1), no_classes=2, no_points=None):
        self.source_dir = source_dir
        self.img_shape = img_shape
        self.no_classes = no_classes
        self.no_points = no_points

    def __len__(self):
        return len(self._get_data_points_list())

    def get_data_shape(self):
        return self.img_shape

    def get_label_shape(self):
        return (self.no_classes)

    def _get_data_points_list(self):
        image_file = []
        metadata_file = []
        for file in os.listdir(self.source_dir):
            if os.path.isfile(os.path.join(self.source_dir, file)) and file.endswith('.png'):
                filename_image = os.path.join(self.source_dir, file)
                filename = os.path.splitext(file)[0]
                filename_metadata = os.path.join(self.source_dir, filename + '.json')
                if os.path.isfile(filename_metadata):
                    image_file.append(filename_image)
                    metadata_file.append(filename_metadata)

        data_points = list(zip(image_file, metadata_file))
        shuffled_data_points = random.sample(data_points, len(data_points))

        if self.no_points is None or self.no_points >= len(image_file):
            return shuffled_data_points
        elif self.no_points < len(image_file):
            return shuffled_data_points[:self.no_points]
        else:
            return None

    def read_image(self, file):
        img = cv2.imread(file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_height, img_width, _ = self.img_shape
        img = cv2.resize(img, (img_width, img_height), interpolation=cv2.INTER_CUBIC)
        img = img/255
        return np.expand_dims(img, axis=-1)

    def read_label(self, file):
        one_hot_encoder = tf.one_hot(range(self.no_classes), self.no_classes)
        with open(file) as f:
            json_content = json.load(f)
            label_int = json_content['labels']['flow_regime']['value']
            label = one_hot_encoder[label_int]
            return label

    def _data_generator(self, list_data_points, repeats=1):
        for repeat in range(repeats):
            for data_point in random.sample(list_data_points, len(list_data_points)):
                image_file = data_point[0]
                image_data = self.read_image(image_file)

                label_file = data_point[1]
                label_data = self.read_label(label_file)
                #print(f'Image: {image_file}, label: {label_file}')
                yield image_data, label_data

    def get_data_point_generator(self):
        data_points = self._get_data_points_list()
        return self._data_generator(data_points)

    def split_and_cast_tf_datasets(self, split_ratio, epochs=1, save_test_dataset=True):
        data_points = self._get_data_points_list()
        dataset_len = len(data_points)

        no_train_points = int(split_ratio[0] / sum(split_ratio) * dataset_len)
        no_val_points = int(split_ratio[1] / sum(split_ratio) * dataset_len)
        no_test_points = int(split_ratio[2] / sum(split_ratio) * dataset_len)

        data_points_train = data_points[:no_train_points]
        data_points_val = data_points[no_train_points:no_train_points + no_val_points]
        data_points_test = data_points[no_train_points + no_val_points:]

        data_gen_train = self._data_generator(data_points_train, repeats=epochs)

        output_signature = (tf.TensorSpec(shape=self.img_shape, dtype=tf.float32),
                            tf.TensorSpec(shape=(self.no_classes), dtype=tf.bool))

        dataset_train = tf.data.Dataset.from_generator(lambda: data_gen_train, output_signature=output_signature)
        dataset_train._len = no_train_points

        data_gen_val = self._data_generator(data_points_val, repeats=1)
        dataset_val = tf.data.Dataset.from_generator(lambda: data_gen_val, output_signature=output_signature)
        dataset_val._len = no_val_points

        data_gen_test = self._data_generator(data_points_test, repeats=1)
        dataset_test = tf.data.Dataset.from_generator(lambda: data_gen_test, output_signature=output_signature)
        dataset_test._len = no_test_points

        if save_test_dataset:
            tf.data.experimental.save(dataset_test, '../cache/ds_test.tf')

        return (dataset_train, dataset_val, dataset_test)
