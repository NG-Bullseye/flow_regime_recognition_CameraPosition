import os
import random
import tensorflow as tf
import json
import cv2
import numpy as np
import yaml
from copy import deepcopy
import joblib
from pathlib import Path


class DataPoint:
    def __init__(self, datapoint_id, path_to_data, path_to_metadata):
        self.datapoint_id = datapoint_id
        self.path_to_data = path_to_data
        self.path_to_metadata = path_to_metadata


class DataParser:
    def __init__(self, path_to_data):
        self.path_to_data = path_to_data
        self.data_points = []
        self.parse_data_points()

    def parse_data_points(self):
        for file in os.listdir(self.path_to_data):
            if os.path.isfile(os.path.join(self.path_to_data, file)) and file.endswith('.png'):
                filename_image = os.path.join(self.path_to_data, file)
                filename = os.path.splitext(file)[0]
                filename_metadata = os.path.join(self.path_to_data, filename + '.json')
                if os.path.isfile(filename_metadata):
                    data_point = DataPoint(filename, filename_image, filename_metadata)
                    self.data_points.append(data_point)
        random.sample(self.data_points, len(self.data_points))


class ImageDataExtractor:
    def __init__(self, output_image_shape):
        self.output_image_shape = output_image_shape

    def get_data(self, data_point):
        filename = data_point.path_to_data
        img = cv2.imread(filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_preprocessed = self.preprocess_image(img)
        return np.expand_dims(img_preprocessed, axis=-1)

    def preprocess_image(self, img):
        img = cv2.resize(img, (self.output_image_shape[0], self.output_image_shape[1]), interpolation=cv2.INTER_CUBIC)
        img = img / 255
        return img

    def get_output_signature(self):
        return self.output_image_shape


class LabelExtractor:
    def __init__(self, no_classes):
        self.no_classes = no_classes
        self.one_hot_encoder = tf.one_hot(range(self.no_classes), self.no_classes)

    def get_data(self, data_point):
        with open(data_point.path_to_metadata) as f:
            metadata = json.load(f)
            label = self.extract_label(metadata)
            return label

    def extract_label(self, metadata):
        label_int = metadata['labels']['flow_regime']['value']
        label_one_hot = self.one_hot_encoder[label_int]
        return label_one_hot

    def get_output_signature(self):
        return (self.no_classes)


class DataSetCreator:
    def __init__(self, data_parser: DataParser, data_extractor, label_extractor, no_repeats=1):
        self.data_points = data_parser.data_points
        self.data_extractor = data_extractor
        self.label_extractor = label_extractor
        self.no_repeats = no_repeats
        self.data_signature_output = self.data_extractor.get_output_signature()
        self.label_signature_output = self.label_extractor.get_output_signature()

    def __len__(self):
        return len(self.data_points)

    def take(self, starting_point=0, no_points=-1):
        if not self.take_args_valid(starting_point, no_points):
            raise Exception
        if no_points == -1:
            data_points = self.data_points[starting_point:]
        else:
            data_points = self.data_points[starting_point:starting_point + no_points]
        new_instance = deepcopy(self)
        new_instance.data_points = data_points
        return new_instance

    def take_args_valid(self, starting_point, no_points):
        return True

    def _data_generator(self):
        for repeat in range(self.no_repeats):
            for data_point in random.sample(self.data_points, len(self.data_points)):
                data = self.data_extractor.get_data(data_point)
                label = self.label_extractor.get_data(data_point)
                yield data, label

    def get_data_point_generator(self):
        return self._data_generator()

    def cast_tf_dataset(self):
        output_signature = (tf.TensorSpec(shape=self.data_signature_output, dtype=tf.float32),
                            tf.TensorSpec(shape=self.label_signature_output, dtype=tf.float32))

        return tf.data.Dataset.from_generator(lambda: self._data_generator(), output_signature=output_signature)


if __name__ == '__main__':
    with open('params.yaml', 'r') as stream:
        params = yaml.safe_load(stream)
    no_epochs = params['training']['no_epochs']
    batch_size = params['training']['batch_size']

    data_dir = '/mnt/0A60B2CB60B2BD2F/Datasets/bioreactor_flow_regimes/02_data'
    data_parser = DataParser(data_dir)
    image_data_extractor = ImageDataExtractor((64, 64, 1))
    label_extractor = LabelExtractor(no_classes=3)
    dataset = DataSetCreator(data_parser, image_data_extractor, label_extractor, no_repeats=no_epochs)
    no_points = 5000  # len(dataset)
    no_points_train = int(no_points * 0.8)
    no_points_val = int(no_points * 0.1)

    dataset_train = dataset.take(no_points=no_points_train)
    dataset_val = dataset.take(starting_point=no_points_train + 1, no_points=no_points_val)
    dataset_test = dataset.take(starting_point=no_points_train + no_points_val + 2,
                                no_points=no_points - no_points_train - no_points_val)

    addit_info = {'data_shape': dataset_train.data_signature_output,
                  'label_shape': dataset_train.label_signature_output,
                  'no_points': no_points,
                  'no_points_train': no_points_train,
                  'no_points_val': no_points_val}

    joblib.dump(addit_info, 'cache/additional_info.joblib')

    tf_dataset_train = dataset_train.cast_tf_dataset().batch(batch_size)
    tf_dataset_val = dataset_val.cast_tf_dataset().batch(batch_size)
    tf_dataset_test = dataset_test.cast_tf_dataset().batch(batch_size)

    tf.data.experimental.save(tf_dataset_train, 'cache/ds_train.tf')
    tf.data.experimental.save(tf_dataset_val, 'cache/ds_val.tf')
    tf.data.experimental.save(tf_dataset_test, 'cache/ds_test.tf')
