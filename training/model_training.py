import tensorflow as tf
import json
import os
import sys
import yaml

p = os.path.abspath('.')
sys.path.insert(1, p)
from models.lenet5 import LeNet
from data_import_and_preprocessing.dataset_formation import DataParser, ImageDataExtractor, LabelExtractor, \
    DataSetCreator

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == '__main__':
    with open('params.yaml', 'r') as stream:
        params = yaml.safe_load(stream)
    no_epochs = params['training']['no_epochs']
    batch_size = params['training']['batch_size']
    loss = params['training']['loss']

    data_dir = 'data/preprocessed'
    data_parser = DataParser(data_dir)
    image_data_extractor = ImageDataExtractor((64, 64, 1))
    label_extractor = LabelExtractor(no_classes=3)
    dataset = DataSetCreator(data_parser, image_data_extractor, label_extractor, no_repeats=no_epochs)
    no_points = len(dataset)
    no_points_train = int(no_points * 0.8)
    no_points_val = int(no_points * 0.1)

    dataset_train = dataset.take(no_points=no_points_train)
    dataset_val = dataset.take(starting_point=no_points_train + 1, no_points=no_points_val)
    dataset_test = dataset.take(starting_point=no_points_train + no_points_val + 2,
                                no_points=no_points - no_points_train - no_points_val)
    dataset_test.no_repeats = 1

    addit_info = {'data_shape': dataset_train.data_signature_output,
                  'label_shape': dataset_train.label_signature_output,
                  'no_points': no_points,
                  'no_points_train': no_points_train,
                  'no_points_val': no_points_val,
                  'no_points_test': no_points - no_points_train - no_points_val,
                  'data_points_train': [datapoint.datapoint_id for datapoint in dataset_train.data_points],
                  'data_points_val': [datapoint.datapoint_id for datapoint in dataset_val.data_points],
                  'data_points_test': [datapoint.datapoint_id for datapoint in dataset_test.data_points]}

    with open('cache/additional_info.yaml', 'w') as outfile:
        yaml.dump(addit_info, outfile, default_flow_style=False, sort_keys=False)

    tf_dataset_train = dataset_train.cast_tf_dataset().batch(batch_size).prefetch(1)
    tf_dataset_val = dataset_val.cast_tf_dataset().batch(batch_size).prefetch(1)
    tf_dataset_test = dataset_test.cast_tf_dataset().batch(batch_size)

    tf.data.experimental.save(tf_dataset_test, 'cache/ds_test.tf')

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model = LeNet
    model_built = model.build(data_shape=dataset_train.data_signature_output,
                              label_shape=dataset_train.label_signature_output)
    model_built.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

    history = model_built.fit(tf_dataset_train,
                              epochs=no_epochs,
                              batch_size=batch_size,
                              steps_per_epoch=no_points_train // batch_size,
                              validation_data=tf_dataset_val,
                              validation_steps=no_points_val // batch_size,
                              callbacks=[],
                              )

    model_built.save('training/model/trained_model.h5')

    with open('training/results/training_report.json', 'w', encoding='utf-8') as f:
        json.dump(history.history, f, ensure_ascii=False, indent=4)
