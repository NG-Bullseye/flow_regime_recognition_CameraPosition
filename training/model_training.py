import tensorflow as tf
import json
import os
import yaml
import joblib

from lenet5 import LeNet

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

if __name__ == '__main__':
    with open('params.yaml', 'r') as stream:
        params = yaml.safe_load(stream)
    no_epochs = params['training']['no_epochs']
    batch_size = params['training']['batch_size']
    loss = params['training']['loss']

    addit_info = joblib.load('cache/additional_info.joblib')
    data_shape = addit_info['data_shape']
    label_shape = addit_info['label_shape']
    no_points_train = addit_info['no_points_train']
    no_points_val = addit_info['no_points_val']

    tf_dataset_train = tf.data.experimental.load('cache/ds_train.tf')
    tf_dataset_val = tf.data.experimental.load('cache/ds_val.tf')

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model = LeNet
    model_built = model.build(data_shape=data_shape,
                              label_shape=label_shape)
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
