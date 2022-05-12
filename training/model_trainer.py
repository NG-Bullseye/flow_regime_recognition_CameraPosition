from model.model import LeNet
from data_import_and_preprocessing.dataset_generator import DatasetGenerator
import tensorflow as tf
import numpy as np
import cv2
import json
import os


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


class ModelTrainer:
    def __init__(self,
                 data_generator,
                 model,
                 optimizer,
                 split_ratio=(0.8, 0.1, 0.1),
                 no_epochs=1,
                 batch_size=128,
                 loss='categorical_crossentropy',
                 metrics='accuracy'):
        self.data_generator = data_generator
        self.model = model.build(data_shape=self.data_generator.get_data_shape(),
                                 label_shape=self.data_generator.get_label_shape())
        self.optimizer = optimizer
        self.split_ratio = split_ratio
        self.no_epochs = no_epochs
        self.batch_size = batch_size
        self.loss = loss
        self.metrics = metrics

    def _compile_model(self):
        self.model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)

    def _prepare_datasets(self):
        ds_train, ds_val, _ = self.data_generator.split_and_cast_tf_datasets(self.split_ratio, self.no_epochs)
        ds_train_len = ds_train._len
        ds_val_len = ds_val._len

        ds_train = ds_train.batch(self.batch_size).prefetch(2)
        ds_train._len = ds_train_len
        ds_val = ds_val.batch(self.batch_size).prefetch(2)
        ds_val._len = ds_val_len
        return ds_train, ds_val

    def train(self, callbacks=[]):
        ds_train, ds_val = self._prepare_datasets()
        self._compile_model()
        history = self.model.fit(ds_train,
                                 epochs=self.no_epochs,
                                 batch_size=self.batch_size,
                                 steps_per_epoch=ds_train._len // self.batch_size,
                                 validation_data=ds_val,
                                 validation_steps=ds_val._len // self.batch_size,
                                 callbacks=callbacks,
                                 )
        return history

    def get_model(self):
        return self.model


class CustomDatasetGenerator(DatasetGenerator):
    def read_image(self, file):
        img = cv2.imread(file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_height, img_width, _ = self.img_shape
        img = cv2.resize(img, (img_width, img_height), interpolation=cv2.INTER_CUBIC)
        #img = cv2.bitwise_not(img)
        img = img/255
        return np.expand_dims(img, axis=-1)


if __name__ == '__main__':
    data_dir = '/mnt/0A60B2CB60B2BD2F/Datasets/bioreactor_flow_regimes/02_data'
    image_data_generator = DatasetGenerator(data_dir, img_shape=(1920//5, 1080//5, 1), no_classes=3, no_points=1000)
    model_trainer = ModelTrainer(data_generator=image_data_generator,
                                 model=LeNet,
                                 optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                                 split_ratio=(0.8, 0.1, 0.1),
                                 no_epochs=1,
                                 batch_size=64)

    trained_model = model_trainer.get_model()
    trained_model.save('./results/trained_model.h5')

    training_history = model_trainer.train()
    with open('./results/training_report.json', 'w', encoding='utf-8') as f:
        json.dump(training_history.history, f, ensure_ascii=False, indent=4)


