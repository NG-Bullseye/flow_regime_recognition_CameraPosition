import os
from model import LeNet
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import random
import json

from read_and_preprocess_data import get_data_points_list, data_generator

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

NO_EPOCHS = 2
BATCH_SIZE = 16
INIT_LR = 0.001
SPLIT_RATIO = [0.9, 0.1, 0.0]

checkpoint_path = '../model/checkpoint-{epoch:04d}.ckpt'

data_folder = '/mnt/0A60B2CB60B2BD2F/Datasets/bioreactor_flow_regimes/02_data'
data_points = get_data_points_list(data_folder)
shuffled_data_points = random.sample(data_points, len(data_points))
dataset_len = len(shuffled_data_points)

no_train_points = int(SPLIT_RATIO[0]/sum(SPLIT_RATIO)*dataset_len)
no_val_points = int(SPLIT_RATIO[1]/sum(SPLIT_RATIO)*dataset_len)
no_test_points = int(SPLIT_RATIO[2]/sum(SPLIT_RATIO)*dataset_len)
data_points_train = shuffled_data_points[:no_train_points]
data_points_val = shuffled_data_points[no_train_points:no_train_points+no_val_points]
data_points_test = shuffled_data_points[no_train_points+no_val_points:]

data_gen_train = data_generator(data_points_train, NO_EPOCHS)

output_signature = (tf.TensorSpec(shape=(54, 96, 1), dtype=tf.float32),
                    tf.TensorSpec(shape=(3), dtype=tf.bool))

dataset_train = tf.data.Dataset.from_generator(lambda: data_gen_train, output_signature=output_signature)

data_gen_val = data_generator(data_points_val, NO_EPOCHS)
dataset_val = tf.data.Dataset.from_generator(lambda: data_gen_val, output_signature=output_signature)

opt = Adam(learning_rate=INIT_LR, decay=INIT_LR / NO_EPOCHS)
model = LeNet.build(width=54, height=96, depth=1, classes=3)

model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
model.summary()

ds_train_batched = dataset_train.batch(BATCH_SIZE)
ds_val_batched = dataset_val.batch(BATCH_SIZE)

save_every_epoch_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    verbose=1,
    save_weights_only=True,
    save_freq=no_train_points // BATCH_SIZE)

model.save_weights(checkpoint_path.format(epoch=0))

history = model.fit(ds_train_batched,
                    epochs=NO_EPOCHS,
                    batch_size=BATCH_SIZE,
                    steps_per_epoch=no_train_points // BATCH_SIZE,
                    validation_data=ds_val_batched,
                    validation_steps=no_val_points // BATCH_SIZE,
                    callbacks=[save_every_epoch_callback],
                    )

print(history.history)
with open('./report.json', 'w', encoding='utf-8') as f:
    json.dump(history.history, f, ensure_ascii=False, indent=4)
