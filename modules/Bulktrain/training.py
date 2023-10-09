
from silence_tensorflow import silence_tensorflow
import time

from modules.Utility.CustomImageDataExtractor import CustomImageDataExtractor
from modules.Utility.PrettySafeLoader import PrettySafeLoader

silence_tensorflow()
from datetime import datetime
from modules.Utility.notificationBot import senden
import tensorflow as tf
import sys
import yaml
import pandas as pd
import numpy as np
from models.lenet5 import LeNet_baseline
from models.lenet5 import LeNet_drop_reg
from models.lenet5 import LeNet_Hypermodel
from models.lenet5 import LeNet_reduced

from models.lenet5 import Lenet5_kroger
#from models.lenet5 import custom_net as Lenet5_kroger
from collections import defaultdict

import os
import subprocess
import re
import queue
import threading


class Training:


  def __init__(self,input_training_image_path,output_training_path):
    self.output_training_path = output_training_path  # This should already be unique before calling __init__
    self.auc = None
    self.input_training_image_path = input_training_image_path
    self.path_exp = None
    self.evaluate_only = False
    self.tuning = False
    self.tensorboard_url = ""
    self.LEARNRATE_SCHEDULE = False
    self.callback_toogled = []
    self.EARLYSTOP = False
    self.METRIC = 'categorical_accuracy'
    self.best_acc = 0
    self.hyperModel_built = None
    self.dropout_rate = 0
    print("############ INITIALIZATION STARTED ############")
    self.trainings_laufzeit=0
    self.tuning_laufzeit = 0
    self.my_hypermodel = LeNet_Hypermodel()
    self.output_queue = queue.Queue()
    self.model=Lenet5_kroger

    ######################################################
    #METHA PARAMETER WHICH CANT BE PLACED IN PARAMS.YAML
    p = os.path.abspath('../training')
    sys.path.insert(1, p)
    from modules.Preprocessing.dataset_formation import DataParser, LabelExtractor, \
      DataSetCreator
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    self.physical_devices = tf.config.list_physical_devices('CPU')
    #tf.config.experimental.set_memory_growth(self.physical_devices[0], True)



    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)
    os.chdir('../')
    with open('../params.yaml', 'r') as stream:
      self.params = yaml.load(stream, Loader=PrettySafeLoader)
    #start_time = datetime.now()
    print(str(self.params))
    if self.params["run_tuner"]:
      self.tuning=True
    self.lr_start = self.params['training']['learningrate']
    self.lr_max = self.params    ['callbacks']['lr_schedule']['lr_max']
    self.lr_min = self.params    ['callbacks']['lr_schedule']['lr_min']
    self.lr_ramp_ep = self.params['callbacks']['lr_schedule']['lr_ramp_ep']
    self.lr_sus_ep = self.params ['callbacks']['lr_schedule']['lr_sus_ep']
    self.lr_decay = self.params  ['callbacks']['lr_schedule']['lr_decay']

    #Parameter
    picture_width   = self.params['preprocessing']['picture_width']
    picture_hight   = self.params['preprocessing']['picture_hight']
    no_classes      = self.params['preprocessing']['no_classes']

    os.chdir(self.output_training_path)
    print("Training Dir: "+os.getcwd())

    #Hyperparameter Tuned
    self.batch_size = self.params['training']['hp']['batch_size']
    self.dropout_rate   = self.params['training']['hp']['dropout']
    self.regularization = self.params['training']['hp']['regularization']

    self.best_regularization = 0
    self.best_dropout_rate = 0
    self.best_batchsize = 0
    self.best_exp_name=""
    #Parameters not Tuned
    self.learningrate = self.params['training']['learningrate']
    self.lr_start = self.learningrate
    self.callback_learningrate_schedule = tf.keras.callbacks.LearningRateScheduler(self.lrfn, verbose=True)

    self.no_epochs = self.params['training']['no_epochs']

    self.no_epochs = self.params['training']['no_epochs']
    self.loss = self.params['training']['loss']
    self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learningrate)
    self.dataset_test_no_repeats = self.params['training']['dataset_test_no_repeats']

    #Fetch DATASET
    data_parser         = DataParser(self.input_training_image_path)
    image_data_extractor = CustomImageDataExtractor((picture_width, picture_hight, 1))
    label_extractor     = LabelExtractor(no_classes=no_classes) #anzahl der classen
    self.dataset        = DataSetCreator(data_parser, image_data_extractor, label_extractor, no_repeats=self.no_epochs)
    #SPLIT DATASET INTO TRAINING VALIDATION AND TEST
    self.no_points = len(self.dataset)
    if self.tuning:
      self.no_points_train = int(self.no_points * self.params['tuning']['no_points_train_ratioInPercent'])
      self.no_points_val = int(self.no_points * self.params['tuning']['no_points_val_ratioInPercent'])
    else:
      self.no_points_train = int(self.no_points * self.params['training']['no_points_train_ratioInPercent'])
      self.no_points_val   = int(self.no_points * self.params['training']['no_points_val_ratioInPercent'])
    self.no_points_test  = self.no_points - self.no_points_train - self.no_points_val


    import numpy as np
    from sklearn.model_selection import StratifiedShuffleSplit
    X = np.array([[1, 2], [3, 4], [1, 2], [3, 4], [1, 2], [3, 4]])
    y = np.array([0, 0, 0, 1, 1, 1])
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    sss.get_n_splits(X, y)
    for train_index, test_index in sss.split(X, y):
      print("TRAIN:", train_index, "TEST:", test_index)
      X_train, X_test = X[train_index], X[test_index]
      y_train, y_test = y[train_index], y[test_index]

    self.dataset_train   = self.dataset.take(no_points=self.no_points_train)
    self.dataset_val     = self.dataset.take(starting_point=self.no_points_train + 1, no_points=self.no_points_val)
    self.dataset_val.no_repeats = 1
    self.dataset_test    = self.dataset.take(starting_point=self.no_points_train + self.no_points_val + 2,
                                             no_points=self.no_points_test)

    self.dataset_test.no_repeats = 1

    self.tf_dataset_train = self.dataset_train.cast_tf_dataset().batch(self.batch_size).prefetch(1)
    self.tf_dataset_val = self.dataset_val.cast_tf_dataset().batch(self.batch_size).prefetch(1)
    count = sum(1 for _ in self.tf_dataset_val)
    print(f"Number of batches in tf_dataset_val: {count}")
    for batch in self.tf_dataset_val.take(1):
      print(f"Number of images in one batch: {batch[0].shape[0]}")

    self.tf_dataset_test = self.dataset_test.cast_tf_dataset().batch(self.batch_size)
    # Callback to stop training after no performance decrease
    self.callback_early_stopping = tf.keras.callbacks.EarlyStopping(patience=50,
                                                                    restore_best_weights=True,
                                                                    monitor='categorical_accuracy',
                                                                    min_delta=0.01,
                                                                    mode='auto',
                                                                    verbose=1,
                                                                    baseline=None
                                                                    )

    if (self.params['callbacks']['earlystop']):
      print("EARLY STOPPING: ON")
      self.callback_toogled.append(self.callback_early_stopping)
    else:
      print("EARLY STOPPING: OFF")
    if (self.params['callbacks']['lr_schedule']['enabled']):
      print("LEARNINGRATE SCHEDULE: ON")
      self.callback_toogled.append(self.callback_learningrate_schedule)
    else:
      print("LEARNINGRATE SCHEDULE: OFF")

  def start_tensorboard(self):
    # Start TensorBoard as a subprocess
    self.proc = subprocess.Popen(
      [
        "tensorboard", "dev", "upload",
        "--logdir", "./tensorboard",
        "--name",
        f"Batchsize={self.params['training']['hp']['batch_size']} Dropout={self.params['training']['hp']['dropout']} Reg={self.params['training']['hp']['regularization']} Lr={self.params['training']['learningrate']}",
        "--description", f"Experiment path: {self.output_training_path}"
      ],
      stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)

    # Use regex to match TensorBoard url pattern
    url_pattern = re.compile(r'https?://tensorboard.dev/experiment/.*')

    # Run until the TensorBoard process finishes
    for line in iter(self.proc.stdout.readline, ''):
      # The real code does filtering here
      print(line.strip())
      url_match = url_pattern.search(line)
      if url_match:
        self.tensorboard_url = url_match.group(0)
        # Put URL in queue to signal main thread
        self.output_queue.put(self.tensorboard_url)

  def check_queue(self):
    while True:
      try:
        url = self.output_queue.get(timeout=5)
        print(f"Received TensorBoard URL: {url}")
      except queue.Empty:
        print("Waiting for TensorBoard URL...")
        continue
      else:
        # We've got the URL, no need to check anymore
        break

  def start_threads(self):
    self.thread_tensorboard = threading.Thread(target=self.start_tensorboard)
    self.thread_tensorboard.start()

    self.thread_check_queue = threading.Thread(target=self.check_queue)
    self.thread_check_queue.start()

  def stop_tensorboard(self):
    if self.proc:
      self.proc.terminate()
    if self.thread_tensorboard.is_alive():
      self.thread_tensorboard.join()
    if self.thread_check_queue.is_alive():
      self.thread_check_queue.join()
  def train(self):
    print("Training Dataset element specification:")
    for features, labels in self.tf_dataset_train.take(1):
      print(f"Features: {features.shape}, Labels: {labels.shape}")

    print("Validation Dataset element specification:")
    for features, labels in self.tf_dataset_val.take(1):
      print(f"Features: {features.shape}, Labels: {labels.shape}")

    # Callback to write logs onto tensorboard
    # To run tensorboard execute command: tensorboard --logdir training/results/tensorboard
    os.chdir(self.output_training_path)
    #print(os.getcwd())
    if not os.path.exists("tensorboard"):
      os.makedirs("tensorboard")
    self.tb_callback = tf.keras.callbacks.TensorBoard(log_dir=self.output_training_path + "/tensorboard",
                                                      histogram_freq=1,
                                                      write_graph=True,
                                                      write_images=False,
                                                      update_freq='epoch',
                                                      profile_batch=2,
                                                      embeddings_freq=1)
    # Initialize an empty Queue to pass messages from the output processing thread to the main thread
   # self.start_threads()

    self.time_start = datetime.now()
    # Custom Scheduler Function
    self.model_built = self.model.build(self.params['datashape'],self.params['labelshape'],dropout_rate= self.dropout_rate,regularization= self.regularization)

    self.model_built.compile(loss=self.loss, optimizer=self.optimizer, metrics=[tf.keras.metrics.CategoricalAccuracy()],run_eagerly=True)
    self.model_built.summary()
    print("epochs: "          +str(self.no_epochs))
    print("batch_size: "      +str(self.batch_size))
    print("dropout: "+str(self.dropout_rate))
    print("regulaization: " +str(self.regularization))
    print("no_points_train: " +str(self.no_points_train ))
    print("no_points_val: "   +str(self.no_points_val))
    print("Output Path " + self.output_training_path)
    print("Input Data Path " + self.input_training_image_path)

    self.path_checkpoint = self.output_training_path + f"/checkpoints/B{str(self.batch_size).split('.', 1)[0]}_D{str(self.dropout_rate).split('.', 1)[0]}_R{str(self.regularization).split('.', 1)[0]}"
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
      filepath=self.path_checkpoint,
      save_weights_only=True,
      monitor='val_categorical_accuracy',
      mode='max',
      save_best_only=True,
      verbose=0)
    callbacks=self.callback_toogled
    #callbacks.append(self.tb_callback)
    callbacks.append(model_checkpoint_callback)
    self.history = self.model_built.fit(self.tf_dataset_train,
                                        epochs=self.no_epochs,
                                        batch_size=self.batch_size,
                                        steps_per_epoch=self.no_points_train // self.batch_size,
                                        validation_data=self.tf_dataset_val,
                                        validation_steps=self.no_points_val // self.batch_size,
                                        callbacks=callbacks)

    td =datetime.now() - self.time_start
    days = td.days
    hours, remainder = divmod(td.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    self.trainings_laufzeit  = f"Days:{days}, Hours:{hours}, Minutes:{minutes}"
    print("Saving test Data Split tf_dataset_test..")
    tf.data.experimental.save(self.tf_dataset_test, self.output_training_path + '/tf_dataset_test.tf')
    ds_element_spec = self.tf_dataset_test.element_spec
    import pickle
    # Save the ds_element_spec to a file using pickle for gradcam generation later on
    with open(self.output_training_path + '/element_spec.pkl', 'wb') as file:
      pickle.dump(ds_element_spec, file)
    #self.model_built.load_weights(self.path_checkpoint)
    if self.params['callbacks']['earlystop']:
      print("EARLY STOPPED AT: " + str(self.callback_early_stopping.stopped_epoch))

  def parseConfMat(self, matrix):
    return str(str(str(matrix).split('(',1)[1]).split(',',1)[0])
  def lrfn(self, epoch):
    if epoch < self.lr_ramp_ep:
      lr = (self.lr_max - self.lr_start) / self.lr_ramp_ep * epoch + self.lr_start

    elif epoch < self.lr_ramp_ep + self.lr_sus_ep:
      lr = self.lr_max

    else:
      lr = (self.lr_max - self.lr_min) * self.lr_decay ** (epoch - self.lr_ramp_ep - self.lr_sus_ep) + self.lr_min
    print("Updated Learningrate: " + str(lr))
    return lr

  def start_tensorboard_tuning(self):
    os.chdir(self.output_training_path)
    print(f"starting tensorboard in dir: {os.getcwd()}/hparam_tuning")
    print(os.system(f"tensorboard dev upload \
    --logdir ./hparam_tuning \
    --name 'Tuning {self.params['path_dataset']}' \
    --description 'Experiment path: {self.output_training_path}'"))

  def loadModel(self, path_model):
    self.tf_dataset_test= tf.data.experimental.load('../cache/ds_test.tf')
    self.model_built=tf.keras.models.load_model(f"{self.output_training_path}/{path_model}")
    self.evaluate_only=True

  def evaluate(self):
    #MODEL PREDICTS
    print("#################### EVALUTATION ####################")

    self.results = self.model_built.evaluate(self.tf_dataset_test, verbose=True)
    print(f"Best Weights Evaluation: loss {self.results[0]}, acc {self.results[1]}")
    os.chdir(self.output_training_path)
    x_test = []
    y_test = []
    for image, label in list(self.tf_dataset_test.unbatch().as_numpy_iterator()):
      tf_image=tf.convert_to_tensor(image)
      tf_image_batched=tf.expand_dims(tf_image, axis=0)
      x_test.append(tf_image_batched)
      y_test.append(label)
    y_test_idx = tf.argmax(y_test, axis=1)
    if self.tuning:
      print(
        f"BEST TUNING HYPERPARAMETERS: Batchsize={self.best_batchsize} Reg={self.best_regularization} Drop={self.best_dropout_rate}")
    print("### PREDICT TEST DATASET FOR CONSUFION MATRIX ###")
    predictions = [self.model_built.predict_on_batch(x) for x in x_test]
    predictions_mat = np.vstack(predictions)
    predictions_idx = np.argmax(predictions_mat, axis=1)
    self.conf_mat = tf.math.confusion_matrix(y_test_idx, predictions_idx)
    print("Confusion Matrix: " + self.parseConfMat(self.conf_mat))
    # Then interpolate all ROC curves at this points
    self.model_built.save(self.output_training_path + '/cnn_acc' + str(round(self.results[1] * 100000, 0)/1000) +'.h5')
  def report(self):
    print("##############################  \n ########## REPORT ########## \n ##############################")
    os.chdir(self.output_training_path)
    # convert the history.history dict to a pandas DataFrame:
    hist_df = pd.DataFrame(self.history.history)
    # save to json:
    hist_json_file = './training_history.json'
    with open(hist_json_file, mode='w') as f:
      hist_df.to_json(f)
    print("RESULTS: "+str(self.results))
    # Check if self.tensorboard_url has been set
    time.sleep(10)  # 10 seconds, adjust as needed
    if hasattr(self, 'tensorboard_url'):
      # Now it's safe to use self.tensorboard_url in your main thread
      print("Tensorboard URL: "+self.tensorboard_url)
    else:
      # Handle case where self.tensorboard_url is not yet available
      print("TensorBoard URL not yet available.")

    training_info = {
      'tensorboard_url': self.tensorboard_url,
      'val_loss': self.results[0],
      'val_acc': self.results[1],
      'conf_mat': self.conf_mat,
      'results_array': self.results,
      'learing_rate': self.learningrate,
      'no_epochs': self.no_epochs,
      'batch_size': self.batch_size,
      'datashape': self.dataset_train.data_signature_output,
      'label_shape': self.dataset_train.label_signature_output,
      'no_points': self.no_points,
      'no_points_train': self.no_points_train,
      'no_points_val': self.no_points_val,
      'no_points_test': self.no_points - self.no_points_train - self.no_points_val,
      'data_points_train': [datapoint.datapoint_id for datapoint in self.dataset_train.data_points],
      'data_points_val': [datapoint.datapoint_id for datapoint in self.dataset_val.data_points],
      'data_points_test': [datapoint.datapoint_id for datapoint in self.dataset_test.data_points],
      'trainings_laufzeit': self.trainings_laufzeit,
      'tuning_laufzeit': self.tuning_laufzeit
    }

    with open(self.output_training_path + '/results.yaml', 'w') as outfile:
      yaml.dump(training_info, outfile, default_flow_style=False, sort_keys=False)
    with open(self.output_training_path + '/params_for_this_training.yaml', 'w') as outfile:
        yaml.dump(self.params, outfile, default_flow_style=False, sort_keys=False)
    try:
      with open(self.output_training_path + '/tensorboard_link.txt', 'w') as f:
        f.write(self.tensorboard_url)
    except FileNotFoundError:
      print(f"The {self.output_training_path} directory does not exist")
    if self.tuning:
      senden("Tuning Finished! "    + "ACC: " + str(self.results[1]) + "\n Tuning_Laufzeit: "      + str(self.tuning_laufzeit)+" Confusion_matrix:"+ str(self.conf_mat)+" reg: "+str(self.best_regularization)+" batch:"+str(self.best_batchsize)+" drop:"+str(self.best_dropout_rate))
    else:
      senden("Training Finished! " + "ACC: " + str(self.results[1]) + "\n Trainings_Laufzeit: " + str(self.trainings_laufzeit) +" Confusion_matrix:" + self.parseConfMat(self.conf_mat) +"\n AUC: " + str(self.auc) +"\n Path: " + self.output_training_path + "\n Tensorboard: " + self.tensorboard_url)

    #self.stop_tensorboard()

def runTraining(DATAPATH, EPOCH, BATCHSIZE):
  print("########### TRAINING ###########")
  training = Training(DATAPATH)
  training.train()
  training.evaluate()
  training.report()

if __name__ == '__main__':
    input_preprocessed_img_dir ="/home/lwecke/Datensätze/Datensatz_v1_50p_3reg/preprocessed_sorded_by_yaw/yaw_6.4285712242126465"
    output_bulktraining_path ="/home/lwecke/Datensätze/Datensatz_v1_50p_3reg/Bulktraining_Outputs/Einzeltraining"
    yaw_value = os.path.basename(input_preprocessed_img_dir)
    print(f"Training for {input_preprocessed_img_dir}...")
    # Initialize and train the model
    trainer = Training(input_preprocessed_img_dir,output_bulktraining_path + "/" + yaw_value)  # Or however you initialize your training object
    trainer.train()
    trainer.evaluate()
    trainer.report()