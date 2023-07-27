from silence_tensorflow import silence_tensorflow
import time

from modules.Utility.CustomImageDataExtractor import CustomImageDataExtractor
from modules.Utility.PrettySafeLoader import PrettySafeLoader
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.preprocessing import LabelBinarizer
from itertools import cycle

from modules.data_import_and_preprocessing.dataset_formation import DataParser, LabelExtractor, DataSetCreator

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
import os
import subprocess
import re
import queue
import threading


class Training:

  def __init__(self, DATAPATH, EPOCH, BATCHSIZE):
    # Initialize variables and environment
    self.initialize_variables()
    self.setup_environment()

    # Load the hyperparameters
    self.load_hyperparams(DATAPATH)

    # Set model
    self.set_model()

    # Set more parameters
    self.set_more_params(EPOCH, BATCHSIZE)

    # Create and split dataset
    self.create_dataset(DATAPATH)
    self.split_dataset()

    # Set callbacks
    self.set_callbacks()

  def initialize_variables(self):
    self.auc = None
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
    self.trainings_laufzeit = 0
    self.tuning_laufzeit = 0
    self.my_hypermodel = LeNet_Hypermodel()
    self.physical_devices = tf.config.list_physical_devices('CPU')
    # tf.config.experimental.set_memory_growth(self.physical_devices[0], True)

  def setup_environment(self):
    p = os.path.abspath('../training')
    sys.path.insert(1, p)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)
    os.chdir('../')

  def load_hyperparams(self, DATAPATH):
    # Code to load params.yaml
    PrettySafeLoader.add_constructor(u'tag:yaml.org,2002:python/tuple', PrettySafeLoader.construct_python_tuple)
    with open('./params.yaml', 'r') as stream:
      self.params = yaml.load(stream, Loader=PrettySafeLoader)
    start_time = datetime.now()
    if DATAPATH != "":
      self.path_preprocessed_images = DATAPATH
      self.path = DATAPATH + "/" + str(start_time)
    else:
      self.path_preprocessed_images = self.params['dest_path_preprocessed']
      self.path = self.params[
                    'output_path_dataset_training'] + f"/output_training{datetime.now().strftime('%Y-%m-%d_%H-%M')}"
    os.mkdir(self.path)
    os.chdir(self.path)
    print(str(self.params))

  def set_model(self):
    if not self.params["training"]["model_baseline"]:
      print("DROPOUT MODEL IN USE")
      self.model = LeNet_drop_reg
    else:
      print("Reduced MODEL IN USE")
      self.model = LeNet_reduced
    try:
      if os.environ["TEST"]:
        self.model = LeNet_baseline
    except:
      pass

  def set_more_params(self, EPOCH, BATCHSIZE):
    if self.params["run_tuner"]:
      self.tuning = True
    self.lr_start = self.params['training']['learningrate']
    self.lr_max = self.params['callbacks']['lr_schedule']['lr_max']
    self.lr_min = self.params['callbacks']['lr_schedule']['lr_min']
    self.lr_ramp_ep = self.params['callbacks']['lr_schedule']['lr_ramp_ep']
    self.lr_sus_ep = self.params['callbacks']['lr_schedule']['lr_sus_ep']
    self.lr_decay = self.params['callbacks']['lr_schedule']['lr_decay']

    # Parameter
    self.picture_width = self.params['preprocessing']['picture_width']
    self.picture_height = self.params['preprocessing']['picture_hight']
    self.no_classes = self.params['preprocessing']['no_classes']

    os.chdir(self.path)
    print("Training Dir: " + os.getcwd())

    # Hyperparameter Tuned
    self.batch_size = self.params['training']['hp']['batch_size']
    if BATCHSIZE != -1:
      self.batch_size = int(BATCHSIZE)
    if self.params["training"]["model_baseline"]:  # baseline means, no regularization and no dropout
      self.dropout_rate = 0  # won't be used for the model anyways. Just for console printout and result.yaml
      self.regularization = 0  # won't be used for the model anyways. Just for console printout and result.yaml
    else:
      self.dropout_rate = self.params['training']['hp']['dropout']
      self.regularization = self.params['training']['hp']['regularization']

    self.best_regularization = 0
    self.best_dropout_rate = 0
    self.best_batchsize = 0
    self.best_exp_name = ""
    # Parameters not Tuned
    self.learningrate = self.params['training']['learningrate']
    self.lr_start = self.learningrate
    self.callback_learningrate_schedule = tf.keras.callbacks.LearningRateScheduler(self.lrfn, verbose=True)

    self.no_epochs = self.params['training']['no_epochs']
    if EPOCH != -1:
      self.no_epochs = int(EPOCH)
    self.loss = self.params['training']['loss']
    self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learningrate)
    self.dataset_test_no_repeats = self.params['training']['dataset_test_no_repeats']

  def create_dataset(self, path_preprocessed_images):
    data_parser = DataParser(path_preprocessed_images)
    image_data_extractor = CustomImageDataExtractor((self.picture_width, self.picture_height, 1))
    label_extractor = LabelExtractor(no_classes=self.no_classes)  # number of classes
    self.dataset = DataSetCreator(data_parser, image_data_extractor, label_extractor, no_repeats=self.no_epochs)

  def split_dataset(self):
    self.no_points = len(self.dataset)
    if self.tuning:
      self.no_points_train = int(self.no_points * self.params['tuning']['no_points'])
    print("############ INITIALIZATION DONE ############")

  def load_yaml_params(self):
    PrettySafeLoader.add_constructor(
      u'tag:yaml.org,2002:python/tuple',
      PrettySafeLoader.construct_python_tuple)
    with open('../params.yaml', 'r') as stream:
      return yaml.load(stream, Loader=PrettySafeLoader)

  def set_callbacks(self):
    self.callback_learningrate_schedule = tf.keras.callbacks.LearningRateScheduler(self.lrfn, verbose=True)
    self.callback_early_stopping = tf.keras.callbacks.EarlyStopping(patience=30, restore_best_weights=False,
                                                                    monitor='categorical_accuracy', min_delta=0.01,
                                                                    mode='auto', verbose=1, baseline=None)
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
        "--description", f"Experiment path: {self.path}"
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
    os.chdir(self.path)
    #print(os.getcwd())
    if not os.path.exists("tensorboard"):
      os.makedirs("tensorboard")
    self.tb_callback = tf.keras.callbacks.TensorBoard(log_dir=self.path + "/tensorboard",
                                                      histogram_freq=1,
                                                      write_graph=True,
                                                      write_images=False,
                                                      update_freq='epoch',
                                                      profile_batch=2,
                                                      embeddings_freq=1)
    # Initialize an empty Queue to pass messages from the output processing thread to the main thread
    self.start_threads()

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
    print("Output Path "+self.path)
    print("Input Data Path "+self.path_preprocessed_images)

    self.path_checkpoint = self.path + f"/checkpoints/B{str(self.batch_size).split('.',1)[0]}_D{str(self.dropout_rate).split('.',1)[0]}_R{str(self.regularization).split('.',1)[0]}"
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
      filepath=self.path_checkpoint,
      save_weights_only=True,
      monitor='val_categorical_accuracy',
      mode='max',
      save_best_only=True,
      verbose=0)
    callbacks=self.callback_toogled
    callbacks.append(self.tb_callback)
    callbacks.append(model_checkpoint_callback)
    tf.config.run_functions_eagerly(True)
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
    #tf.data.Dataset.save(self.tf_dataset_test, self.path +'/tf_dataset_test.tf')
    self.model_built.load_weights(self.path_checkpoint)
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
    os.chdir(self.path)
    print(f"starting tensorboard in dir: {os.getcwd()}/hparam_tuning")
    print(os.system(f"tensorboard dev upload \
    --logdir ./hparam_tuning \
    --name 'Tuning {self.params['path_dataset']}' \
    --description 'Experiment path: {self.path}'"))

  def loadModel(self, path_model):
    self.tf_dataset_test= tf.data.experimental.load('../cache/ds_test.tf')
    self.model_built=tf.keras.models.load_model(f"{self.path}/{path_model}")
    self.evaluate_only=True

  def evaluate(self):
    print("#################### EVALUTATION ####################")
    self.results = self.model_built.evaluate(self.tf_dataset_test, verbose=True)
    print(f"Best Weights Evaluation: loss {self.results[0]}, acc {self.results[1]}")
    os.chdir(self.path)
    x_test, y_test = self.get_test_samples()
    if self.tuning:
      print(
        f"BEST TUNING HYPERPARAMETERS: Batchsize={self.best_batchsize} Reg={self.best_regularization} Drop={self.best_dropout_rate}"
      )
    print("### PREDICT TEST DATASET FOR CONSUFION MATRIX ###")

    predictions = [self.model_built.predict_on_batch(x) for x in x_test]
    self.conf_mat = self.get_confusion_matrix(predictions, y_test)

    print("Confusion Matrix: " + self.parseConfMat(self.conf_mat))

    self.plot_roc_curves(y_test, predictions)

    self.model_built.save(f"{self.path}/cnn_acc{round(self.results[1] * 100, 0)}_auc{round(self.auc, 1)}.h5")

  def get_test_samples(self):
    x_test, y_test = [], []
    for image, label in list(self.tf_dataset_test.unbatch().as_numpy_iterator()):
      tf_image = tf.convert_to_tensor(image)
      tf_image_batched = tf.expand_dims(tf_image, axis=0)
      x_test.append(tf_image_batched)
      y_test.append(label)

    return x_test, tf.argmax(y_test, axis=1)

  def get_confusion_matrix(self, predictions, y_test):
    predictions_mat = np.vstack(predictions)
    predictions_idx = np.argmax(predictions_mat, axis=1)

    return tf.math.confusion_matrix(y_test, predictions_idx)

  def plot_roc_curves(self, y_test, predictions):
    target = ['Dispersed', 'Loaded', 'Flooded']
    fig, c_ax = plt.subplots(1, 1, figsize=(12, 8))

    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test_bin = lb.transform(y_test)
    y_pred_bin = lb.transform(np.argmax(np.vstack(predictions), axis=1))

    for (idx, c_label) in enumerate(target):
      fpr, tpr, _ = roc_curve(y_test_bin[:, idx], y_pred_bin[:, idx])
      c_ax.plot(fpr, tpr, label='%s (AUC:%0.2f)' % (c_label, auc(fpr, tpr)))

    self.auc = roc_auc_score(y_test_bin, y_pred_bin, average="macro")
    print('ROC AUC score:', self.auc)

    c_ax.plot([0, 1], [0, 1], 'b-', label='Random Guessing')
    c_ax.legend()
    c_ax.set_xlabel('False Positive Rate')
    c_ax.set_ylabel('True Positive Rate')
    plt.show()

    self.plot_individual_roc_curves(y_test_bin, y_pred_bin, target)

  def plot_individual_roc_curves(self, y_test_bin, y_pred_bin, target):
    fpr, tpr, roc_auc = self.compute_roc_auc(y_test_bin, y_pred_bin, target)

    # Plot all ROC curves
    plt.figure()
    lw = 2
    colors = cycle(["aqua", "darkorange", "green", "blue"])
    for i, color in zip(range(len(target)), colors):
      plt.plot(fpr[i], tpr[i], color=color, lw=lw,
               label=f"ROC curve of class {target[i]} (area = {roc_auc[i]:0.2f})")
    plt.plot([0, 1], [0, 1], "k--", lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(self.path)
    plt.legend(loc="lower right")
    plt.savefig(f"{self.path}/Roc.png")
    plt.show()

  def compute_roc_auc(self, y_test, y_pred, target):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for (i, classname) in enumerate(target):
      fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred[:, i])
      roc_auc[i] = auc(fpr[i], tpr[i])
    return fpr, tpr, roc_auc

  def report(self):
    print("##############################  \n ########## REPORT ########## \n ##############################")
    os.chdir(self.path)
    self.save_history()
    self.print_results()
    self.print_tensorboard_url()
    self.save_training_info()
    self.save_params()
    self.save_tensorboard_link()
    self.send_notification()
    self.stop_tensorboard()

  def save_history(self):
    hist_df = pd.DataFrame(self.history.history)
    hist_json_file = './training_history.json'
    with open(hist_json_file, mode='w') as f:
      hist_df.to_json(f)

  def print_results(self):
    print("RESULTS: " + str(self.results))

  def print_tensorboard_url(self):
    time.sleep(10)
    tensorboard_url_message = "Tensorboard URL: " + self.tensorboard_url if hasattr(self,
                                                                                    'tensorboard_url') else "TensorBoard URL not yet available."
    print(tensorboard_url_message)

  def save_training_info(self):
    training_info = self.prepare_training_info()
    with open(self.path + '/results.yaml', 'w') as outfile:
      yaml.dump(training_info, outfile, default_flow_style=False, sort_keys=False)

  def prepare_training_info(self):
    return {
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

  def save_params(self):
    with open(self.path + '/params_for_this_training.yaml', 'w') as outfile:
      yaml.dump(self.params, outfile, default_flow_style=False, sort_keys=False)

  def save_tensorboard_link(self):
    try:
      with open(self.path + '/tensorboard_link.txt', 'w') as f:
        f.write(self.tensorboard_url)
    except FileNotFoundError:
      print(f"The {self.path} directory does not exist")

  def send_notification(self):
    if self.tuning:
      senden("Tuning Finished! " + "ACC: " + str(self.results[1]) + "\n Tuning_Laufzeit: " + str(
        self.tuning_laufzeit) + " Confusion_matrix:" + str(self.conf_mat) + " reg: " + str(
        self.best_regularization) + " batch:" + str(self.best_batchsize) + " drop:" + str(self.best_dropout_rate))
    else:
      senden("Training Finished! " + "ACC: " + str(self.results[1]) + "\n Trainings_Laufzeit: " + str(
        self.trainings_laufzeit) + " Confusion_matrix:" + self.parseConfMat(self.conf_mat) + "\n AUC: " + str(
        self.auc) + "\n Path: " + self.path + "\n Tensorboard: " + self.tensorboard_url)


def runTuner(DATAPATH,EPOCH,BATCHSIZE):
  print("########### TUNING ###########")
  training = Training(DATAPATH, EPOCH,BATCHSIZE)
  training.hp_optimization()

def runTraining(DATAPATH, EPOCH, BATCHSIZE):
  print("########### TRAINING ###########")
  training = Training(DATAPATH, EPOCH, BATCHSIZE)
  training.train()
  training.evaluate()
  training.report()