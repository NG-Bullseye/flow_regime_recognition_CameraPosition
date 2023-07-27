import threading

from silence_tensorflow import silence_tensorflow
silence_tensorflow()
from datetime import datetime
from modules.Utility.notificationBot import senden
import tensorflow as tf
import os
import yaml
import pandas as pd
import numpy as np
from models.lenet5 import LeNet_baseline
from models.lenet5 import LeNet_drop_reg
from models.lenet5 import LeNet_Hypermodel
from models.lenet5 import LeNet_reduced
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer, label_binarize
from sklearn.metrics import roc_curve, auc, roc_auc_score
from itertools import cycle
class Training:

  def __init__(self, DATAPATH, EPOCH, BATCHSIZE):
    self.auc, self.path_exp, self.evaluate_only, self.tuning, self.tensorboard_url = None, None, False, False, ""
    self.LEARNRATE_SCHEDULE, self.callback_toogled, self.EARLYSTOP, self.METRIC = False, [], False, 'categorical_accuracy'
    self.best_acc, self.hyperModel_built, self.dropout_rate, self.trainings_laufzeit, self.tuning_laufzeit = 0, None, 0, 0, 0
    self.my_hypermodel = LeNet_Hypermodel()

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    self.physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(self.physical_devices[0], True)

    # Replacing print statements with single line comments
    # Loading params
    with open('./params.yaml', 'r') as stream:
      self.params = yaml.load(stream, Loader=PrettySafeLoader)
    start_time = datetime.now()
    path_preprocessed_images = (DATAPATH + "/preprocessed_images" + str(
      self.params['preprocessing']['picture_width'])) if DATAPATH != "" else (
              self.params['path_dataset'] + "/preprocessed_images" + str(self.params['preprocessing']['picture_width']))
    self.path = DATAPATH + "/" + str(start_time) if DATAPATH != "" else self.params['path_dataset'] + "/" + str(
      start_time)

    # Setup model
    if os.getenv("TEST", None):
      self.model = LeNet_baseline
    else:
      self.model = LeNet_drop_reg if not self.params["training"]["model_baseline"] else LeNet_reduced

    # Removed print statements

    self.tuning = True if self.params["run_tuner"] else False
    self.lr_start, self.lr_max, self.lr_min = self.params['training']['learningrate'], \
    self.params['callbacks']['lr_schedule']['lr_max'], self.params['callbacks']['lr_schedule']['lr_min']
    self.lr_ramp_ep, self.lr_sus_ep, self.lr_decay = self.params['callbacks']['lr_schedule']['lr_ramp_ep'], \
    self.params['callbacks']['lr_schedule']['lr_sus_ep'], self.params['callbacks']['lr_schedule']['lr_decay']
    self.batch_size = int(BATCHSIZE) if BATCHSIZE != -1 else self.params['training']['hp']['batch_size']
    self.dropout_rate = 0 if self.params["training"]["model_baseline"] else self.params['training']['hp']['dropout']
    self.regularization = 0 if self.params["training"]["model_baseline"] else self.params['training']['hp'][
      'regularization']
    self.learningrate, self.optimizer = self.params['training']['learningrate'], tf.keras.optimizers.Adam(
      learning_rate=self.learningrate)
    self.no_epochs = int(EPOCH) if EPOCH != -1 else self.params['training']['no_epochs']
    self.dataset_test_no_repeats, self.loss = self.params['training']['dataset_test_no_repeats'], \
    self.params['training']['loss']
    self.callback_learningrate_schedule = tf.keras.callbacks.LearningRateScheduler(self.lrfn, verbose=True)

    # Fetch and split dataset
    # Removed print statements

    # Prepare tf datasets
    self.tf_dataset_train = self.dataset_train.cast_tf_dataset().batch(self.batch_size).prefetch(1)
    self.tf_dataset_val = self.dataset_val.cast_tf_dataset().batch(self.batch_size).prefetch(1)
    self.tf_dataset_test = self.dataset_test.cast_tf_dataset().batch(self.batch_size)

    # Callbacks
    self.callback_early_stopping = tf.keras.callbacks.EarlyStopping(patience=30, restore_best_weights=False,
                                                                    monitor='categorical_accuracy', min_delta=0.01,
                                                                    mode='auto', verbose=1, baseline=None)
    if self.params['callbacks']['earlystop']:
      self.callback_toogled.append(self.callback_early_stopping)
    if self.params['callbacks']['lr_schedule']['enabled']:
      self.callback_toogled.append(self.callback_learningrate_schedule)

  def train(self):
    self.prepare_tensorboard()

    self.model_built = self.create_and_compile_model()

    self.print_training_configuration()

    self.path_checkpoint = self.create_model_checkpoint_path()
    callbacks = self.prepare_callbacks()

    self.history = self.fit_model(callbacks)

    self.save_test_data_and_weights()

    if self.params['callbacks']['earlystop']:
      print("EARLY STOPPED AT: " + str(self.callback_early_stopping.stopped_epoch))

  def prepare_tensorboard(self):
    os.chdir(self.path)
    if not os.path.exists("tensorboard"):
      os.makedirs("tensorboard")
    self.tb_callback = tf.keras.callbacks.TensorBoard(log_dir=self.path + "/tensorboard",
                                                      histogram_freq=1,
                                                      write_graph=True,
                                                      write_images=True,
                                                      update_freq='epoch',
                                                      profile_batch=2,
                                                      embeddings_freq=1)
    threading.Thread(target=self.start_tensorboard_training).start()
    self.time_start = datetime.now()

  def create_and_compile_model(self):
    model = self.model.build(self.params['datashape'], self.params['labelshape'],
                             dropout_rate=self.dropout_rate, regularization=self.regularization)
    model.compile(loss=self.loss, optimizer=self.optimizer, metrics=[tf.keras.metrics.CategoricalAccuracy()])
    model.summary()
    return model

  def print_training_configuration(self):
    print(f"epochs: {self.no_epochs}\nbatch_size: {self.batch_size}\ndropout: {self.dropout_rate}"
          f"\nregularization: {self.regularization}\nno_points_train: {self.no_points_train}\nno_points_val: {self.no_points_val}")

  def create_model_checkpoint_path(self):
    return self.path + f"/checkpoints/B{str(self.batch_size).split('.', 1)[0]}_D{str(self.dropout_rate).split('.', 1)[0]}_R{str(self.regularization).split('.', 1)[0]}"

  def prepare_callbacks(self):
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
      filepath=self.path_checkpoint,
      save_weights_only=True,
      monitor='val_categorical_accuracy',
      mode='max',
      save_best_only=True,
      verbose=0)
    return self.callback_toogled + [self.tb_callback, model_checkpoint_callback]

  def fit_model(self, callbacks):
    return self.model_built.fit(self.tf_dataset_train,
                                epochs=self.no_epochs,
                                batch_size=self.batch_size,
                                steps_per_epoch=self.no_points_train // self.batch_size,
                                validation_data=self.tf_dataset_val,
                                validation_steps=self.no_points_val // self.batch_size,
                                callbacks=callbacks)

  def save_test_data_and_weights(self):
    td = datetime.now() - self.time_start
    days, remainder = divmod(td.days * 24 * 3600 + td.seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, _ = divmod(remainder, 60)
    self.trainings_laufzeit = f"Days:{days}, Hours:{hours}, Minutes:{minutes}"
    print("Saving test Data Split tf_dataset_test..")
    tf.data.Dataset.save(self.tf_dataset_test, self.path + '/tf_dataset_test.tf')
    self.model_built.load_weights(self.path_checkpoint)

  def parseConfMat(self, matrix):
    return str(matrix).split('(', 1)[1].split(',', 1)[0]

  def lrfn(self, epoch):
    lr = self.lr_start if epoch < self.lr_ramp_ep else self.lr_max if epoch < self.lr_ramp_ep + self.lr_sus_ep else (
                                                                                                                              self.lr_max - self.lr_min) * self.lr_decay ** (
                                                                                                                              epoch - self.lr_ramp_ep - self.lr_sus_ep) + self.lr_min
    print(f"Updated Learningrate: {lr}")
    return lr

  def start_tensorboard(self, folder, name):
    os.chdir(self.path)
    print(f"starting tensorboard in dir: {os.getcwd()}/{folder}")
    print(os.system(
      f"tensorboard dev upload --logdir ./{folder} --name '{name}' --description 'Experiment path: {self.path}'"))

  def start_tensorboard_training(self):
    self.start_tensorboard("tensorboard",
                           f"Batchsize={self.params['training']['hp']['batch_size']} Dropout={self.params['training']['hp']['dropout']} Reg={self.params['training']['hp']['regularization']} Lr={self.params['training']['learningrate']}'")

  def start_tensorboard_tuning(self):
    self.start_tensorboard("hparam_tuning", f"Tuning {self.params['path_dataset']}'")

  def loadModel(self, path_model):
    self.tf_dataset_test = tf.data.experimental.load('../cache/ds_test.tf')
    self.model_built = tf.keras.models.load_model(f"{self.path}/{path_model}")
    self.evaluate_only = True

  def getTensorboardLinkFromLogFile(self):
    try:
      with open(f"{self.params['path_logs']}/terminalOutput{os.environ['LOGFILE']}.log") as log_fh:
        tb_link = next(("https" + line.split("https", 2)[1] for line in log_fh if "https" in line), "error")
    except:
      tb_link = "error"
      print(f"ERROR IN PARSING OR LOADING LOG FILE IN {self.params['path_logs']}/terminalOutput")
      print(
        "ENVIROMENT VARIABLE LOGFILE must be set. e.g. 4. referes to log file and is used to run different runconfigurations in parallel while still writing to log files.")
      print("MAXIMUM LOGFILE NUMBER IS 5.")
    print(tb_link)
    return tb_link

  def evaluate(self):
    print("#################### EVALUATION ####################")
    self.results = self.model_built.evaluate(self.tf_dataset_test, verbose=True)
    print(f"Best Weights Evaluation: loss {self.results[0]}, acc {self.results[1]}")
    os.chdir(self.path)

    x_test, y_test = zip(*[(tf.expand_dims(tf.convert_to_tensor(image), axis=0), label) for image, label in
                           self.tf_dataset_test.unbatch().as_numpy_iterator()])
    y_test_idx = tf.argmax(y_test, axis=1)

    if self.tuning: print(
      f"BEST TUNING HYPERPARAMETERS: Batchsize={self.best_batchsize} Reg={self.best_regularization} Drop={self.best_dropout_rate}")
    print("### PREDICT TEST DATASET FOR CONSUFION MATRIX ###")

    predictions_mat = np.vstack([self.model_built.predict_on_batch(x) for x in x_test])
    predictions_idx = np.argmax(predictions_mat, axis=1)
    self.conf_mat = tf.math.confusion_matrix(y_test_idx, predictions_idx)
    print("Confusion Matrix: " + self.parseConfMat(self.conf_mat))

    target = ['Dispersed', 'Transition', 'Loaded', 'Flooded']
    fig, c_ax = plt.subplots(1, 1, figsize=(12, 8))

    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test_idx)
    y_pred = lb.transform(predictions_idx)

    for (idx, c_label) in enumerate(target):
      fpr, tpr, _ = roc_curve(y_test[:, idx], y_pred[:, idx])
      c_ax.plot(fpr, tpr, label='%s (AUC:%0.2f)' % (c_label, auc(fpr, tpr)))

    c_ax.plot(fpr, fpr, 'b-', label='Random Guessing')
    self.auc = roc_auc_score(y_test, y_pred, average="macro")
    print('ROC AUC score:', self.auc)

    c_ax.legend()
    c_ax.set_xlabel('False Positive Rate')
    c_ax.set_ylabel('True Positive Rate')
    plt.show()

    y = label_binarize(y_test, classes=target)
    n_classes = 4
    fpr, tpr, roc_auc = {}, {}, {}

    for i in range(n_classes):
      fpr[i], tpr[i], _ = roc_curve(y[:, i], y_pred[:, i])
      roc_auc[i] = auc(fpr[i], tpr[i])

    fpr["micro"], tpr["micro"], _ = roc_curve(y.ravel(), y_pred.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.mean([np.interp(all_fpr, fpr[i], tpr[i]) for i in range(n_classes)], axis=0)
    fpr["macro"], tpr["macro"], roc_auc["macro"] = all_fpr, mean_tpr, auc(fpr["macro"], tpr["macro"])

    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"], label=f"micro-average ROC curve (area = {roc_auc['micro']:.2f})",
             color="deeppink", linestyle=":", linewidth=2)
    plt.plot(fpr["macro"], tpr["macro"], label=f"macro-average ROC curve (area = {roc_auc['macro']:.2f})", color="navy",
             linestyle=":", linewidth=2)
    for i, color in zip(range(n_classes), cycle(["aqua", "darkorange", "green", "blue"])):
      plt.plot(fpr[i], tpr[i], color=color, lw=2, label=f"ROC curve of class {target[i]} (area = {roc_auc[i]:.2f})")

    plt.plot([0, 1], [0, 1], "k--", lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(self.path)
    plt.legend(loc="lower right")
    plt.savefig(f"{self.path}/Roc.png")
    self.model_built.save(
      self.path + '/cnn_acc' + str(round(self.results[1] * 100, 0)) + "_auc" + str(round(self.auc, 1)) + '.h5')
    plt.show()

  def report(self):
    print("########## REPORT ##########")
    os.chdir(self.path)
    pd.DataFrame(self.history.history).to_json('./training_history.json')
    print("RESULTS: " + str(self.results))

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

    for filename, data in zip(['results.yaml', 'params_for_this_training.yaml'], [training_info, self.params]):
      with open(self.path + '/' + filename, 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False, sort_keys=False)

    try:
      with open(self.path + '/tensorboard_link.txt', 'w') as f:
        f.write(self.getTensorboardLinkFromLogFile())
    except FileNotFoundError:
      print(f"The {self.path} directory does not exist")

    message = "Tuning Finished! " if self.tuning else "Training Finished! "
    message += "ACC: " + str(self.results[1]) + "\n" + (
      "Tuning_Laufzeit: " if self.tuning else "Trainings_Laufzeit: ") + str(
      self.tuning_laufzeit if self.tuning else self.trainings_laufzeit) + " Confusion_matrix:" + str(
      self.conf_mat if self.tuning else self.parseConfMat(self.conf_mat))
    if not self.tuning:
      message += "\n AUC: " + str(
        self.auc) + "\n Path: " + self.path + "\n Tensorboard: " + self.getTensorboardLinkFromLogFile()
    senden(message)

  def runTuner(DATAPATH, EPOCH, BATCHSIZE):
    print("########### TUNING ###########")
    training = Training(DATAPATH, EPOCH, BATCHSIZE)
    training.hp_optimization()

  def runTraining(DATAPATH, EPOCH, BATCHSIZE):
    print("########### TRAINING ###########")
    training = Training(DATAPATH, EPOCH, BATCHSIZE)
    for method in [training.train, training.evaluate, training.report]:
      method()
