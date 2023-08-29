import random
import threading
import time

from silence_tensorflow import silence_tensorflow
silence_tensorflow()
from datetime import datetime
from modules.Utility.notificationBot import senden
import tensorflow as tf
import os
import sys
import cv2
import yaml
import pandas as pd
import numpy as np
from models.lenet5 import LeNet_baseline
from models.lenet5 import LeNet_drop_reg
from models.lenet5 import LeNet_Hypermodel
from models.lenet5 import LeNet_reduced
from tensorboard.plugins.hparams import api as hp

class Training:
  def __init__(self,DATAPATH,EPOCH,BATCHSIZE):
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
    self.trainings_laufzeit=0
    self.tuning_laufzeit = 0
    self.my_hypermodel = LeNet_Hypermodel()

    ######################################################
    #METHA PARAMETER WHICH CANT BE PLACED IN PARAMS.YAML
    p = os.path.abspath('../training')
    sys.path.insert(1, p)
    from modules.data_import_and_preprocessing.dataset_formation import DataParser, ImageDataExtractor, LabelExtractor, \
      DataSetCreator
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    self.physical_devices = tf.config.list_physical_devices('CPU')
    #tf.config.experimental.set_memory_growth(self.physical_devices[0], True)

    class CustomImageDataExtractor(ImageDataExtractor):
      def get_data(self, data_point):
        filename = data_point.path_to_data
        img = cv2.imread(filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.rotate_image(img, random.choice([0,90,180,270]))  # TODO: check if pictures actually rotate
        img_preprocessed = self.preprocess_image(img)
        return np.expand_dims(img_preprocessed, axis=-1)

      def rotate_image(self,image, angle):
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
        return result


    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)
    os.chdir('../')
    import yaml

    class PrettySafeLoader(yaml.SafeLoader):

      def construct_python_tuple(self, node):
        return tuple(self.construct_sequence(node))
    with open('./params.yaml', 'r') as stream:
      self.params = yaml.load(stream,Loader=PrettySafeLoader)
    start_time = datetime.now()
    if DATAPATH != "":
      path_preprocessed_images = DATAPATH
      self.path = DATAPATH+"/"+str(start_time)
    else:
      path_preprocessed_images = self.params['dest_path_preprocessed']
      self.path = self.params['output_path_dataset_training']+f"/output_training{datetime.now().strftime('%Y-%m-%d_%H-%M')}"
    if not os.path.exists(self.path):
      os.mkdir(self.path)
    os.chdir(self.path)
    print(str(self.params))
    if not self.params["training"]["model_baseline"]:
      print("DROPOUT MODEL IN USE")
      self.model=LeNet_drop_reg
    else:
      print("Reduced MODEL IN USE")
      self.model=LeNet_reduced
    try:
      if os.environ["TEST"]:
        self.model=LeNet_baseline
    except:
      pass
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

    os.chdir(self.path)
    print("Training Dir: "+os.getcwd())

    #Hyperparameter Tuned
    self.batch_size = self.params['training']['hp']['batch_size']
    if BATCHSIZE != -1:
      self.batch_size = int(BATCHSIZE)
    if  self.params["training"]["model_baseline"]: #baseline means, no regularization and no dropout
      self.dropout_rate =0 #wont be used for the model anyways. Just for consol printout and result.yaml
      self.regularization =0 #wont be used for the model anyways. Just for consol printout and result.yaml
    else:
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
    if EPOCH != -1:
      self.no_epochs = int(EPOCH)
    self.loss = self.params['training']['loss']
    self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learningrate)
    self.dataset_test_no_repeats = self.params['training']['dataset_test_no_repeats']

    #Fetch DATASET
    data_parser         = DataParser(path_preprocessed_images)
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
    self.tf_dataset_test = self.dataset_test.cast_tf_dataset().batch(self.batch_size)
    # Callback to stop training after no performance decrease
    self.callback_early_stopping = tf.keras.callbacks.EarlyStopping(patience=30,
                                                                    restore_best_weights=False,
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


  def hp_optimization(self):
    self.time_start = datetime.now()
    session_num = 0
    HP_BATCHSIZE        = hp.HParam('batch', hp.Discrete([1024]))
    HP_DROPOUT          = hp.HParam('drop', hp.Discrete([0.0,0.2,0.6]))
    HP_REGULAIZATION    = hp.HParam('reg', hp.Discrete([0.0,0.005,0.009]))

    hyperparameter = []
    hyperparameter.append(HP_REGULAIZATION),
    hyperparameter.append(HP_DROPOUT),
    hyperparameter.append(HP_BATCHSIZE)
    self.my_hypermodel.declare_hyperparameters(hyperparameter)
    self.best_acc = 0
    firstIteration=True
    for batchsize in HP_BATCHSIZE.domain.values:
      self.tf_dataset_val = self.dataset_val.cast_tf_dataset().batch(batchsize).prefetch(1)
      for regularization in HP_REGULAIZATION.domain.values:
        for dropout_rate in HP_DROPOUT.domain.values:
          if  dropout_rate==0.6 and regularization!=0: #skipps extreme combinations that dont make sense.
            continue
          if dropout_rate != 0 and regularization ==0.009: #skipps extreme combinations that dont make sense.
            continue
          hparams = {
            HP_REGULAIZATION: regularization,
            HP_DROPOUT: dropout_rate,
            HP_BATCHSIZE: batchsize,
          }

          run_name = "run-%d" % session_num
          print('--- Starting trial: %s' % run_name)
          #print("HIGHSCORE: "+"Acc="+str(self.best_acc)+" Batchsize="+str(self.best_batchsize)+" Dropout="+str(self.best_dropout_rate)+" Reg="+str(self.best_regularization)+" Name= "+str(self.best_exp_name))
          self.path_exp=self.path + "/hparam_tuning/experiment-"+ datetime.now().strftime("%Y%m%d-%H%M%S")

          #run
          with tf.summary.create_file_writer(self.path_exp).as_default():
            hp.hparams(hparams)  # record the values used in this trial
          print("Current: " + "Batchsize=" + str(hparams.get(self.my_hypermodel.getBatchsize()))
                + " Dropout=" + str(hparams.get(self.my_hypermodel.getDropout())) + " Reg=" + str(hparams.get(self.my_hypermodel.getRegularization())))
          self.hyperModel_built = self.my_hypermodel.build(hparams)
          self.lr_start = self.params['training']['learningrate']
          self.lr_max     = self.params['callbacks']['lr_schedule']['lr_max']
          self.lr_min     = self.params['callbacks']['lr_schedule']['lr_min']
          self.lr_ramp_ep = self.params['callbacks']['lr_schedule']['lr_ramp_ep']
          self.lr_sus_ep  = self.params['callbacks']['lr_schedule']['lr_sus_ep']
          self.lr_decay   = self.params['callbacks']['lr_schedule']['lr_decay']
          self.path_checkpoint = self.path + f"/checkpoints/{str(datetime.now())}"
          self.hyperModel_built.compile(loss=self.loss, optimizer=self.optimizer,
                                   metrics=[tf.keras.metrics.CategoricalAccuracy()])

          self.tb_callback = tf.keras.callbacks.TensorBoard(log_dir=self.path_exp,
                                                            histogram_freq=1,
                                                            write_graph=True,
                                                            write_images=True,
                                                            update_freq='epoch',
                                                            profile_batch=2,
                                                            embeddings_freq=1)
          if firstIteration:
            self.thread_tensorboard = threading.Thread(target=self.start_tensorboard_tuning)
            self.thread_tensorboard.start()
            time.sleep(5)
            firstIteration = False
          self.history = self.hyperModel_built.fit(self.tf_dataset_train,
                                                   epochs=int(self.no_epochs),
                                                   batch_size=batchsize,
                                                   steps_per_epoch=self.no_points_train // batchsize,
                                                   validation_data=self.tf_dataset_val,
                                                   validation_steps=self.no_points_val // batchsize,
                                                   callbacks=[self.tb_callback,
                                                              hp.KerasCallback(self.path_exp, hparams),
                                                              self.callback_early_stopping])  # Run with 1 epoch to speed things up for demo purposes

          #if (accuracy>self.best_acc):
          #  self.best_acc = accuracy
          #  model_best=self.hyperModel_built
          #  self.best_regularization = regularization
          #  self.best_dropout_rate   = dropout_rate
          #  self.best_batchsize      = batchsize
          #  self.best_exp_name=self.path_exp
          session_num += 1

    #self.model_built = model_best
    #td =datetime.now() - self.time_start
    #days = td.days
    #hours, remainder = divmod(td.seconds, 3600)
    #minutes, seconds = divmod(remainder, 60)
    #self.trainings_laufzeit  = f"Days:{days}, Hours:{hours}, Minutes:{minutes}"

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
                                                      write_images=True,
                                                      update_freq='epoch',
                                                      profile_batch=2,
                                                      embeddings_freq=1)
    self.thread_tensorboard = threading.Thread(target=self.start_tensorboard_training)
    self.thread_tensorboard.start()
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
    tf.data.Dataset.save(self.tf_dataset_test, self.path +'/tf_dataset_test.tf')
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
  def start_tensorboard_training(self):
    print(f"starting tensorboard in dir: {os.getcwd()}/tensorboard")
    print(os.system(f"tensorboard dev upload \
    --logdir ./tensorboard \
    --name 'Batchsize={self.params['training']['hp']['batch_size']} Dropout={self.params['training']['hp']['dropout']} Reg={self.params['training']['hp']['regularization']} Lr={self.params['training']['learningrate']}' \
    --description 'Experiment path: {self.path}'"))
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

  def getTensorboardLinkFromLogFile(self):
    tb_link = "error"
    try:
      with open(
        f"{self.params['path_logs']}/terminalOutput{os.environ['LOGFILE']}.log") as log_fh:
        for line in log_fh:
          if line.__contains__("https"):  # you might want a better check here
            tb_link = "https" + line.split("https", 2)[1]
            break
      print(tb_link)

    except:
      print(f"ERROR IN PARSING OR LOADING LOG FILE IN {self.params['path_logs']}/terminalOutput")
      print(
        "ENVIROMENT VARIABLE LOGFILE must be set. e.g. 4. referes to log file and is used to run different runconfigurations in parallel while still writing to log files.")
      print("MAXMIUM LOGFILE NUMBER IS 5.")
    return tb_link
  def evaluate(self):
    #MODEL PREDICTS
    print("#################### EVALUTATION ####################")

    self.results = self.model_built.evaluate(self.tf_dataset_test, verbose=True)
    print(f"Best Weights Evaluation: loss {self.results[0]}, acc {self.results[1]}")
    os.chdir(self.path)
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

    import matplotlib.pyplot as plt
    from sklearn.preprocessing import LabelBinarizer
    from sklearn.metrics import roc_curve, auc, roc_auc_score

    target = ['Dispersed', 'Transition', 'Loaded', 'Flooded']
    # set plot figure size
    fig, c_ax = plt.subplots(1, 1, figsize=(12, 8))

    # function for scoring roc auc score for multi-class

    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test_idx)
    y_pred = lb.transform(predictions_idx)
    for (idx, c_label) in enumerate(target):
      fpr, tpr, thresholds = roc_curve(y_test[:,idx], y_pred[:,idx])
      c_ax.plot(fpr, tpr, label='%s (AUC:%0.2f)' % (c_label, auc(fpr, tpr)))
    c_ax.plot(fpr, fpr, 'b-', label='Random Guessing')
    self.auc=roc_auc_score(y_test, y_pred, average="macro")
    print('ROC AUC score:',self.auc)

    c_ax.legend()
    c_ax.set_xlabel('False Positive Rate')
    c_ax.set_ylabel('True Positive Rate')
    plt.show()
##########y
    # Compute ROC curve and ROC area for each class

    from sklearn.preprocessing import label_binarize

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    y = label_binarize(y_test, classes=["Dispersed", "Transition", "Loaded","Flooded"])
    n_classes=4
    print("y_test "+str(y_test)+" y_pred "+str(y_pred))
    for (i, classname) in enumerate(target):
      fpr[i], tpr[i], thresholds = roc_curve(y_test[:, i], y_pred[:, i])
      roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_pred.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    plt.figure()
    lw = 2
    plt.plot(
      fpr[2],
      tpr[2],
      color="darkorange",
      lw=lw,
      label="ROC curve (area = %0.2f)" % roc_auc[2],
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver operating characteristic example")
    plt.legend(loc="lower right")
    plt.show()

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
      mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure()
    plt.plot(
      fpr["micro"],
      tpr["micro"],
      label="micro-average ROC curve (area = {0:0.2f})".format(roc_auc["micro"]),
      color="deeppink",
      linestyle=":",
      linewidth=2,
    )

    plt.plot(
      fpr["macro"],
      tpr["macro"],
      label="macro-average ROC curve (area = {0:0.2f})".format(roc_auc["macro"]),
      color="navy",
      linestyle=":",
      linewidth=2,
    )
    import matplotlib.pyplot as plt
    from itertools import cycle
    colors = cycle(["aqua", "darkorange", "green","blue"])
    for i, color in zip(range(n_classes), colors):
      plt.plot(
        fpr[i],
        tpr[i],
        color=color,
        lw=lw,
        label=f"ROC curve of class {target[i]} "+"(area = {1:0.2f})".format(i, roc_auc[i]),
      )

    plt.plot([0, 1], [0, 1], "k--", lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(self.path)
    plt.legend(loc="lower right")
    plt.savefig(f"{self.path}/Roc.png")
    self.model_built.save(self.path+'/cnn_acc'+str(round(self.results[1]*100,0))+"_auc"+str(round(self.auc, 1))+'.h5')
    plt.show()
  def report(self):
    print("########## REPORT ##########")
    os.chdir(self.path)
    # convert the history.history dict to a pandas DataFrame:
    hist_df = pd.DataFrame(self.history.history)
    # save to json:
    hist_json_file = './training_history.json'
    with open(hist_json_file, mode='w') as f:
      hist_df.to_json(f)
    print("RESULTS: "+str(self.results))
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

    with open(self.path+'/results.yaml', 'w') as outfile:
      yaml.dump(training_info, outfile, default_flow_style=False, sort_keys=False)
    with open(self.path+'/params_for_this_training.yaml', 'w') as outfile:
        yaml.dump(self.params, outfile, default_flow_style=False, sort_keys=False)
    link=self.getTensorboardLinkFromLogFile()
    try:
      with open(self.path+'/tensorboard_link.txt', 'w') as f:
        f.write(link)
    except FileNotFoundError:
      print(f"The {self.path} directory does not exist")
    if self.tuning:
      senden("Tuning Finished! "    + "ACC: " + str(self.results[1]) + "\n Tuning_Laufzeit: "      + str(self.tuning_laufzeit)+" Confusion_matrix:"+ str(self.conf_mat)+" reg: "+str(self.best_regularization)+" batch:"+str(self.best_batchsize)+" drop:"+str(self.best_dropout_rate))
    else:
      senden("Training Finished! "  + "ACC: " + str(self.results[1]) + "\n Trainings_Laufzeit: " + str(self.trainings_laufzeit)+" Confusion_matrix:"+ self.parseConfMat(self.conf_mat)+"\n AUC: "+str(self.auc) +"\n Path: "+self.path+ "\n Tensorboard: "+link)


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