#from silence_tensorflow import silence_tensorflow
#silence_tensorflow()
import os
import yaml
import sys
import tensorflow as tf

import modules.training_for_each_subdir
from modules.data_import_and_preprocessing import copyPasteBigDataFromUsbStick, batch_data_preprocessing, label_balance_PicturesByRpmAndGasflow
from modules import model_training

if __name__ == '__main__':
  print(tf.__version__)

  if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
  else:
    print("Please install GPU version of TF")

  p = os.path.abspath('.')
  sys.path.insert(1, p)
  os.environ["CUDA_VISIBLE_DEVICES"] = "0"

  physical_devices = tf.config.list_physical_devices('CPU')
  #tf.config.experimental.set_memory_growth(physical_devices[0], True)
  print("RUNNING STARTER WITH")
  class PrettySafeLoader(yaml.SafeLoader):
    def construct_python_tuple(self, node):
      return tuple(self.construct_sequence(node))
  PrettySafeLoader.add_constructor(
    u'tag:yaml.org,2002:python/tuple',
    PrettySafeLoader.construct_python_tuple)
  with open('./params.yaml', 'r') as stream:
    params = yaml.load(stream, Loader=PrettySafeLoader)
  DATAPATH=""
  EPOCH=-1
  BATCHSIZE=-1
  try:
    DATAPATH=os.environ['DATAPATH']
    print("ENVIRONMENT VAR FOUND FOR DATAPATH. USING VALUE FROM run configuration.")
  except:
    print("NO ENVIRONMENT VAR FOUND FOR DATAPATH. USING VALUE FROM params.yaml.")
    DATAPATH=""
  try:
    EPOCH = os.environ['EPOCH']
    print("ENVIRONMENT VAR FOUND FOR EPOCH. USING VALUE FROM run configuration.")
  except:
    print("NO ENVIRONMENT VAR FOUND FOR EPOCH. USING VALUE FROM params.yaml.")
    EPOCH = -1
  try:
    BATCHSIZE = os.environ['BATCHSIZE']
    print("ENVIRONMENT VAR FOUND FOR BATCHSIZE. USING VALUE FROM run configuration.")
  except:
    print("NO ENVIRONMENT VAR FOUND FOR BATCHSIZE. USING VALUE FROM params.yaml.")
    BATCHSIZE = -1

  run_copyingDatasetFromUsbStick  =   params['run_copyingDatasetFromUsbStick']
  run_labeling_balancing          =   params['run_labeling_balancing']
  run_preprocessing               =   params['run_preprocessing']
  run_training                    =   params['run_training']
  run_training_for_subdir         =   params['run_training_for_subdir']
  run_saliencemapCreation         =   params['run_saliencemapCreation']
  run_tuner                       =   params['run_tuner']
  run_evaluation_only             =   params['run_evaluation_only']

  evaluation_only_modelName = params['path_for_loading_model_for_eval_only']

  print('run_copyingDatasetFromUsbStick '+str(run_copyingDatasetFromUsbStick                       ))
  print('run_labeling                   '+str(run_labeling_balancing                     ) )
  print('run_preprocessing              '+str(run_preprocessing                        ))
  print('run_training                   '+str(run_training                     )  )
  print('run_evaluation_only            '+str(run_evaluation_only                     )  )
  print('run_saliencemapCreation        '+str(run_saliencemapCreation   )                   )

  if run_copyingDatasetFromUsbStick:
      copyPasteBigDataFromUsbStick.run()
      print('DONE run_copyingDatasetFromUsbStick ' )

  if run_preprocessing:
      batch_data_preprocessing.run()
      print('DONE batch_data_preprocessing ' )

  if run_labeling_balancing:
      label_balance_PicturesByRpmAndGasflow.run()
      print('DONE run_labeling ' )

  if run_tuner:
      model_training.runTuner(DATAPATH, EPOCH, BATCHSIZE)
      print('DONE run_training')

  if run_training:
      model_training.runTraining(DATAPATH, EPOCH, BATCHSIZE)
      print('DONE run_training')
  if run_training_for_subdir:
      modules.training_for_each_subdir.main()
      print('DONE training_for_subdir')
  if run_evaluation_only:
      model_training.runEvaluation(evaluation_only_modelName)
      print('DONE run_evaluation_only')
  print("Starter has finished all jobs.")

