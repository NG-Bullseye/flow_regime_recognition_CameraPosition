import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg
import tensorflow as tf
from keras_preprocessing.image.utils import load_img, img_to_array, array_to_img
from matplotlib import cm
from modules.Utility import grad_cam, gradcamplusplus, extract_yaw_from_pathstring
from silence_tensorflow import silence_tensorflow


silence_tensorflow()


def getImgPaths(dir_path):
    exts = ['png', 'jpg', 'jpeg', 'bmp']
    return [f for ext in exts for f in glob.glob(f"{dir_path}/*.{ext}")]


def compute_gradcamPP_mean_and_populate_gradcam_path(model_path, yaw_dir):
    model = tf.keras.models.load_model(model_path)
    total = np.zeros((28, 28))
    count = 0

    for layer in model.layers:
        print(layer.name)

    for img_path in getImgPaths(yaw_dir):
        img = cv2.imread(img_path)
        gray_img = np.expand_dims(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), axis=-1)

        # Compute gradcam heatmap
        gradcam_heatmap = gradcamplusplus.get_heatmap(gray_img, model, 'conv2d_81')

        # Print heatmap for debugging
        print("GradCam Heatmap: ", gradcam_heatmap)

        # Resize heatmap
        resized_heatmap = cv2.resize(gradcam_heatmap, (28, 28))

        # Print resized heatmap for debugging
        print("Resized Heatmap: ", resized_heatmap)

        total += resized_heatmap
        count += 1

        print(f"total: {total} count: {count}")

    mean = total / count
    cmap = cm.get_cmap("cool")(np.arange(256))[:, :3]
    cmap[0:0] = 0
    print(f"mean: {mean}")

    heatmap_disp_Img = array_to_img(cmap[(mean * 255).astype(int)])
    target_dir = os.path.dirname(model_path)

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    heatmap_disp_Img.save(os.path.join(target_dir, f"gradcamPP_mean_Img_yaw.png"))


if __name__ == '__main__':
    compute_gradcamPP_mean_and_populate_gradcam_path(
       #'/home/lwecke/Datensätze/Datensatz_v1_50p_3reg/Bulktraining_Outputs/Bulktraining_2023_07_24/yaw_10.102041244506836_20230726_135811_training_output/cnn_acc41.0_auc0.5.h5',
         '/home/lwecke/Datensätze/Datensatz_v1_50p_3reg/Bulktraining_Outputs/Bulktraining_2023_07_24/yaw/yaw_-45.0_20230726_140612_training_output/cnn_acc20.0_auc0.5.h5',
        #'working /home/lwecke/Datensätze/Datensatz_v1_50p_3reg/preprocessed_sorded_by_yaw/yaw_45.0'
          '/home/lwecke/Datensätze/Datensatz_v1_50p_3reg/yaw_11.938775062561035'
    )

    #images are okay of 30 yaw. model is flawed of 30 yaw
    #
