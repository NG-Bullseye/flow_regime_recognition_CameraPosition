import os
from silence_tensorflow import silence_tensorflow
silence_tensorflow()
import tensorflow as tf
from modules.Utility import grad_cam
from modules.Utility import gradcamplusplus
import imutils #rotate image
from matplotlib import cm
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
import numpy as np
from modules.Utility import extract_yaw_from_pathstring
from keras_preprocessing.image.utils import img_to_array,load_img, array_to_img
from modules.Utility.database import Sim_data_db as sim_db

class Postprocessor:

    def __init__(self,root_path_to_bulktraining_output):
        self.root_path_to_bulktraining_output=root_path_to_bulktraining_output
        self.plotting_enabled = False
        self.vector_yaw_mapping = None
        self.imagePath_gravity_dict = {}
        self.yaw_gradcampath_mapping ={}
        # Initialize the database
        self.db = sim_db('my_database.db')
        self.db.reset()
        self.image_dict = {}
        self.root_path_for_preprocessed_yaw_sorted_images="/home/lwecke/Datensätze/Datensatz_v1_50p_3reg/preprocessed_sorded_by_yaw"
        # Add data to the database
        # db.put(image_path='\home\leo\succmd.png', gradcam_mean_path='\home\leo\gradcams\succmd_gradcam.png')

        # Get data from the database

        # Don't forget to close the connection when you're done

    def populate_db_with_yaw_model_path(self):
        for yaw_training_folders_name in os.listdir(
                self.root_path_to_bulktraining_output):  # for each yaw position training output folder
            yaw_training_folder_path = os.path.join(self.root_path_to_bulktraining_output,
                                                    yaw_training_folders_name)  # get path of every sub folder
            for training_output_model_name in os.listdir(
                    yaw_training_folder_path):  # training output files like model.h5 dataset.tf element_spec grad_cam_mean
                training_output_model_path = os.path.join(yaw_training_folder_path,
                                                          training_output_model_name)  # get path of every sub folder

                if not os.path.isdir(training_output_model_path) and ".h5" in training_output_model_name.lower():
                    self.db.put(yaw=str(extract_yaw_from_pathstring.extract_value(training_output_model_path)),
                                model_paths=training_output_model_path)
    def populate_db_with_yaw_images_path(self):
        for parent_yaw_image_folders_name in os.listdir(
                self.root_path_to_bulktraining_output):  # for each yaw position training output folder
            yaw_folder_with_preprocessed_images_path = os.path.join(self.root_path_to_bulktraining_output,
                                                    parent_yaw_image_folders_name)  # get path of every sub folder
            for folder_name_yaw_value in os.listdir(
                    yaw_folder_with_preprocessed_images_path):  # training output files like model.h5 dataset.tf element_spec grad_cam_mean
                folder_name_yaw_value_path = os.path.join(yaw_folder_with_preprocessed_images_path,
                                                          folder_name_yaw_value)  # get path of every sub folder
                self.db.put(yaw= str(extract_yaw_from_pathstring.extract_value(folder_name_yaw_value_path)),image_path=folder_name_yaw_value_path)
    def populate_db_with_yaw_acc(self):
        pass

    def bulk_compute_gradcamPP_mean_for_each_yaw(self):
        for i in range(self.db.max_distinct_count()-1):
            # Load the image
            self.image_dict{}

            # Check if the image was correctly loaded
            if img is not None:
                self.compute_gradcamPP_mean(path_model,path_tf_dataset_test)
            #compute_gradcamPP_mean and calculate rec_scalar and save in yaw rec_scalar mapping which is done already in compute_bulk_rec_scalar. but compute_bulk_rec_scalar should not do this in bulk it should do it only for one. Use the bulk stuff in this method instead.
    def populate_db_with_yaw_gradcam_mean_path(self):
        for yaw_training_folders_name in os.listdir(
                self.root_path_to_bulktraining_output):  # for each yaw position training output folder
            yaw_training_folder_path = os.path.join(self.root_path_to_bulktraining_output,
                                                    yaw_training_folders_name)  # get path of every sub folder
            for training_output_gradcam_mean_name in os.listdir(
                    yaw_training_folder_path):  # training output files like model.h5 dataset.tf element_spec grad_cam_mean
                training_output_gradcam_mean_path = os.path.join(yaw_training_folder_path,
                                                          training_output_gradcam_mean_name)  # get path of every sub folder

                if not os.path.isdir(training_output_gradcam_mean_path) and "gradcam" in training_output_gradcam_mean_name.lower():
                    self.db.put(yaw=str(extract_yaw_from_pathstring.extract_value(training_output_gradcam_mean_path)),
                                gradcam_mean_path=training_output_gradcam_mean_path)
    def compute_gradcamPP_mean(self,path_model,path_tf_dataset_test,path_element_spec):
        x_test = [];y_test = [];imgs= []

        sum   = np.zeros((28,28));count  =0
        for image,label,preds in zip(imgs,y_test):
            heatmap = gradcamplusplus.get_heatmap(path_tf_dataset_test, path_model, 'conv2d_1')
            sum+=heatmap
            count+=1
        mean =     sum/count

        cool = cm.get_cmap("cool")
        cool_colors = cool(np.arange(256))[:, :3]
        colorThreshold=0
        cool_colors[0:colorThreshold] = 0
        fig,axs= plt.subplots(2,2,figsize=(28,28))#neuer plot
        heatmap_disp_Array2Dint = cool_colors[(mean * 255).astype(int)]
        heatmap_disp_Img = tf.keras.preprocessing.image.array_to_img(heatmap_disp_Array2Dint)
        axs[0,0].set_title("Disp N="+str(count), fontdict={'color':  'black','weight': 'bold','size': 24})
        axs[0,0].set_xticks([])
        axs[0,0].set_yticks([])
        axs[0,0].imshow(heatmap_disp_Img.rotate(180))

        plt.savefig(f"{path_model.split('cnn')[0]}mean.png")
        plt.show()

        print("Model: " +path_model)
        print("Data: " +path_tf_dataset_test)
    def compute_gradcamPP_from_png(self,model_path,input_img_path):
        img = cv2.imread(input_img_path)
        img_preprocessed = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_preprocessed = cv2.resize(img_preprocessed, (32, 32), interpolation=cv2.INTER_CUBIC)
        img_preprocessed = img_preprocessed / 255
        reconstructed_model = tf.keras.models.load_model(model_path)
        reconstructed_model.compile(loss='categorical_crossentropy', metrics='accuracy')
        alpha = 0.97
        alphaPlusPlus = 0.99

        alphaIncrement = 0.0001
        alphaPlusPlusIncrement = 0.0001

        no_pictures = 100
        heatmapType = "cool"
        return gradcamplusplus.get_heatmap(img_preprocessed, reconstructed_model, "conv2d_1")
    def compute_bulk_rec_scalar(self):
        # Adjust the size of the figure
        for idx, (image_path, image) in enumerate(self.image_dict.items()):
            img = mpimg.imread(image_path)

            # If the image was not correctly loaded, return None
            if img is None:
                print("Image not found at path:", image_path)
                self.imagePath_gravity_dict[image_path] = (-1, -1)
                continue

            # Convert the image to grayscale
            grayscale_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

            # Compute the weighted average of the pixel coordinates
            height, width = grayscale_img.shape
            x_coords, y_coords = np.meshgrid(np.arange(width), np.arange(height))
            total_weight = np.sum(grayscale_img)
            avg_x = np.sum(x_coords * grayscale_img) / total_weight
            avg_y = np.sum(y_coords * grayscale_img) / total_weight

            # Round the average coordinates
            avg_x_rounded = round(avg_x)
            avg_y_rounded = round(avg_y)

            # Compute the center of the image
            center_x = round(width / 2)
            center_y = round(height / 2)

            # Compute the vector from the center of the image to the center of gravity
            vector_x = avg_x_rounded - center_x
            vector_y = avg_y_rounded - center_y
            vector_length = np.sqrt(vector_x ** 2 + vector_y ** 2)

            # Print the average coordinates
            print("x: " + str(vector_x) + " " + "y: " + str(vector_y))
            print("Vector length: " + str(vector_length))
            if self.plotting_enabled:
                self.plot_rec_scalar(self.image_dict,idx,img,avg_x_rounded,avg_y_rounded,center_x,center_y,vector_x,vector_y,vector_length)
    def populate_db_with_yaw_rec_scalar(self):
        pass
    def plot_rec_scalar(self,image_dict,idx,img,avg_x_rounded,avg_y_rounded,center_x,center_y,vector_x,vector_y,vector_length):
            fig = plt.figure(figsize=(20, 40))
            # Plot the original image
            ax1 = fig.add_subplot(len(image_dict), 2, 2 * idx + 1)
            ax1.imshow(img)
            ax1.axis('off')
            ax1.set_title(f"x: {avg_x_rounded}, y: {avg_y_rounded}", fontsize=10, pad=-20)

            # Plot the image with the center of gravity marked
            ax2 = fig.add_subplot(len(image_dict), 2, 2 * idx + 2)
            ax2.imshow(img)

            # Draw a green square at the center of gravity
            rect_gravity = patches.Rectangle((avg_x_rounded - 5, avg_y_rounded - 5), 10, 10, linewidth=1, edgecolor='g',
                                             facecolor='none')
            ax2.add_patch(rect_gravity)

            # Draw a red square at the center of the image
            rect_image = patches.Rectangle((center_x - 5, center_y - 5), 10, 10, linewidth=1, edgecolor='r',
                                           facecolor='none')
            ax2.add_patch(rect_image)

            # Draw a line from the center of the image to the center of gravity
            ax2.arrow(center_x, center_y, vector_x, vector_y, head_width=5, head_length=5, fc='blue', ec='blue')

            ax2.axis('off')
            ax2.set_title(f"x: {avg_x_rounded}, y: {avg_y_rounded}, Vector length: {vector_length}", fontsize=10,
                          pad=-20)

            plt.tight_layout(pad=3.0)
            plt.show()

def main():
    postprocessor = Postprocessor("/home/lwecke/Datensätze/Datensatz_v1_50p_3reg/Bulktraining_Outputs/Bulktraining_2023_07_24")
    postprocessor.populate_db_with_yaw_images_path()
    postprocessor.populate_db_with_yaw_model_path()
    postprocessor.populate_db_with_yaw_acc()

    postprocessor.bulk_compute_gradcamPP_mean_for_each_yaw()
    postprocessor.populate_db_with_yaw_gradcam_mean_path()
    postprocessor.compute_bulk_rec_scalar()
    postprocessor.populate_db_with_yaw_rec_scalar()

if __name__ == '__main__':
    main()
