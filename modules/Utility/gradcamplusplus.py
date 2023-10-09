from silence_tensorflow import silence_tensorflow
silence_tensorflow()
import tensorflow as tf
from tensorflow import keras
import matplotlib.cm as cm
import numpy as np
from tensorflow.keras.preprocessing import image

def find_highest_numbered_conv2d(model):
    your_list = [layer.name for layer in model.layers]
    highest_number = -1
    conv2d_string = ''
    for string in your_list:
        if 'conv2d' in string:
            try:
                number = int(string.split('_')[-1])
            except ValueError:  # handles the case when string is 'conv2d'
                number = 0
            if number >= highest_number:
                highest_number = number
                conv2d_string = string
    return conv2d_string


def get_heatmap(image, model, last_conv_layer_name,
                label_name=None,
                  category_id=None):
    #Get a heatmap by Grad-CAM++.
    #Parameters:
    #    model: A model object, build from tf.keras 2.X.
    #    img: An image ndarray.
    #    layer_name: A string, layer name in model.
    #    label_name: A list or None,
    #       show the label name by assign this argument,
    #        it should be a list of all label names.
    #   category_id: An integer, index of the class.
    #       Default is the category with the highest score in the prediction.
    #Return:
    #   A heatmap ndarray(without color).
    image=image/255
    img_tensor = np.expand_dims(image, axis=0)
    last_conv_layer_name=find_highest_numbered_conv2d(model)
    conv_layer = model.get_layer(last_conv_layer_name)
    # print("last conv layer name: "  +  last_conv_layer_name)
    heatmap_model = keras.Model([model.inputs], [conv_layer.output, model.output])

    with tf.GradientTape() as gtape1:
        with tf.GradientTape() as gtape2:
            with tf.GradientTape() as gtape3:
                conv_output, predictions = heatmap_model(img_tensor)
                if category_id == None:
                    category_id = np.argmax(predictions[0])
                if label_name:
                    print(label_name[category_id])
                output = predictions[:, category_id]
                conv_first_grad = gtape3.gradient(output, conv_output)
            conv_second_grad = gtape2.gradient(conv_first_grad, conv_output)
        conv_third_grad = gtape1.gradient(conv_second_grad, conv_output)

    global_sum = np.sum(conv_output, axis=(0, 1, 2))

    alpha_num = conv_second_grad[0]
    alpha_denom = conv_second_grad[0] * 2.0 + conv_third_grad[0] * global_sum
    alpha_denom = np.where(alpha_denom != 0.0, alpha_denom, 1e-10)

    alphas = alpha_num / alpha_denom
    alpha_normalization_constant = np.sum(alphas, axis=(0, 1))
    alphas /= alpha_normalization_constant

    weights = np.maximum(conv_first_grad[0], 0.0)

    deep_linearization_weights = np.sum(weights * alphas, axis=(0, 1))
    grad_CAM_map = np.sum(deep_linearization_weights * conv_output[0], axis=2)

    heatmap = np.maximum(grad_CAM_map, 0)
    max_heat = np.max(heatmap)
    if max_heat == 0:
        max_heat = 1e-10
    heatmap /= max_heat

    return heatmap


def get_preprocessed_img_from_path(img_path, target_size=(32, 32)):
    img = image.load_img(img_path, target_size=target_size)
    img = image.img_to_array(img)
    img /= 255
    return img

def combine_image_and_heatmap(image, heatmap, alpha=0.4, heatMapCorlorspace="cool", intensity=None,colorThreshold=30):
    #how the image with heatmap.
    #Args:
    #    img: nparray.
    #    heatmap:  image array, get it by calling grad_cam().
    #    alpha:    float, transparency of heatmap.
    #    return_array: bool, return a superimposed image array or not.
    #Return:
    #    None or image array.
    jet = cm.get_cmap(heatMapCorlorspace)
    # print(jet)
    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_colors[0:colorThreshold] = 0

    heatmap = heatmap * 255
    jet_heatmap = jet_colors[heatmap.astype(int)]

    # Create an image with RGB colorized heatmap
    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((image.shape[1], image.shape[0]))
    jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap) / 255

    # Superimpose the heatmap on original image
    superimposed_img = alpha * jet_heatmap + (1 - alpha) * image

    return superimposed_img

def heatmapArrayToHeatmapImg(heatmap, heatMapCorlorspace="cool"):
    jet = cm.get_cmap(heatMapCorlorspace)
    # print(jet)
    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_colors[0:0] = 0

    heatmap = heatmap * 255
    heatmapimg = jet_colors[heatmap.astype(int)]

    # Create an image with RGB colorized heatmap
    heatmapimg = tf.keras.preprocessing.image.array_to_img(heatmapimg)
    heatmapimg = tf.keras.preprocessing.image.img_to_array(heatmapimg) / 255

    return heatmapimg