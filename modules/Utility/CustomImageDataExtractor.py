from modules.data_import_and_preprocessing.dataset_formation import ImageDataExtractor
import cv2
import random
import numpy as np
class CustomImageDataExtractor(ImageDataExtractor):
    def get_data(self, data_point):
        filename = data_point.path_to_data
        img = cv2.imread(filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.rotate_image(img, random.choice([0, 90, 180, 270]))  # TODO: check if pictures actually rotate
        img_preprocessed = self.preprocess_image(img)
        return np.expand_dims(img_preprocessed, axis=-1)

    def rotate_image(self, image, angle):
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
        return result