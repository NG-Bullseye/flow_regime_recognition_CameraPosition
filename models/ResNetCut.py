from keras.applications import ResNet50V2
from keras.applications.resnet_v2 import preprocess_input, decode_predictions
from keras.utils import load_img,img_to_array
import numpy as np

model = ResNet50V2(weights='imagenet')
img_path = 'D:\Kersas_Data\jelly.jpg'
img = load_img(img_path, target_size=(224, 224))
x = img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = model.predict(x)

# decode the results into a list of tuples (class, description, probability)
print('Predicted:', decode_predictions(preds, top=3)[0])