import os
import sys
import yaml
import cv2
import json
from tqdm import tqdm

p = os.path.abspath('')
sys.path.insert(1, p)
from modules.data_import_and_preprocessing.dataset_formation import DataParser, ImageDataExtractor, LabelExtractor

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def run():
    print("######## PREPROCESSING ########")
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)
    print("Terminal: cd " + os.getcwd())
    os.chdir('../../')
    print("Terminal: cd " + os.getcwd())
    class PrettySafeLoader(yaml.SafeLoader):
      def construct_python_tuple(self, node):
        return tuple(self.construct_sequence(node))
    PrettySafeLoader.add_constructor(
      u'tag:yaml.org,2002:python/tuple',
      PrettySafeLoader.construct_python_tuple)
    class PrettySafeLoader(yaml.SafeLoader):
      def construct_python_tuple(self, node):
        return tuple(self.construct_sequence(node))
    PrettySafeLoader.add_constructor(
      u'tag:yaml.org,2002:python/tuple',
      PrettySafeLoader.construct_python_tuple)
    with open('./params.yaml', 'r') as stream:
      params = yaml.load(stream,Loader=PrettySafeLoader)
    picture_width = params['preprocessing']['picture_width']
    picture_hight = params['preprocessing']['picture_hight']
    no_classes = params['preprocessing']['no_classes']
    data_dir = params['path_SdCardPicture']
    print("DATA SOURCE PATH: "+data_dir)

    file_type_picture = params['preprocessing']['file_type_picture']

    class PrelimImageProcessor(ImageDataExtractor):
        def preprocess_image(self, img):
            img = cv2.resize(img,
                             (self.output_image_shape[0], self.output_image_shape[1]),
                             interpolation=cv2.INTER_CUBIC)
            return img

    class PrelimMetadataProcessor(LabelExtractor):
        def get_data(self, data_point):
            with open(data_point.path_to_metadata) as f:
                metadata = json.load(f)
                return metadata

    class BatchDataProcessor:
        def __init__(self, data_parser: DataParser, data_extractor, label_extractor):
            self.data_points = data_parser.data_points
            self.data_extractor = data_extractor
            self.label_extractor = label_extractor
            self.save_path = params['path_dataset']+"/preprocessed_images"+str(params['preprocessing']['picture_width'])

        def run_processing(self, file_type_picture):
            pbar = tqdm(desc='Data preprocessing', total=len(self.data_points), leave=True)
            for data_point in self.data_points:
                os.makedirs(self.save_path, exist_ok=True)
                data = self.data_extractor.get_data(data_point)
                metadata = self.label_extractor.get_data(data_point)
                filename = data_point.datapoint_id
                fullname_img = os.path.join(self.save_path, filename + '.' + file_type_picture)
                cv2.imwrite(fullname_img, data)
                jsonFileName_metadata_WITHOUT_CAMERA_FRAME=filename.split('_',3)[0]

                fullname_metadata = os.path.join(self.save_path, jsonFileName_metadata_WITHOUT_CAMERA_FRAME + '.json')
                with open(fullname_metadata, 'w') as outfile:
                    json.dump(metadata, outfile, indent=4)
                pbar.update(1)
            pbar.close()

    data_parser = DataParser(data_dir)
    image_data_extractor = PrelimImageProcessor((picture_width, picture_hight, 1))
    label_extractor = PrelimMetadataProcessor(no_classes=no_classes)
    data_processor = BatchDataProcessor(data_parser, image_data_extractor, label_extractor)
    data_processor.run_processing(file_type_picture)


if __name__ == '__main__':
    run()