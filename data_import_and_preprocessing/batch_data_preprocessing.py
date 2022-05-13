import os
import sys
import yaml
import cv2
import json
from tqdm import tqdm

p = os.path.abspath('.')
sys.path.insert(1, p)

from data_import_and_preprocessing.dataset_formation import DataParser, ImageDataExtractor, LabelExtractor, \
    DataSetCreator

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


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
        self.save_path = 'data/preprocessed'

    def run_processing(self):
        pbar = tqdm(desc='Data preprocessing', total=len(self.data_points), leave=True)
        for data_point in self.data_points:
            os.makedirs(self.save_path, exist_ok=True)
            data = self.data_extractor.get_data(data_point)
            metadata = self.label_extractor.get_data(data_point)
            filename = data_point.datapoint_id
            fullname_img = os.path.join(self.save_path, filename + '.png')
            cv2.imwrite(fullname_img, data)

            fullname_metadata = os.path.join(self.save_path, filename + '.json')
            with open(fullname_metadata, 'w') as outfile:
                json.dump(metadata, outfile, indent=4)
            pbar.update(1)
        pbar.close()


if __name__ == '__main__':
    with open('params.yaml', 'r') as stream:
        params = yaml.safe_load(stream)
    no_epochs = params['training']['no_epochs']
    batch_size = params['training']['batch_size']

    data_dir = '/mnt/0A60B2CB60B2BD2F/Datasets/bioreactor_flow_regimes/02_data'
    data_parser = DataParser(data_dir)
    image_data_extractor = PrelimImageProcessor((32, 32, 1))
    label_extractor = PrelimMetadataProcessor(no_classes=3)
    data_processor = BatchDataProcessor(data_parser, image_data_extractor, label_extractor)
    data_processor.run_processing()
