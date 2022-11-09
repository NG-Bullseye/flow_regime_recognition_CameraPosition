import os, shutil
import json
import yaml

def run():
    def isInTimeStampInterval(filename):
        try:
            string = ""
            if filename.find("_camera_frame.png") != -1:
                string = filename.replace("_camera_frame.png", "")
            else:
                string = filename.replace(".json", "")
            timeInMs = int(string)
            inExperiment = False
            inLabel = False

            for dict in expTimestampList:
                begin = dict["begin"]
                end = dict["end"]
                if begin <= timeInMs <= end:
                    inExperiment = True
            for dict in labelTimestampList:
                begin = dict["begin"]
                end = dict["end"]
                if begin <= timeInMs <= end:
                    inLabel = True
            if inLabel & inExperiment:
                return True
            return False
        except Exception as e:
            print(
                'isInTimeStampInterval Failed. Maybe check Image names. should end with_camera_frame.png %s. Reason: %s' % (
                filename, e))
    import sys
    p = os.path.abspath('')
    sys.path.insert(1, p)
    print("Terminal: cd " + os.getcwd())


    class PrettySafeLoader(yaml.SafeLoader):
      def construct_python_tuple(self, node):
        return tuple(self.construct_sequence(node))
    PrettySafeLoader.add_constructor(
      u'tag:yaml.org,2002:python/tuple',
      PrettySafeLoader.construct_python_tuple)
    with open('./params.yaml', 'r') as stream:
      params = yaml.load(stream, Loader=PrettySafeLoader)
    path_images = params['path_dataset']+"/preprocessed_images"+str(params['preprocessing']['picture_width'])
    expTimestampList = []
    labelTimestampList = []

# load expermient timestamps
    input_file = open(params['path_dataset']+"/"+ params['preprocessing']['filename_ExpermientTimestamps'])
    json_array = json.load(input_file)
    for item in json_array:
        details = {"begin": None, "end": None}
        details['begin'] = item['begin']
        details['end'] = item['end']
        expTimestampList.append(details)

# load labels json
    input_file = open(params['path_dataset']+"/"+ params['preprocessing']['filename_LabelTimestamps'])
    json_array = json.load(input_file)
    for item in json_array:
        details = {"begin": None, "end": None}
        details['begin'] = item['begin']
        details['end'] = item['end']
        labelTimestampList.append(details)

#execute cutting
    for filename in os.listdir(path_images):
        file_path = os.path.join(path_images, filename)
        try:
            if isInTimeStampInterval(filename):
                continue
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
            print(filename+" deleted", sep=' ', end='\n')
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

if __name__ == '__main__':
    run()