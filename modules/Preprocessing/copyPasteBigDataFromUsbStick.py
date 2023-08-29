import shutil
import os
import yaml
import sys
def run():
    p = os.path.abspath('')
    sys.path.insert(1, p)
    print("1" + os.getcwd())
    print("2" + os.getcwd())

    class PrettySafeLoader(yaml.SafeLoader):
      def construct_python_tuple(self, node):
        return tuple(self.construct_sequence(node))

    with open('params.yaml', 'r') as stream:
      params = yaml.load(stream, Loader=PrettySafeLoader)
    # path to source directory
    src_dir = params['path_SdCardPicture']

    # path to destination directory. Dest directory must not exist. Will be created, otherwise error
    dest_dir = params['path_dataset']+"/images"

    # getting all the files in the source directory
    files = os.listdir(src_dir)

    shutil.copytree(src_dir, dest_dir)