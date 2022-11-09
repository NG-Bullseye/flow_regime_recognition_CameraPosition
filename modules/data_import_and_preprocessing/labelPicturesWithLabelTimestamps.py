import os, shutil
import json
import sys
import random

import numpy.matlib
import yaml

def choose(path):

  res = [[], [], [], []]
  os.chdir(path)
  for file_name in os.listdir(path):
    if not file_name.endswith(".json"):
      continue
    with open(file_name, "r") as f:
      try:
        json1 = json.load(f)
        label = json1["flow_regime"]["parameters"]["value"]
        res[label].append(file_name)
      except Exception as e:
        print(file_name, e)

  min_len = min([len(x) for x in res])
  choices = []
  for r in res:
    choices.append(random.sample(r, min_len))


  return choices


def cut_additional_frames(choices: list, path: str):
  all_files = [file for file in os.listdir(path) if file.endswith(".json")]

  chosen_files = []
  for label_list in choices:
    chosen_files += label_list

  files_to_delete = set(all_files).difference(set(chosen_files))
  for df in files_to_delete:
    mdf = df.removesuffix(".json")
    os.remove(f"{path}/{mdf}_camera_frame.png")
    os.remove(f"{path}/{df}")

def run():

    def isInTimeStampInterval(filename):
        string = ""
        if filename.find(".json") == -1:
            return -1
        else:
            string = filename.replace(".json", "")
        timeInMs = int(string)
        label = -1
        for dict in labelTimestampList:
            label = label + 1
            begin = dict["begin"]
            end = dict["end"]
            if begin <= timeInMs <= end:
                return label
        return -1

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
    with open('params.yaml', 'r') as stream:
      params = yaml.load(stream, Loader=PrettySafeLoader)
    path = params['path_dataset'] + "/preprocessed_images" + str(params['preprocessing']['picture_width'])
        # path to source directory
    labelTimestampList = []

    file =open (params['path_dataset']+"/"+params['preprocessing']['filename_LabelTimestamps'])
    json_array = json.load(file)

    for item in json_array:
        details = {"begin": None, "end": None}
        details['begin'] = item['begin']
        details['end'] = item['end']
        labelTimestampList.append(details)
    class0=0
    class1=0
    class2=0
    class3=0
    print("labeling..")
    for filename in os.listdir(path):
        label = isInTimeStampInterval(filename)
        if label == -1:
          if(filename.split(".",1)[1]=="json"):
            print(f"no label for {filename}")
          continue
        input_file = open(path +"/"+ filename, "r")
        json1 = json.load(input_file)
        input_file = open(path +"/"+ filename, "w")
#
        json1["flow_regime"]["parameters"]["value"] = label
        json.dump(json1,input_file)
        if label==0:
          class0+=1
        elif label==1:
          class1+=1
        elif label==2:
          class2+=1
        elif label==3:
          class3+=1
    print(" class 0 " + "= "+ str(class0), sep=' ', end='\n')
    print(" class 1 " + "= "+str(class1), sep=' ', end='\n')
    print(" class 2 " + "= "+str(class2), sep=' ', end='\n')
    print(" class 3 " + "= "+str(class3), sep=' ', end='\n')

    choices = choose(path)
    for c in choices:
      print(len(c))
    cut_additional_frames(choices, path)
    classlabelnumberMinimum1= numpy.minimum(class0,class1)
    classlabelnumberMinimum2= numpy.minimum(class2,class3)
    classlabelnumberMinimum= numpy.minimum(classlabelnumberMinimum1,classlabelnumberMinimum2)
    print(classlabelnumberMinimum)

    class0 = 0
    class1 = 0
    class2 = 0
    class3 = 0
    print("counting..")
    for filename in os.listdir(path):
      label = isInTimeStampInterval(filename)
      if label == -1:
        continue
      input_file = open(path + "/" + filename, "r")
      json1 = json.load(input_file)
      if label == 0:
        class0 += 1
      elif label == 1:
        class1 += 1
      elif label == 2:
        class2 += 1
      elif label == 3:
        class3 += 1
    print("New dataset distribution")
    print(" Dispersed 0 " + "= " + str(class0), sep=' ', end='\n')
    print(" Transition 1 " + "= " + str(class1), sep=' ', end='\n')
    print(" Loaded 2 " + "= " + str(class2), sep=' ', end='\n')
    print(" Flooded 3 " + "= " + str(class3), sep=' ', end='\n')
if __name__ == '__main__':
    run()
