import os
import json
import random

def choose(path):

  res = [[]*4]
  os.chdir("/datasets_preprocessed/RightHighBelichtung/preprocessed_images56")
  for file_name in os.listdir(path):
    if not file_name.endswith(".json"):
      continue
    with open(file_name, "r") as f:
      json1 = json.load(f)
      label = json1["flow_regime"]["parameters"]["value"]
      res[label].append(file_name)

  min_len = min([len(x) for x in res])
  choices = []
  for r in res:
    choices.append(random.sample(r, min_len))


  return choices


#choose("/mnt/0A60B2CB60B2BD2F/Projects/flow_regime_recognition_CameraPosition/datasets_preprocessed/RightHighBelichtung/preprocessed_images56")
print([[], [], [], []])