import cv2
import numpy as np
import scipy.spatial

import torch
from torch.utils import data

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

import fiftyone as fo
import fiftyone.utils.coco as fouc
import os

shell_dir = os.getcwd().replace("\\", "/") + "/"
images_folder = "dextran_frames"
dataset_file = "dextran_v03b_50x50_dataset.json"

# create dataset with absolute directories
make_abs_path_file = True
if(os.path.exists(dataset_file.replace(".json", "_abs_path.json"))):
  dataset_file = dataset_file.replace(".json", "_abs_path.json")
  make_abs_path_file = False
if(make_abs_path_file):
  f = open(dataset_file)
  coco_abs_path = f.read().replace(f"{images_folder}/", f"{shell_dir}{images_folder}/")
  f.close()
  dataset_file = dataset_file.replace(".json", "_abs_path.json")
  f = open(dataset_file, "wt")
  f.write(coco_abs_path)
  f.close()

# create FiftyOne dataset
dextran_dataset = fo.Dataset.from_dir(
  dataset_type = fo.types.COCODetectionDataset,
  data_path = images_folder,
  labels_path = dataset_file,
  name = "dextran_dataset",
  include_id = True,
  label_field = ""
)

# train, validation, and test sets
train_set = dextran_dataset.take(7360, seed = 51)
val_set = dextran_dataset.exclude([s.id for s in train_set]).take(100, seed = 51)
test_set = dextran_dataset.exclude([s.id for s in train_set] + [s.id for s in val_set])
