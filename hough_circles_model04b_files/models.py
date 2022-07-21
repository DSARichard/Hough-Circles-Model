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

# class to construct PyTorch dataset from FiftyOne dataset
class fotorchDataset(data.Dataset):
  def __init__(
    self, fiftyone_dataset,
    transforms = None, gt_field = "ground_truth", classes = None
  ):
    self.samples = fiftyone_dataset
    self.transforms = transforms
    self.gt_field = gt_field
    self.img_paths = self.samples.values("filepath")
    self.classes = classes
    if(not self.classes):
      self.classes = self.samples.default_classes
    if(self.classes[0] != "background"):
      self.classes = ["background"] + self.classes
    self.labels_map_rev = {c: i for i, c in enumerate(self.classes)}
  
  def __getitem__(self, idx):
    img_path = self.img_paths[idx]
    sample = self.samples[img_path]
    metadata = sample.metadata
    img = np.moveaxis(np.float32(cv2.imread(img_path)), -1, 0)
    
    boxes, labels, area, iscrowd = [], [], [], []
    detections = sample[self.gt_field].detections
    for det in detections:
      category_id = self.labels_map_rev[det.label]
      coco_obj = fouc.COCOObject.from_label(det, metadata, category_id = category_id)
      x, y, w, h = coco_obj.bbox
      boxes.append([x, y, x + w, y + h])
      labels.append(coco_obj.category_id)
      area.append(w*h)
      iscrowd.append(0)
    
    target = {}
    target["boxes"] = torch.as_tensor(boxes, dtype = torch.float32)
    target["labels"] = torch.as_tensor(labels, dtype = torch.int64)
    target["image_id"] = torch.as_tensor([idx])
    target["area"] = torch.as_tensor(area, dtype = torch.float32)
    target["iscrowd"] = torch.as_tensor(iscrowd, dtype = torch.int64)
    if(self.transforms is not None):
      img, target = self.transforms(img, target)
    return img, target
  
  def __len__(self):
    return len(self.img_paths)
  
  def get_classes(self):
    return self.classes

# define collate function
collate_fn = lambda batch: tuple(zip(*batch))

# train on GPU if available else CPU
device = torch.device("cuda") if(torch.cuda.is_available()) else torch.device("cpu")

# generalized intersection over union loss function
def GIoU_loss(bbox1, bbox2):
  # bounding boxes
  x1, y1, x2, y2 = bbox1
  x3, y3, x4, y4 = bbox2
  x1, y1, x2, y2 = min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)
  x3, y3, x4, y4 = min(x3, x4), min(y3, y4), max(x3, x4), max(y3, y4)
  A1 = (x2 - x1)*(y2 - y1)
  A2 = (x4 - x3)*(y4 - y3)
  
  # intersection of bounding boxes
  xI1, yI1, xI2, yI2 = max(x1, x3), max(y1, y3), min(x2, x4), min(y2, y4)
  I = (
    (xI2 - xI1)*(yI2 - yI1) if(xI1 < xI2 and yI1 < yI2)
    else 0.0
  )
  
  # enclosing box of bounding boxes
  xC1, yC1, xC2, yC2 = min(x1, x3), min(y1, y3), max(x2, x4), max(y2, y4)
  C = (xC2 - xC1)*(yC2 - yC1)
  
  # return GIoU loss
  U = A1 + A2 - I
  IoU = I/U if(U > 0) else I
  GIoU = IoU - (C - U)/C if(C > 0) else IoU
  return 1 - GIoU

# prune detections for repeated similar detections
def prune_detections(detections, loss_fn, thresh = 0.75, iterations = -1):
  iteration = 0
  detections = np.array(sorted(detections.tolist()))
  while(iteration != iterations):
    # find closest neighboring detections
    old_detections = detections.copy()
    bbox_nbrs = []
    for i in range(detections.shape[0]):
      closest_nbr, closest_nbr_dist, closest_nbr_ind = (None,)*3
      for j in range(i + 1, detections.shape[0]):
        nbr_dist = loss_fn(detections[i], detections[j])
        if(nbr_dist <= thresh):
          if(closest_nbr is None or nbr_dist < closest_nbr_dist):
            closest_nbr = detections[j]
            closest_nbr_dist = nbr_dist
            closest_nbr_ind = j
      if(closest_nbr is None):
        closest_nbr = detections[i]
        closest_nbr_dist = 0.0
        closest_nbr_ind = i
      bbox_nbrs.append(closest_nbr_ind)
    
    # find chains of neighbors of the same detection
    # closed chains loop upon themselves, broken chains do not
    closed_chains = []
    broken_chains = []
    for i in range(len(bbox_nbrs)):
      nbr_chain = [i]
      closed = True
      while(len(nbr_chain) == len(set(nbr_chain))):
        nbr_chain.append(bbox_nbrs[nbr_chain[-1]])
        if(loss_fn(detections[i], detections[nbr_chain[-1]]) > thresh):
          closed = False
          break
      nbr_chain = nbr_chain[:-1]
      if(closed):
        closed_chains.append(nbr_chain)
      else:
        broken_chains.append(nbr_chain)
    closed_chains, broken_chains = map(
      lambda chains: sorted(chains, key = len, reverse = True),
      (closed_chains, broken_chains)
    )
    
    # remove partial repetitions in closed chains
    i = 0
    while(i < len(closed_chains)):
      good_chain = True
      for j in range(i):
        prev_chain = set(closed_chains[j])
        if(set(closed_chains[i]).union(prev_chain) == prev_chain):
          del(closed_chains[i])
          good_chain = False
          break
      if(good_chain):
        i += 1
    
    # average each cluster of neighboring detections
    closed_detections, broken_detections = map(
      lambda chains: list(map(lambda chain: np.mean([
        detections[i]
        for i in (chain if(len(chain) > 0) else [[]])
      ], axis = 0).tolist(), chains)),
      (closed_chains, broken_chains)
    )
    
    # output final detections, containing all closed detections and unique broken detections
    detections = closed_detections.copy()
    for broken_detection in broken_detections:
      unique_detection = True
      for closed_detection in closed_detections:
        if(loss_fn(broken_detection, closed_detection) <= thresh):
          unique_detection = False
          break
      if(unique_detection):
        detections.append(broken_detection)
    detections = np.int_(np.round(detections))
    detections = np.array(sorted(detections.tolist()))
    if(np.array_equal(detections, old_detections)):
      break
    iteration += 1
  return detections

# compute loss given ground truth and prediction bounding boxes
def bbox_loss(true_bboxes, pred_bboxes, loss_fn, overlap_thresh = 0.75, prune_iter = -1):
  if(len(pred_bboxes) == 0):
    loss = 2.0 if(len(true_bboxes) > 0) else 0.0
  elif(len(true_bboxes) == 0):
    loss = 2.0
  else:
    true_bboxes, pred_bboxes = map(lambda detections: np.float64(prune_detections(detections, loss_fn, thresh = overlap_thresh, iterations = prune_iter)), (true_bboxes, pred_bboxes))
    true_bboxes_tree = scipy.spatial.KDTree(true_bboxes.tolist())
    pred_bboxes_tree = scipy.spatial.KDTree(pred_bboxes.tolist())
    loss = []
    for true_bbox in true_bboxes:
      closest_pred_bbox = pred_bboxes[pred_bboxes_tree.query(true_bbox)[1]]
      bbox_loss = loss_fn(true_bbox, closest_pred_bbox)
      loss.append(bbox_loss)
    for pred_bbox in pred_bboxes:
      closest_true_bbox = true_bboxes[true_bboxes_tree.query(pred_bbox)[1]]
      bbox_loss = loss_fn(pred_bbox, closest_true_bbox)
      x1, y1, x2, y2 = pred_bbox
      min_side, max_side = min(x2 - x1, y2 - y1), max(x2 - x1, y2 - y1)
      side_ratio = max_side/min_side if(min_side > 0) else np.Inf
      side_ratio = 4*np.tanh(side_ratio/4)
      size_loss = np.tanh((max_side - 20)/2)/2 + 1.5
      loss.append(bbox_loss*side_ratio*size_loss)
    loss = np.mean(loss)
  return loss

# load non-pretrained model
def get_model(num_classes):
  model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained = False)
  in_features = model.roi_heads.box_predictor.cls_score.in_features
  model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
  return model
