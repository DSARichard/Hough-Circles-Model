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

# https://towardsdatascience.com/stop-wasting-time-with-pytorch-datasets-17cac2c22fa8
# https://medium.com/fullstackai/how-to-train-an-object-detector-with-your-own-coco-dataset-in-pytorch-319e7090da5
# https://giou.stanford.edu/

shell_dir = os.getcwd().replace("\\", "/") + "/"
images_folder = "dextran_frames"
dataset_file = "dextran_v03_50x50_dataset.json"

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

# train and test sets
train_set = dextran_dataset.take(544, seed = 51)
test_set = dextran_dataset.exclude([s.id for s in train_set])

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

# load non-pretrained model
def get_model(num_classes):
  model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained = False)
  in_features = model.roi_heads.box_predictor.cls_score.in_features
  model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
  return model

# train and test datasets
torch_dextran_dataset = fotorchDataset(train_set, gt_field = "detections")
torch_dextran_dataset_test = fotorchDataset(test_set, gt_field = "detections")

# define collate function
collate_fn = lambda batch: tuple(zip(*batch))

# train on GPU if available else CPU
device = torch.device("cuda") if(torch.cuda.is_available()) else torch.device("cpu")
print(f"Using device: {device}")

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
  IoU = I/U
  GIoU = IoU - (C - U)/C
  return 1 - GIoU

# compute loss given ground truth and prediction bounding boxes
def bbox_loss(true_bboxes, pred_bboxes, loss_fn):
  if(len(pred_bboxes) == 0):
    loss = 2.0 if(len(true_bboxes) > 0) else 0.0
  elif(len(true_bboxes) == 0):
    loss = 2.0
  else:
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
      loss.append(bbox_loss)
    loss = np.mean(loss)
  return loss

# train model
def do_training(
  model, torch_dataset, torch_dataset_test,
  num_epochs, train_batch_size, test_batch_size, lr, lr_gamma
):
  # define train and validation data loaders
  data_loader = data.DataLoader(
    torch_dataset, batch_size = train_batch_size, shuffle = True, collate_fn = collate_fn
  )
  data_loader_test = data.DataLoader(
    torch_dataset_test, batch_size = test_batch_size, shuffle = False, collate_fn = collate_fn
  )
  
  # train on appropriate device
  model.to(device)
  
  # optimizer and learning rate scheduler
  params = [p for p in model.parameters() if(p.requires_grad)]
  optimizer = torch.optim.SGD(params, lr = lr, momentum = 0.9, weight_decay = 0.0005)
  lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 1, gamma = lr_gamma)
  
  # train and evaluate model
  for epoch in range(num_epochs):
    epoch_str = str(epoch + 1).zfill(len(str(num_epochs)))
    # train model
    model.train()
    train_losses = []
    i = 0
    for imgs, annotations in data_loader:
      imgs = [torch.from_numpy(img).to(device) for img in imgs]
      annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
      model.train()
      loss_dict = model(imgs, annotations)
      train_loss = loss_dict.values()
      train_loss = sum(train_loss)/len(train_loss)
      optimizer.zero_grad()
      train_loss.backward()
      optimizer.step()
      model.eval()
      true_bboxes = annotations[0]["boxes"].detach().numpy()
      pred_bboxes = model(imgs)[0]["boxes"].detach().numpy()
      train_loss = bbox_loss(true_bboxes, pred_bboxes, GIoU_loss)
      if(i > len(data_loader)//2):
        train_losses.append(train_loss)
      i += 1
      i_str = str(i).zfill(len(str(len(data_loader))))
      print(f"Epoch: {epoch_str}/{num_epochs}, Iteration: {i_str}/{len(data_loader)}, Train loss: {train_loss}")
    
    # evaluate model
    model.eval()
    val_losses = []
    i = 0
    for imgs, annotations in data_loader_test:
      imgs = [torch.from_numpy(img) for img in imgs]
      annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
      true_bboxes = annotations[0]["boxes"].detach().numpy()
      pred_bboxes = model(imgs)[0]["boxes"].detach().numpy()
      val_loss = bbox_loss(true_bboxes, pred_bboxes, GIoU_loss)
      val_losses.append(val_loss)
      i += 1
      i_str = str(i).zfill(len(str(len(data_loader_test))))
      print(f"Epoch: {epoch_str}/{num_epochs}, Iteration: {i_str}/{len(data_loader_test)}, Val loss: {val_loss}")
    
    # print epoch and loss, update learning rate
    train_loss, val_loss = map(np.mean, (train_losses, val_losses))
    print(f"Epochs completed: {epoch + 1}/{num_epochs}, Truncated average train loss: {train_loss}, Average val loss: {val_loss}")
    lr_scheduler.step()

# train and validate model
num_classes = len(torch_dextran_dataset.get_classes())
model = get_model(num_classes)
do_training(model, torch_dextran_dataset, torch_dextran_dataset_test, 1, 1, 1, 0.00003, 0.96)

# get validation images for prediction image writing
data_loader_test = data.DataLoader(
  torch_dextran_dataset_test, batch_size = 1, shuffle = False, collate_fn = collate_fn
)

# path for storing prediction images
if(not os.path.exists("dextran_v04_model_pred_imgs")):
  os.makedirs("dextran_v04_model_pred_imgs")

# begin evaluation
model.eval()
true_color = (255, 0, 0) # blue
pred_color = (0, 165, 255) # orange

# write validation model prediction image files
count = 0
for imgs, annotations in data_loader_test:
  imgs = [torch.from_numpy(img) for img in imgs]
  annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
  true_bboxes = annotations[0]["boxes"].detach().numpy()
  pred_bboxes = model(imgs)[0]["boxes"].detach().numpy()
  img = np.moveaxis(imgs[0].detach().numpy(), 0, -1)
  true_pred_img, pred_img = img.copy(), img.copy()
  
  for true_bbox in true_bboxes:
    x1, y1, x2, y2 = true_bbox.astype(int)
    cv2.rectangle(true_pred_img, (x1, y1), (x2, y2), true_color, 1)
  for pred_bbox in pred_bboxes:
    x1, y1, x2, y2 = pred_bbox.astype(int)
    cv2.rectangle(true_pred_img, (x1, y1), (x2, y2), pred_color, 1)
    cv2.rectangle(pred_img, (x1, y1), (x2, y2), pred_color, 1)
  cv2.imwrite("dextran_v04_model_pred_imgs/val" + str(count).zfill(2) + "_true_pred_img.jpg", true_pred_img)
  cv2.imwrite("dextran_v04_model_pred_imgs/val" + str(count).zfill(2) + "_pred_img.jpg", pred_img)
  count += 1

# save model
torch.save(model, "dextran_v04_544_faster_rcnn_model.pt")
