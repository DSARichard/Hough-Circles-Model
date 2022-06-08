from models import get_model


from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

import fiftyone as fo
import fiftyone.utils.coco as fouc
import os


# load data
def load_data():
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

    return train_set, val_set, test_set



# train model
def do_training(
  model, torch_dataset, torch_dataset_val,
  num_epochs, train_batch_size, lr, lr_gamma
):
  # define train and validation data loaders
  data_loader = data.DataLoader(
    torch_dataset, batch_size = train_batch_size, shuffle = True, collate_fn = collate_fn
  )
  data_loader_val = data.DataLoader(
    torch_dataset_val, batch_size = 1, shuffle = False, collate_fn = collate_fn
  )
  
  # train on appropriate device
  model.to(device)
  
  # optimizer and learning rate scheduler
  params = [p for p in model.parameters() if(p.requires_grad)]
  optimizer = torch.optim.SGD(params, lr = lr, momentum = 0.9, weight_decay = 0.0005)
  lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 1, gamma = lr_gamma)
  
  # train and evaluate model
  results = [[], [num_epochs, train_batch_size, lr, lr_gamma]]
  for epoch in range(num_epochs):
    epoch_str = str(epoch + 1).zfill(len(str(num_epochs)))
    # train model on train set
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
    
    # evaluate model on validation set
    model.eval()
    val_losses = []
    i = 0
    for imgs, annotations in data_loader_val:
      imgs = [torch.from_numpy(img) for img in imgs]
      annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
      true_bboxes = annotations[0]["boxes"].detach().numpy()
      pred_bboxes = model(imgs)[0]["boxes"].detach().numpy()
      val_loss = bbox_loss(true_bboxes, pred_bboxes, GIoU_loss)
      val_losses.append(val_loss)
      i += 1
      i_str = str(i).zfill(len(str(len(data_loader_val))))
      print(f"Epoch: {epoch_str}/{num_epochs}, Iteration: {i_str}/{len(data_loader_val)}, Val loss: {val_loss}")
    
    # print epoch and loss, update learning rate
    train_loss, val_loss = map(np.mean, (train_losses, val_losses))
    print(f"Epochs completed: {epoch + 1}/{num_epochs}, Truncated average train loss: {train_loss}, Average val loss: {val_loss}")
    lr_scheduler.step()
    results[0].append(val_loss)
  return results

# train and validate model
num_classes = len(torch_dextran_dataset.get_classes())
model = get_model(num_classes)
model_results = do_training(model, torch_dextran_dataset, torch_dextran_dataset_val, 10, 32, 0.00080, 0.96)
print(model_results)
test_and_save = False

# test model and save if specified
if(test_and_save):
  # define test data loader
  data_loader_test = data.DataLoader(
    torch_dextran_dataset_test, batch_size = 1, shuffle = False, collate_fn = collate_fn
  )
  
  # path for storing prediction images
  if(not os.path.exists("dextran_v04b_model_pred_imgs")):
    os.makedirs("dextran_v04b_model_pred_imgs")
  
  # evaluate model on test set
  model.eval()
  test_losses = []
  true_color = (255, 0, 0) # blue
  pred_color = (0, 165, 255) # orange
  count = 0
  for imgs, annotations in data_loader_test:
    # evaluate and get loss
    imgs = [torch.from_numpy(img) for img in imgs]
    annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
    true_bboxes = annotations[0]["boxes"].detach().numpy()
    pred_bboxes = model(imgs)[0]["boxes"].detach().numpy()
    test_loss = bbox_loss(true_bboxes, pred_bboxes, GIoU_loss)
    test_losses.append(test_loss)
    
    # write images
    img = np.moveaxis(imgs[0].detach().numpy(), 0, -1)
    true_pred_img, pred_img = img.copy(), img.copy()
    for true_bbox in true_bboxes:
      x1, y1, x2, y2 = true_bbox.astype(int)
      cv2.rectangle(true_pred_img, (x1, y1), (x2, y2), true_color, 1)
    for pred_bbox in pred_bboxes:
      x1, y1, x2, y2 = pred_bbox.astype(int)
      cv2.rectangle(true_pred_img, (x1, y1), (x2, y2), pred_color, 1)
      cv2.rectangle(pred_img, (x1, y1), (x2, y2), pred_color, 1)
    cv2.imwrite("dextran_v04b_model_pred_imgs/test" + str(count).zfill(2) + "_true_pred_img.jpg", true_pred_img)
    cv2.imwrite("dextran_v04b_model_pred_imgs/test" + str(count).zfill(2) + "_pred_img.jpg", pred_img)
    count += 1
    i_str = str(count).zfill(len(str(len(data_loader_test))))
    print(f"Iteration: {i_str}/{len(data_loader_test)}, Test loss: {test_loss}")
  
  # print loss
  test_loss = np.mean(test_losses)
  print(f"Average test loss: {test_loss}")
  
  # save model
  torch.save(model, "dextran_v04b_7360_faster_rcnn_model.pt")


if __name__ == '__main__':
    model = get_model()
    do_training(model)