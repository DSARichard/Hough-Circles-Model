from models import *

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
  
  # create or load FiftyOne dataset
  if(fo.dataset_exists("dextran_dataset")):
    print("Loading dataset")
    dextran_dataset = fo.load_dataset("dextran_dataset")
    print("Dataset loaded")
  else:
    print("Creating dataset")
    dextran_dataset = fo.Dataset.from_dir(
      dataset_type = fo.types.COCODetectionDataset,
      data_path = images_folder,
      labels_path = dataset_file,
      name = "dextran_dataset",
      include_id = True
    )
    dextran_dataset.persistent = True
    print("Dataset created")
  
  # train, validation, and test sets
  print("Creating test set")
  test_set = dextran_dataset.take(49, seed = 51)
  print("Test set created")
  print("Creating val set")
  val_set = dextran_dataset.exclude([s.id for s in test_set]).take(100, seed = 51)
  print("Val set created")
  print("Creating train set")
  train_set = dextran_dataset.exclude([s.id for s in val_set] + [s.id for s in test_set])
  print("Train set created")
  
  # train and validation datasets
  torch_dextran_dataset = fotorchDataset(train_set, gt_field = "detections")
  torch_dextran_dataset_val = fotorchDataset(val_set, gt_field = "detections")
  torch_dextran_dataset_test = fotorchDataset(test_set, gt_field = "detections")
  return torch_dextran_dataset, torch_dextran_dataset_val, torch_dextran_dataset_test

# train model
def do_training(
  model, torch_dataset, torch_dataset_val,
  num_epochs, train_batch_size, lr, momentum, weight_decay, lr_gamma0, lr_gamma1, overlap_thresh = 0.75, prune_iter = -1
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
  print(f"Using device: {device}")
  
  # optimizer and learning rate scheduler
  params = [p for p in model.parameters() if(p.requires_grad)]
  optimizer = torch.optim.SGD(params, lr = lr, momentum = momentum, weight_decay = weight_decay)
  lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 1, gamma = lr_gamma0)
  
  # train and evaluate model
  results = [[], [num_epochs, train_batch_size, lr, momentum, weight_decay, lr_gamma0, lr_gamma1]]
  for epoch in range(num_epochs):
    epoch_str = str(epoch + 1).zfill(len(str(num_epochs)))
    # train model on train set
    model.train()
    train_losses = []
    i = 0
    for imgs, annotations in data_loader:
      imgs = [torch.from_numpy(img).to(device) for img in imgs]
      annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
      model.eval()
      true_bboxes = annotations[0]["boxes"].detach().cpu().numpy()
      pred_bboxes = model(imgs)[0]["boxes"].detach().cpu().numpy()
      train_loss = bbox_loss(true_bboxes, pred_bboxes, GIoU_loss, overlap_thresh, prune_iter)
      model.train()
      train_loss_tensor = model(imgs, annotations).values()
      train_loss_tensor = sum(train_loss_tensor)/len(train_loss_tensor)
      if(i < len(data_loader)//5 and epoch == 0):
        train_loss_tensor *= 1.5
      else:
        train_loss_tensor *= torch.tensor([train_loss**2], requires_grad = True).to(device)[0]
      optimizer.zero_grad()
      train_loss_tensor.backward()
      optimizer.step()
      if(i%(len(data_loader)//4) == 0 and i != 0):
        lr_scheduler.step()
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
      imgs = [torch.from_numpy(img).to(device) for img in imgs]
      annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
      true_bboxes = annotations[0]["boxes"].detach().cpu().numpy()
      pred_bboxes = model(imgs)[0]["boxes"].detach().cpu().numpy()
      val_loss = bbox_loss(true_bboxes, pred_bboxes, GIoU_loss, overlap_thresh, prune_iter)
      val_losses.append(val_loss)
      i += 1
      i_str = str(i).zfill(len(str(len(data_loader_val))))
      print(f"Epoch: {epoch_str}/{num_epochs}, Iteration: {i_str}/{len(data_loader_val)}, Val loss: {val_loss}")
    
    # print epoch and loss, update learning rate
    train_loss, val_loss = map(np.mean, (train_losses, val_losses))
    print(f"Epochs completed: {epoch + 1}/{num_epochs}, Truncated average train loss: {train_loss}, Average val loss: {val_loss}")
    lr_scheduler.step()
    if(epoch == 0):
      lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 1, gamma = lr_gamma1)
    results[0].append(val_loss)
  return results

# test model and save if specified
def do_testing(model, save_model = True, overlap_thresh = 0.75, prune_iter = -1):
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
  true_color = (210, 45, 0) # blue
  pred_color = (0, 165, 255) # orange
  prune_color = (0, 0, 255) # red
  count = 0
  for imgs, annotations in data_loader_test:
    # evaluate and get loss
    imgs = [torch.from_numpy(img).to(device) for img in imgs]
    annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
    true_bboxes = annotations[0]["boxes"].detach().cpu().numpy()
    pred_bboxes = model(imgs)[0]["boxes"].detach().cpu().numpy()
    test_loss = bbox_loss(true_bboxes, pred_bboxes, GIoU_loss, overlap_thresh, prune_iter)
    test_losses.append(test_loss)
    
    # write images
    img = np.moveaxis(imgs[0].detach().cpu().numpy(), 0, -1)
    true_pred_prune_img, prune_img = img.copy(), img.copy()
    for true_bbox in true_bboxes:
      x1, y1, x2, y2 = true_bbox.astype(int)
      cv2.rectangle(true_pred_prune_img, (x1, y1), (x2, y2), true_color, 1)
    for pred_bbox in pred_bboxes:
      x1, y1, x2, y2 = pred_bbox.astype(int)
      cv2.rectangle(true_pred_prune_img, (x1, y1), (x2, y2), pred_color, 1)
    for prune_bbox in prune_detections(pred_bboxes, GIoU_loss, iterations = prune_iter):
      x1, y1, x2, y2 = prune_bbox.astype(int)
      cv2.rectangle(true_pred_prune_img, (x1, y1), (x2, y2), prune_color, 1)
      cv2.rectangle(prune_img, (x1, y1), (x2, y2), prune_color, 1)
    cv2.imwrite("dextran_v04b_model_pred_imgs/test" + str(count).zfill(2) + "_true_pred_prune_img.jpg", true_pred_prune_img)
    cv2.imwrite("dextran_v04b_model_pred_imgs/test" + str(count).zfill(2) + "_prune_img.jpg", prune_img)
    count += 1
    i_str = str(count).zfill(len(str(len(data_loader_test))))
    print(f"Iteration: {i_str}/{len(data_loader_test)}, Test loss: {test_loss}")
  
  # print loss
  test_loss = np.mean(test_losses)
  print(f"Average test loss: {test_loss}")
  
  # save model if specified
  if(save_model):
    torch.save(model, "dextran_v04b_7360_faster_rcnn_model.pt")



if(__name__ == "__main__"):
  # load data
  torch_dextran_dataset, torch_dextran_dataset_val, torch_dextran_dataset_test = load_data()
  
  # train and validate model
  num_classes = len(torch_dextran_dataset.get_classes())
  model = get_model(num_classes)
  model_results = do_training(model, torch_dextran_dataset, torch_dextran_dataset_val, 4, 16, 0.00005, 0.93, 0.01, 0.75, 0.35)
  print(f"Model results: {model_results}")
  
  # test model
  do_testing(model)
