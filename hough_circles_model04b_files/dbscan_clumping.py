from models import *
from sklearn.cluster import DBSCAN

# circle detection
def circle_detect(gray_image):
  # input image must have shape (720, 72)
  if(gray_image.shape != (720, 72)):
    raise ValueError("input image must have shape (720, 72)")
  
  # manipulate image
  sharpening_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
  gray_image = cv2.filter2D(gray_image, -1, sharpening_kernel)
  
  # detect circles
  # determine top left corner, width, and height of circle's bounding box
  circles = cv2.HoughCircles(gray_image, cv2.HOUGH_GRADIENT, 1.05, 5, param2 = 8.5, minRadius = 1, maxRadius = 7)
  radii = np.int_(np.round(circles[0, :, 2]))
  left = np.int_(np.round(circles[0, :, 0])) - radii
  top = np.int_(np.round(circles[0, :, 1])) - radii
  left, top = np.clip(left, 0, 72), np.clip(top, 0, 720)
  width, height = (np.int_(np.round(circles[0, :, 2]*2)),)*2
  return np.array([left, top, width, height])

# detect clumps using DBSCAN
def clump_dbscan(
  image, hough_eps, pred_eps, hough_min_samples, pred_min_samples,
  hough_centers, hough_radii, hough_bboxes, pred_centers, pred_bboxes
):
  # perform DBSCAN
  hough_clumping = DBSCAN(eps = hough_eps, min_samples = hough_min_samples).fit(hough_centers)
  pred_clumping = DBSCAN(eps = pred_eps, min_samples = pred_min_samples).fit(pred_centers)
  clump_image = image.copy()
  hough_clump_bboxes = [
    (72, 720, 0, 0)
    for i in range(max(hough_clumping.labels_) + 1)
  ]
  pred_clump_bboxes = [
    (72, 720, 0, 0)
    for i in range(max(pred_clumping.labels_) + 1)
  ]
  
  # draw circles (hough circles) and rectangles (model)
  # find clump bbox coordinates
  i = 0
  get_color = lambda loss: loss**2/4*255
  hough_bboxes_tree = scipy.spatial.KDTree(hough_bboxes.tolist())
  pred_bboxes_tree = scipy.spatial.KDTree(pred_bboxes.tolist())
  for x, y, r in np.concatenate((hough_centers, hough_radii[:, np.newaxis]), axis = 1):
    label = hough_clumping.labels_[i]
    hough_bbox = [x - r, y - r, x + r, y + r]
    closest_pred_bbox = pred_bboxes[pred_bboxes_tree.query(hough_bbox)[1]]
    bbox_loss = GIoU_loss(hough_bbox, closest_pred_bbox)
    cv2.circle(clump_image, (x, y), r, (255 - get_color(bbox_loss), 0, get_color(bbox_loss)), 1)
    if(label > -1):
      x0, y0, x1, y1 = hough_clump_bboxes[label]
      x0, y0, x1, y1 = min(x0, x - r), min(y0, y - r), max(x1, x + r), max(y1, y + r)
      x0, x1 = np.clip((x0, x1), 0, 72)
      y0, y1 = np.clip((y0, y1), 0, 720)
      hough_clump_bboxes[label] = (x0, y0, x1, y1)
    i += 1
  i = 0
  for x2, y2, x3, y3 in pred_bboxes:
    label = pred_clumping.labels_[i]
    pred_bbox = [x2, y2, x3, y3]
    closest_hough_bbox = hough_bboxes[hough_bboxes_tree.query(pred_bbox)[1]]
    bbox_loss = GIoU_loss(pred_bbox, closest_hough_bbox)
    cv2.rectangle(clump_image, (x2, y2), (x3, y3), (255 - get_color(bbox_loss), 0, get_color(bbox_loss)), 1)
    if(label > -1):
      x0, y0, x1, y1 = pred_clump_bboxes[label]
      x0, y0, x1, y1 = min(x0, x2), min(y0, y2), max(x1, x3), max(y1, y3)
      x0, x1 = np.clip((x0, x1), 0, 72)
      y0, y1 = np.clip((y0, y1), 0, 720)
      pred_clump_bboxes[label] = (x0, y0, x1, y1)
    i += 1
  
  # draw clump bboxes
  color = (30, 225, 75) # green
  for x0, y0, x1, y1 in hough_clump_bboxes:
    cv2.rectangle(clump_image, (x0, y0), (x1, y1), color, 1)
  color = (30, 150, 240) # orange
  for x0, y0, x1, y1 in pred_clump_bboxes:
    cv2.rectangle(clump_image, (x0, y0), (x1, y1), color, 1)
  color = (30, 225, 240) # yellow
  thresh = 0.85
  hybrid_clump_bboxes = []
  for hough_clump_bbox in hough_clump_bboxes:
    closest_pred_clump_bbox = None
    for pred_clump_bbox in pred_clump_bboxes:
      if(GIoU_loss(hough_clump_bbox, pred_clump_bbox) <= thresh):
        if(
          closest_pred_clump_bbox is None
          or GIoU_loss(hough_clump_bbox, pred_clump_bbox) < GIoU_loss(hough_clump_bbox, closest_pred_clump_bbox)
        ):
          closest_pred_clump_bbox = pred_clump_bbox
    if(closest_pred_clump_bbox is not None):
      x0, y0, x1, y1 = np.int_(np.round(np.mean((hough_clump_bbox, closest_pred_clump_bbox), axis = 0)))
      hybrid_clump_bboxes.append((x0, y0, x1, y1))
  hybrid_clump_bboxes = prune_detections(np.array(hybrid_clump_bboxes), GIoU_loss)
  for x0, y0, x1, y1 in hybrid_clump_bboxes:
    cv2.rectangle(clump_image, (x0, y0), (x1, y1), color, 2)
  return clump_image

# detect clumping
def det_clumping(model):
  # prepare model
  model.to(device)
  model.eval()
  if(not os.path.exists("dextran_v04b_clump_imgs")):
    os.makedirs("dextran_v04b_clump_imgs")
  
  # iterate through each image and perform DBSCAN on hough circle and model predictions
  for count in range(7509):
    img = cv2.imread("dextran_frames/frame" + str(count).zfill(4) + ".jpg")
    gray_img = np.uint8(cv2.cvtColor(np.float32(img), cv2.COLOR_BGR2GRAY))
    hough_centers = np.int_(np.round([
      (left + width/2, top + height/2)
      for left, top, width, height in circle_detect(gray_img).T
    ]))
    hough_radii = np.int_(np.round([
      (width + height)/4
      for left, top, width, height in circle_detect(gray_img).T
    ]))
    hough_bboxes = np.int_(np.round([
      (left, top, left + width, top + height)
      for left, top, width, height in circle_detect(gray_img).T
    ]))
    img_tensor = torch.from_numpy(np.moveaxis(np.float32(img), -1, 0)).to(device)
    pred_bboxes = prune_detections(
      model([img_tensor])[0]["boxes"].detach().cpu().numpy().astype(int), GIoU_loss
    )
    pred_bboxes = np.array([(x0, y0, x1, y1) for x0, y0, x1, y1 in pred_bboxes.tolist() if(max(x1 - x0, y1 - y0) <= 20)])
    pred_centers = np.int_(np.round([
      ((x0 + x1)/2, (y0 + y1)/2)
      for x0, y0, x1, y1 in pred_bboxes
    ]))
    clump_img = clump_dbscan(img, 17, 22, 5, 3, hough_centers, hough_radii, hough_bboxes, pred_centers, pred_bboxes)
    img_file = "dextran_v04b_clump_imgs/clump" + str(count).zfill(4) + ".jpg"
    cv2.imwrite(img_file, clump_img)
    print(f"Image {img_file} successfully written")



if(__name__ == "__main__"):
  # load model
  model = torch.load("dextran_v04b_7360_faster_rcnn_model.pt")
  
  # detect clumping
  det_clumping(model)
