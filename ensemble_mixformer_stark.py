import vot
import cv2
import numpy as np
import sys

# project path change accordingly
prj_path = './Stark/'
sys.path.append(prj_path)

# project path change accordingly
mixformer_path = './MixFormer/'
sys.path.append(mixformer_path)

from lib.test.evaluation.tracker import Tracker as TrackerStark
from lib_mixformer.test.evaluation.tracker import Tracker as TrackerMixFormer
from vot_data_preprocessing import _mask_to_bbox
from segment_anything import sam_model_registry, SamPredictor

def _build_init_info(box):
    return {'init_bbox': box}

class mixformerTracker(object):

    def __init__(self, tracker_name='mixformer_vit_online', tracker_param='baseline_large'):

        tracker_params = {'model': 'mixformer_vit_large_online.pth.tar'}
        tracker_info = TrackerMixFormer(tracker_name, tracker_param, "vot20", tracker_params=tracker_params)
        params = tracker_info.params
        params.visualization = False
        params.debug = False
        self.tracker = tracker_info.create_tracker(params)

    def initialize(self, img_rgb, bbox):

        self.H, self.W, _ = img_rgb.shape
        self.tracker.initialize(img_rgb, _build_init_info(bbox))


class starkTracker(object):

    def __init__(self, tracker_name='stark_st', tracker_param='baseline'):

        tracker_info = TrackerStark(tracker_name, tracker_param, "vot20", None)
        params = tracker_info.get_parameters()
        params.visualization = False
        params.debug = False
        self.tracker = tracker_info.create_tracker(params)

    def initialize(self, img_rgb, bbox):

        self.H, self.W, _ = img_rgb.shape
        self.tracker.initialize(img_rgb, _build_init_info(bbox))


# path to SAM weights
sam_checkpoint = "./MixFormer/sam_weights/sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)


handle = vot.VOT("mask", multiobject=True)
imagefile = handle.frame()
objects = handle.objects()

# run STARK if there are more than 5 objects in the video sequences
if len(objects) > 4:
    image = cv2.cvtColor(cv2.imread(imagefile), cv2.COLOR_BGR2RGB)
    trackers = [mixformerTracker() for object in objects]

    for ind in range(len(trackers)):
        trackers[ind].initialize(image, _mask_to_bbox(objects[ind]))

    while True:
      imagefile = handle.frame()
      if not imagefile:
          break

      image = cv2.cvtColor(cv2.imread(imagefile), cv2.COLOR_BGR2RGB)
      predictor.set_image(image)
    
      pred_list = []

      for tracker in trackers:
          box = tracker.tracker.track(image)['target_bbox']
          input_box = np.array([box[0], box[1], box[0]+box[2], box[1]+box[3]])
          masks, scores, _ = predictor.predict(point_coords=None,point_labels=None,box=input_box[None, :],multimask_output=True,)
          # get mask with maximum score
          temp = masks[np.argmax(scores)]
          mask = temp.astype(np.uint8)
          pred_list.append(mask)

      handle.report(pred_list)
# run MixFormer if there are less than 5 objects in the video sequences
else:
    image = cv2.cvtColor(cv2.imread(imagefile), cv2.COLOR_BGR2RGB)
    trackers = [starkTracker() for object in objects]

    for ind in range(len(trackers)):
        trackers[ind].initialize(image, _mask_to_bbox(objects[ind]))

    while True:
      imagefile = handle.frame()
      if not imagefile:
          break

      image = cv2.cvtColor(cv2.imread(imagefile), cv2.COLOR_BGR2RGB)
      predictor.set_image(image)
    
      pred_list = []

      for tracker in trackers:
          box = tracker.tracker.track(image)['target_bbox']
          input_box = np.array([box[0], box[1], box[0]+box[2], box[1]+box[3]])
          masks, scores, _ = predictor.predict(point_coords=None,point_labels=None,box=input_box[None, :],multimask_output=True,)
          # get mask with maximum score
          temp = masks[np.argmax(scores)]
          mask = temp.astype(np.uint8)
          pred_list.append(mask)

      handle.report(pred_list)
