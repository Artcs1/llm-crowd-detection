from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from collections import defaultdict
import heapq
import logging
import time
import numpy as np
import pprint

country_to_region_globe = {
    "United Kingdom": "Anglo",
    "Belgium": "Germanic Europe",
    "Ireland": "Anglo",
    "France": "Latin Europe",
    "Germany": "Germanic Europe",
    "Portugal": "Latin Europe",
    "Spain": "Latin Europe",
    "Italy": "Latin Europe",
    "Switzerland": "Latin Europe",
    "Netherlands": "Germanic Europe",
    "Austria": "Germanic Europe",
    "Malta": "Latin Europe",
    "Denmark": "Nordic Europe",
    "Norway": "Nordic Europe",
    "Sweden": "Nordic Europe",
    "Finland": "Nordic Europe",
    "Iceland": "Nordic Europe",
    "Greece": "Eastern Europe",
    "Poland": "Eastern Europe",
    "Czech Republic": "Eastern Europe",
    "Slovakia": "Eastern Europe",
    "Hungary": "Eastern Europe",
    "Romania": "Latin Europe",
    "Belarus": "Eastern Europe",
    "Serbia": "Eastern Europe",
    "Moldova": "Latin Europe",
    "Estonia": "Nordic Europe",
    "Latvia": "Nordic Europe",
    "United States": "Anglo",
    "Canada": "Anglo",
    "Bermuda": "Other",
    "Mexico": "Latin America",
    "Brazil": "Latin America",
    "Chile": "Latin America",
    "Peru": "Latin America",
    "Uruguay": "Latin America",
    "Paraguay": "Latin America",
    "Colombia": "Latin America",
    "Costa Rica": "Latin America",
    "Venezuela": "Latin America",
    "Jamaica": "African",
    "Barbados": "African",
    "Antigua and Barbuda": "African",
    "Saint Lucia": "Other",
    "Kenya": "African",
    "Malawi": "African",
    "South African": "African",
    "Nigeria": "African",
    "Ghana": "African",
    "Senegal": "African",
    "Benin": "African",
    "Madagascar": "African",
    "Ethiopia": "African",
    "Eritrea": "African",
    "Mali": "African",
    "Gambia": "African",
    "United Arab Emirates": "Middle East",
    "Jordan": "Middle East",
    "Syria": "Middle East",
    "Iraq": "Middle East",
    "Lebanon": "Middle East",
    "Egypt": "Middle East",
    "Morocco": "Middle East",
    "Kuwait": "Middle East",
    "Israel": "Latin Europe",
    "Turkey": "Middle East",
    "Afghanistan": "South-East Asia",
    "India": "South-East Asia",
    "Pakistan": "South-East Asia",
    "Bangladesh": "South-East Asia",
    "Sri Lanka": "South-East Asia",
    "Nepal": "South-East Asia",
    "China": "Confucian Asia",
    "Japan": "Confucian Asia",
    "South Korea": "Confucian Asia",
    "Taiwan": "Confucian Asia",
    "Thailand": "South-East Asia",
    "Vietnam": "Confucian Asia",
    "Indonesia": "South-East Asia",
    "Singapore": "Confucian Asia",
    "Brunei": "South-East Asia",
    "Kazakhstan": "Eastern Europe",
    "Kyrgyzstan": "Eastern Europe",
    "Tajikistan": "South-East Asia",
    "Azerbaijan": "Middle East",
    "Georgia": "Eastern Europe",
    "Australia": "Anglo",
    "New Zealand": "Anglo",
    "Samoa": "South-East Asia",
    "Holy See": "Other",
    "South Africa": "African"
}

try:
    from . import jr
except:
    import jr

from scipy.optimize import linear_sum_assignment
from collections import Counter

def print_time(message, start):
  logging.info("==> %g seconds to %s", time.time() - start, message)
def get_overlaps_and_scores_box_mode(detected_boxes, detected_scores, groundtruth_boxes, groundtruth_is_group_of_list):
    """Computes overlaps and scores between detected and groudntruth boxes.

    Args:
      detected_boxes: A numpy array of shape [N, 4] representing detected box
          coordinates
      detected_scores: A 1-d numpy array of length N representing classification
          score
      groundtruth_boxes: A numpy array of shape [M, 4] representing ground truth
          box coordinates
      groundtruth_is_group_of_list: A boolean numpy array of length M denoting
          whether a ground truth box has group-of tag. If a groundtruth box
          is group-of box, every detection matching this box is ignored.

    Returns:
      iou: A float numpy array of size [num_detected_boxes, num_gt_boxes]. If
          gt_non_group_of_boxlist.num_boxes() == 0 it will be None.
      ioa: A float numpy array of size [num_detected_boxes, num_gt_boxes]. If
          gt_group_of_boxlist.num_boxes() == 0 it will be None.
      scores: The score of the detected boxlist.
      num_boxes: Number of non-maximum suppressed detected boxes.
    """
    detected_boxlist = jr.np_box_list.BoxList(detected_boxes)
    detected_boxlist.add_field('scores', detected_scores)
    gt_non_group_of_boxlist = jr.np_box_list.BoxList(
        groundtruth_boxes[~groundtruth_is_group_of_list])
    iou = jr.np_box_list_ops.iou(detected_boxlist, gt_non_group_of_boxlist)
    #print(iou)
    scores = detected_boxlist.get_field('scores')
    num_boxes = detected_boxlist.num_boxes()
    return iou, None, scores, num_boxes
def make_image_key(video_id, keyframe_id):
  """Returns a unique identifier for a video id & keyframe_id."""
  return "%s,%04d" % (video_id, int(keyframe_id))
def refine_group_ids(detected_boxes, detected_scores, groundtruth_boxes, groundtruth_is_group_of_list):

  """Labels boxes detected with the same class from the same image as tp/fp.
  Args:
    detected_boxes: A numpy array of shape [N, 4] representing detected box
        coordinates
    detected_scores: A 1-d numpy array of length N representing classification
        score
    groundtruth_boxes: A numpy array of shape [M, 4] representing ground truth
        box coordinates
    groundtruth_is_difficult_list: A boolean numpy array of length M denoting
        whether a ground truth box is a difficult instance or not. If a
        groundtruth box is difficult, every detection matching this box
        is ignored.
    groundtruth_is_group_of_list: A boolean numpy array of length M denoting
        whether a ground truth box has group-of tag. If a groundtruth box
        is group-of box, every detection matching this box is ignored.
    detected_masks: (optional) A uint8 numpy array of shape
      [N, height, width]. If not None, the scores will be computed based
      on masks.
    groundtruth_masks: (optional) A uint8 numpy array of shape
      [M, height, width].

  Returns:
    Two arrays of the same size, containing all boxes that were evaluated as
    being true positives or false positives; if a box matched to a difficult
    box or to a group-of box, it is ignored.

    scores: A numpy array representing the detection scores.
    tp_fp_labels: a boolean numpy array indicating whether a detection is a
        true positive.
  """

  gt_refine, det_refine, FPs = [], [], []
  #print(detected_boxes)
  #print(groundtruth_boxes)

  if len(detected_boxes) == 0:
      return np.array([], dtype=int), np.array([], dtype=int)

  (iou, _, _, num_detected_boxes) = get_overlaps_and_scores_box_mode(
      detected_boxes=detected_boxes,
      detected_scores=detected_scores,
      groundtruth_boxes=groundtruth_boxes,
      groundtruth_is_group_of_list=groundtruth_is_group_of_list)

  if iou.shape[1] > 0:
      max_overlap_gt_ids = np.argmax(iou, axis=1)
      for i in range(num_detected_boxes):
          gt_id = max_overlap_gt_ids[i]

          if iou[i, gt_id] >= 0.5:
            gt_refine.append(gt_id)
            det_refine.append(i)
          else:
            FPs.append(i)
  
  #print(detected_boxes)
  #print(groundtruth_boxes)

  if gt_refine == []:
      return np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=int)

  return gt_refine, det_refine, FPs
def cluster_acc(y_true, y_pred):
  """
  Calculate clustering accuracy. Require scikit-learn installed

  # Arguments
      y: true labels, numpy.array with shape `(n_samples,)`
      y_pred: predicted labels, numpy.array with shape `(n_samples,)`

  # Return
      accuracy, in [0,1]
  """
  y_true = y_true.astype(np.int64)
  assert y_pred.size == y_true.size

  D = max(y_pred.max(), y_true.max()) + 1
  w = np.zeros((D, D), dtype=np.int64)

  for i in range(y_pred.size):
      w[y_pred[i], y_true[i]] += 1

  row_ind, col_ind = linear_sum_assignment(w.max() - w)
  ind = np.concatenate((row_ind.reshape(-1, 1), col_ind.reshape(-1, 1)), axis=1)
  return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size
def match_assignments(y_true, y_pred):
  y_true = y_true.astype(np.int64)
  assert y_pred.size == y_true.size
  D = max(y_pred.max(), y_true.max()) + 1
  w = np.zeros((D, D), dtype=np.int64)

  for i in range(y_pred.size):
      w[y_pred[i], y_true[i]] += 1
  _, col_ind = linear_sum_assignment(w.max() - w)
  return col_ind
def refine_y_pred(y_true, y_pred):
  #print(y_true)
  #print(y_pred)
  pred_to_true = match_assignments(y_true, y_pred)

  y_pred_refined = np.array(list(map(lambda item: pred_to_true[item], y_pred)))
  return y_pred_refined
def parse_text_file(text_file, is_GT, capacity=0):
  """Loads boxes and class labels from a text file in the JRDB format, for every scene present
  in the file (no scene filtering -- see `select_scenes` for that).
  Args:
    text_file: A file object.
    capacity: Maximum number of labeled boxes allowed for each example.
      Default is 0 where there is no limit.
  Returns:
    boxes: A dictionary mapping each unique image key (string) to a list of
      boxes, given as coordinates [y1, x1, y2, x2].
    labels: A dictionary mapping each unique image key (string) to a list of
      integer class lables, matching the corresponding box in `boxes`.
    scores: A dictionary mapping each unique image key (string) to a list of
      score values lables, matching the corresponding label in `labels`. If
      scores are not provided in the csv, then they will be set to 1.0.
    difficult: A dictionary mapping each unique image key (string) to a list of
    difficulty level labels. If difficult is not provided it will be set to 0.
  """
  entries = defaultdict(list)
  boxes = defaultdict(list)
  g_labels = defaultdict(list)
  act_labels = defaultdict(list)
  scores = defaultdict(list)
  difficult = defaultdict(list)

  with open(text_file.name) as r:
      for l in r.readlines():
        row = l[:-1].split(' ')
        assert len(row) == 9, "Wrong number of columns: " + row
        image_key = make_image_key(row[0], row[1])
        x1, y1, x2, y2 = [float(n) for n in row[2:6]]
        score = 1.0
        diff = 0.0
        g_id = int(row[6])
        act_id = int(row[7])

        if is_GT:
          diff = float(row[8])
        else:
          score = float(row[8])

        if capacity < 1 or len(entries[image_key]) < capacity:
          heapq.heappush(entries[image_key],
                       (score, g_id, act_id, diff, y1, x1, y2, x2))
        elif score > entries[image_key][0][0]:
          heapq.heapreplace(entries[image_key],
                          (score, g_id, act_id, diff, y1, x1, y2, x2))

      for image_key in entries:
        # Evaluation API assumes boxes with descending scores
        entry = sorted(entries[image_key], key=lambda tup: -tup[0])
        for item in entry:
          score, g_id, action_id, diff, y1, x1, y2, x2 = item
          boxes[image_key].append([y1, x1, y2, x2])
          g_labels[image_key].append(g_id)
          act_labels[image_key].append(action_id)
          scores[image_key].append(score)
          if diff>=2:
            difficult[image_key].append(1)
          else:
            difficult[image_key].append(0)
  return boxes, g_labels, act_labels, scores, difficult
def select_scenes(data_dicts, seq_set):
  """Filters a tuple of image_key-keyed dicts (as returned by `parse_text_file`) down to only the
  image_keys whose scene id is in `seq_set`. Pure in-memory filter, no file I/O -- lets
  `parse_text_file` be called once per file and reused across every scene in `evaluate()`'s loop.

  Returns fresh list copies (not references into `data_dicts`) -- evaluate()'s task_2/task_3/task_5
  branches mutate the per-image_key label lists in place, and those lists must not be shared across
  different `seq` iterations."""
  return tuple(
      {k: list(v) for k, v in d.items() if int(k.split(',')[0]) in seq_set}
      for d in data_dicts
  )
def read_labelmap(labelmap_file):
  """Reads a labelmap.
  Args:
    labelmap_file: A file object containing a label map protocol buffer.
  Returns:
    labelmap: The label map in the form used by the object_detection_evaluation
      module - a list of {"id": integer, "name": classname } dicts.
    class_ids: A set containing all of the valid class id integers.
  """
  labelmap = []
  class_ids = set()
  name = ""
  for line in labelmap_file:
    if line.startswith("  name:"):
      name = line.split('"')[1]
    elif line.startswith("  id:") or line.startswith("  label_id:"):
      class_id = int(line.strip().split(" ")[-1])
      labelmap.append({"id": class_id, "name": name})
      class_ids.add(class_id)
  return labelmap, class_ids

def evaluate(labelmap, groundtruth, detections, task, mode):
  """Runs evaluations given input files.
  Args:
    seq: the sequence to perform evaluation on.
    labelmap: file object containing map of labels to consider, in pbtxt format
    groundtruth: file object
    detections: file object
    task: the task to be evaluated.
  """
  categories, class_whitelist = read_labelmap(labelmap)


  seq_len = 0
  for line in detections:
    det = line.strip().split(' ')
    seq_len = max(seq_len, int(det[0]))

  detections.seek(0)

  # print(categories)
  # logging.info("CATEGORIES (%d):\n%s", len(categories), pprint.pformat(categories, indent=2))

  import json
  
  country_mapping = {
    "USA": "United States",
    "UAE": "United Arab Emirates",
    "Korea": "South Korea",
    "Czechia": "Czech Republic",
  }
  
  region_modes = {
    "AF": "African",
    "AN": "Anglo",
    "CA": "Confucian Asia",
    "EU": "Eastern Europe",
    "GE": "Germanic Europe",
    "LA": "Latin America",
    "LE": "Latin Europe",
    "ME": "Middle East",
    "NE": "Nordic Europe",
    "SA": "South-East Asia",
    "O": "Other",
  }
  
  source_dataset = "/lp-dev/jmurrugarral/gold_SEKAI_900_3/"
  
  seq_len += 1

  density_offsets = {"scattered": 0, "moderate": 1, "crowded": 2}

  # True for modes that only ever want the pooled aggregate result (never per-scene entries) --
  # needed so the aggregate-tagging logic below doesn't mistake a thin (possibly single-scene)
  # region/region+density subset for a per-scene entry.
  aggregate_only = False

  if mode == "all":
    seqs = [[i] for i in range(seq_len)]
    seqs.append(list(range(seq_len)))

  elif mode == "scattered":
    seqs = [[i] for i in range(seq_len)]
    seqs.append(list(range(0, seq_len, 3)))

  elif mode == "moderate":
    seqs = [[i] for i in range(seq_len)]
    seqs.append(list(range(1, seq_len, 3)))

  elif mode == "crowded":
    seqs = [[i] for i in range(seq_len)]
    seqs.append(list(range(2, seq_len, 3)))

  elif mode in region_modes:

    aggregate_only = True

    target_region = region_modes[mode]

    seqs = [[]]

    for i in range(seq_len):

      json_file = (
        f"{source_dataset}/jsons_step1/clip_{i+1:04d}.json"
      )

      with open(json_file, "r") as f:
        data = json.load(f)

      country = data['country']
      if country in country_mapping.keys():
        country = country_mapping[country]

      globe = country_to_region_globe[country]

      if globe == target_region:
        seqs[0].append(i)

  elif "_" in mode and mode.split("_", 1)[0] in region_modes and mode.split("_", 1)[1] in density_offsets:

    aggregate_only = True

    region_part, density_part = mode.split("_", 1)
    target_region = region_modes[region_part]
    target_offset = density_offsets[density_part]

    seqs = [[]]

    for i in range(seq_len):

      if i % 3 != target_offset:
        continue  # cheap filter first -- skips the JSON open for 2/3 of scenes

      json_file = (
        f"{source_dataset}/jsons_step1/clip_{i+1:04d}.json"
      )

      with open(json_file, "r") as f:
        data = json.load(f)

      country = data['country']
      if country in country_mapping.keys():
        country = country_mapping[country]

      globe = country_to_region_globe[country]

      if globe == target_region:
        seqs[0].append(i)

  #seqs = [[i for i in range(seq_len)]]
  #print(seqs)
  #seqs = [[45, 90, 175, 370, 375, 398, 401, 404, 429, 438, 443, 444, 452, 471, 507, 514, 530, 538, 573, 639, 651, 657, 671, 711, 716]]
  #seqs = seqs + [[i for i in range(2,seq_len,3)]]
  #print(seqs)

  # Parse each file exactly once -- previously read_text_file() re-opened and re-parsed the whole
  # GT/detections file from scratch on every iteration of the loop below (once per scene).
  all_gt = parse_text_file(groundtruth, True, 0)
  all_pred = parse_text_file(detections, False, 0)

  metrics = {}
  for _, seq in enumerate(seqs):
      seq_set = set(seq)
      pascal_evaluator = jr.object_detection_evaluation.PascalDetectionEvaluator(categories, task)

      # Reads the ground truth data.
      gt_boxes, gt_g_labels, gt_act_labels,  _, gt_difficult = select_scenes(all_gt, seq_set)

      for image_key in gt_boxes:
        if task in ['task_1', 'task_4']:
            pascal_evaluator.add_single_ground_truth_image_info(
                image_key, {
                    jr.standard_fields.InputDataFields.groundtruth_boxes:
                        np.array(gt_boxes[image_key], dtype=float),
                    jr.standard_fields.InputDataFields.groundtruth_classes:
                        np.array(gt_act_labels[image_key], dtype=int),
                    jr.standard_fields.InputDataFields.groundtruth_difficult:
                        np.array(gt_difficult[image_key], dtype=float)
                })

        elif task == 'task_2':
            pascal_evaluator.add_single_ground_truth_image_info(
                image_key, {
                    jr.standard_fields.InputDataFields.groundtruth_boxes:
                        np.array(gt_boxes[image_key], dtype=float),
                    jr.standard_fields.InputDataFields.groundtruth_classes:
                        np.array([1 for _ in range(len(gt_g_labels[image_key]))], dtype=int),
                    jr.standard_fields.InputDataFields.groundtruth_difficult:
                        np.array(gt_difficult[image_key], dtype=float)
                })

        elif task == 'task_3':
            gt_classes_dict = {1: [], 2: [], 3: [], 4: [], 5: []}
            gt_classes = gt_g_labels[image_key]
            gt_classes_occ = Counter(gt_classes)
            for k, v in gt_classes_occ.items():
                if v in gt_classes_dict:
                    gt_classes_dict[v].append(k)
                else:
                    gt_classes_dict[5].append(k)

            gt = []
            for g_c in gt_classes:
                for k, v in gt_classes_dict.items():
                    if g_c in v:
                       gt.append(k)

            pascal_evaluator.add_single_ground_truth_image_info(
                image_key, {
                    jr.standard_fields.InputDataFields.groundtruth_boxes:
                        np.array(gt_boxes[image_key], dtype=float),
                    jr.standard_fields.InputDataFields.groundtruth_classes:
                        np.array(gt, dtype=int),
                    jr.standard_fields.InputDataFields.groundtruth_difficult:
                        np.array(gt_difficult[image_key], dtype=float)
                })

        elif task == 'task_5':
            pascal_evaluator.add_single_ground_truth_image_info(
                image_key, {
                    jr.standard_fields.InputDataFields.groundtruth_boxes:
                        np.array(gt_boxes[image_key], dtype=float),
                    jr.standard_fields.InputDataFields.groundtruth_classes:
                        np.array(gt_act_labels[image_key], dtype=int),
                    jr.standard_fields.InputDataFields.groundtruth_difficult:
                        np.array(gt_difficult[image_key], dtype=float)
                })

      # Reads detections data.
      pred_boxes, pred_g_labels, pred_act_labels, pred_scores, _ = select_scenes(all_pred, seq_set)

      for image_key in pred_boxes:

        if task == 'task_1':
            pascal_evaluator.add_single_detected_image_info(task,
                image_key, {
                    jr.standard_fields.DetectionResultFields.detection_boxes:
                        np.array(pred_boxes[image_key], dtype=float),
                    jr.standard_fields.DetectionResultFields.detection_classes:
                        np.array(pred_act_labels[image_key], dtype=int),
                    jr.standard_fields.DetectionResultFields.detection_scores:
                        np.array(pred_scores[image_key], dtype=float)
                })

        elif task in ['task_2', 'task_3']:
            if task == 'task_2':
                gt_refine, det_refine, FPs = refine_group_ids(np.array(pred_boxes[image_key], dtype=float), np.array(pred_scores[image_key], dtype=float),
                                                                    np.array(gt_boxes[image_key], dtype=float),
                                                              groundtruth_is_group_of_list=np.array([False for _ in range(len(gt_boxes[image_key]))], dtype=bool))
                gt_g_id, det_g_id = [], []

                for idx in range(len(gt_refine)):
                    gt_g_id.append(gt_g_labels[image_key][gt_refine[idx]])
                    det_g_id.append(pred_g_labels[image_key][det_refine[idx]])

                refined_det_g_id = refine_y_pred(np.array(gt_g_id), np.array(det_g_id))

                for idx, d in enumerate(range(len(pred_g_labels[image_key]))):
                    if d in det_refine and refined_det_g_id[det_refine.index(d)] == gt_g_id[det_refine.index(d)]:
                        pred_g_labels[image_key][idx] = 1
                    elif d in det_refine and refined_det_g_id[det_refine.index(d)] != gt_g_id[det_refine.index(d)]:
                        pred_g_labels[image_key][idx] = 2
                    elif d in FPs:
                        pred_g_labels[image_key][idx] = 1

            if task == 'task_3':
                #print(gt_boxes)
                if image_key not in gt_boxes or gt_boxes[image_key] == []:
                    continue

                gt_refine, det_refine, FPs = refine_group_ids(np.array(pred_boxes[image_key], dtype=float), np.array(pred_scores[image_key], dtype=float),
                                                                    np.array(gt_boxes[image_key], dtype=float),
                                                              groundtruth_is_group_of_list=np.array([False for _ in range(len(gt_boxes[image_key]))], dtype=bool))
                gt_g_id, det_g_id = [], []
                for idx in range(len(gt_refine)):
                    gt_g_id.append(gt_g_labels[image_key][gt_refine[idx]])
                    det_g_id.append(pred_g_labels[image_key][det_refine[idx]])

                if gt_g_id == []:
                    continue

                refined_det_g_id = refine_y_pred(np.array(gt_g_id), np.array(det_g_id))

                sub_mem_dict = {}  # substitutions!
                for p_r in range(len(det_g_id)):
                    sub_mem_dict[det_g_id[p_r]] = refined_det_g_id[p_r]

                for idx, label in enumerate(pred_g_labels[image_key]):
                    if label in sub_mem_dict:
                        pred_g_labels[image_key][idx] = sub_mem_dict[label]

                det_classes_dict = {1: [], 2: [], 3: [], 4: [], 5: []}
                det_classes = pred_g_labels[image_key]
                det_classes_occ = Counter(det_classes)
                for k, v in det_classes_occ.items():
                    if v in det_classes_dict.keys():
                        det_classes_dict[v].append(k)
                    else:
                        det_classes_dict[5].append(k)

                gt_classes_dict = {1: [], 2: [], 3: [], 4: [], 5: []}
                gt_classes = gt_g_labels[image_key]
                gt_classes_occ = Counter(gt_classes)
                for k, v in gt_classes_occ.items():
                    if v in gt_classes_dict:
                        gt_classes_dict[v].append(k)
                    else:
                        gt_classes_dict[5].append(k)
		
                for idx, d in enumerate(range(len(pred_g_labels[image_key]))):
                    if d in det_refine and refined_det_g_id[det_refine.index(d)] == gt_g_id[det_refine.index(d)]:
                        pred_g_labels[image_key][idx] = \
                        [int(key) for (key, value) in gt_classes_dict.items() if gt_g_id[det_refine.index(d)] in value][0]
                    elif d in det_refine and refined_det_g_id[det_refine.index(d)] != gt_g_id[det_refine.index(d)]:
                        pred_g_labels[image_key][idx] = 6
                    elif d in FPs:
                        temp = [int(key) for (key, value) in gt_classes_dict.items() if pred_g_labels[image_key][idx] in value]
                        if len(temp) == 0:
                            pred_g_labels[image_key][idx] = \
                                [int(key) for (key, value) in det_classes_dict.items() if pred_g_labels[image_key][idx] in value][0]
                        else:
                            pred_g_labels[image_key][idx] = temp[0]

            pascal_evaluator.add_single_detected_image_info(task,
                image_key, {
                    jr.standard_fields.DetectionResultFields.detection_boxes:
                        np.array(pred_boxes[image_key], dtype=float),
                    jr.standard_fields.DetectionResultFields.detection_classes:
                        np.array(pred_g_labels[image_key], dtype=int),
                    jr.standard_fields.DetectionResultFields.detection_scores:
                        np.array(pred_scores[image_key], dtype=float)
                })

        elif task == 'task_4':
            pascal_evaluator.add_single_detected_image_info(task,
                image_key, {
                    jr.standard_fields.DetectionResultFields.detection_boxes:
                        np.array(pred_boxes[image_key], dtype=float),
                    jr.standard_fields.DetectionResultFields.detection_classes:
                        np.array(pred_act_labels[image_key], dtype=int),
                    jr.standard_fields.DetectionResultFields.detection_scores:
                        np.array(pred_scores[image_key], dtype=float)
                })

        elif task == 'task_5':
            gt_refine, det_refine, _ = refine_group_ids(np.array(pred_boxes[image_key], dtype=float), np.array(pred_scores[image_key], dtype=float),
                                                                np.array(gt_boxes[image_key], dtype=float),
                                                          groundtruth_is_group_of_list=np.array([False for _ in range(len(gt_boxes[image_key]))], dtype=bool))
            gt_g_id, det_g_id = [], []
            for idx in range(len(gt_refine)):
                gt_g_id.append(gt_g_labels[image_key][gt_refine[idx]])
                det_g_id.append(pred_g_labels[image_key][det_refine[idx]])
            refined_det_g_id = refine_y_pred(np.array(gt_g_id), np.array(det_g_id))

            for idx, d in enumerate(range(len(pred_g_labels[image_key]))):
                if d in det_refine and refined_det_g_id[det_refine.index(d)] != gt_g_id[det_refine.index(d)]:
                    pred_act_labels[image_key][idx] = 27

            pascal_evaluator.add_single_detected_image_info(task,
                image_key, {
                    jr.standard_fields.DetectionResultFields.detection_boxes:
                        np.array(pred_boxes[image_key], dtype=float),
                    jr.standard_fields.DetectionResultFields.detection_classes:
                        np.array(pred_act_labels[image_key], dtype=int),
                    jr.standard_fields.DetectionResultFields.detection_scores:
                        np.array(pred_scores[image_key], dtype=float)
                })

      if aggregate_only or len(seq)>1:
        k = 'all'
      else:
        k = seq[0]


      metrics[k] = pascal_evaluator.evaluate()
  return metrics

def parse_arguments():
  """Parses command-line flags.
  Returns:
    args: a named tuple containing three file objects args.labelmap,
    args.groundtruth, args.detections and args.task.
  """
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "-l",
      "--labelmap",
      help="Filename of label map",
      type=argparse.FileType("r"),
      default="./label_map/task_1.pbtxt")

  parser.add_argument(
      "-g",
      "--groundtruth",
      help="text file containing ground truth.",
      type=argparse.FileType("r"),
      default="./gt.txt",
      required=True)

  parser.add_argument(
      "-d",
      "--detections",
      help="text file containing inferred action detections.",
      type=argparse.FileType("r"),
      default="./det.txt",
      required=True)

  parser.add_argument(
      "-t",
      "--task",
      help="The task to be evaluated. task_1: individual_action, task_2: grouping_1, task_3: grouping_2,"
           " task_4: social_activity_1,task_5: social_activity_2",
      #type=str,
      default="task_1",
      required=True)

  parser.add_argument('-o', '--output', help='A desired partiion', default="all", required=True)

  return parser.parse_args()

def main():
      logging.basicConfig(level=logging.INFO)
      args = parse_arguments()
      metrics = evaluate(args.labelmap, args.groundtruth, args.detections, args.task, args.output)
      for accuracy in metrics['all'].values():
        print(accuracy)

      #pprint.pprint(metrics, indent=2)

if __name__ == "__main__":
  main()
