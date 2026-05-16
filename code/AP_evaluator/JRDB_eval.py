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
def read_text_file(seq, text_file, is_GT, capacity=0):
  """Loads boxes and class labels from a text file in the JRDB format and the sequences of data to be evaluated.
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
        if int(row[0]) in seq:
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
  
  seq_len+=1

  if mode == 'all':
    seqs = [[i] for i in range(seq_len)]
    seqs = seqs + [[i for i in range(seq_len)]]
  elif mode == 'scattered':
    seqs = [[i] for i in range(seq_len)]
    seqs = seqs + [[i for i in range(0,seq_len,3)]]
  elif mode == 'moderate':
    seqs = [[i] for i in range(seq_len)]
    seqs = seqs + [[i for i in range(1,seq_len,3)]]
  elif mode == 'crowded':
    seqs = [[i] for i in range(seq_len)]
    seqs.append([i for i in range(2, seq_len, 3)])
  elif mode == 'AF':
    seqs = [[22, 40, 88, 95, 105, 112, 131, 140, 152, 178, 381, 414,
             454, 485, 497, 518, 533, 551, 575, 594, 613, 681, 685,
             699, 708]]
  elif mode == 'AN':
    seqs = [[0, 3, 4, 6, 13, 14, 15, 24, 27, 31, 35, 37, 38, 39, 42,
             44, 47, 50, 58, 61, 62, 68, 75, 80, 86, 91, 94, 103, 106,
             110, 113, 114, 116, 119, 122, 126, 127, 128, 130, 133,
             134, 136, 142, 147, 148, 149, 151, 154, 158, 159, 165,
             168, 172, 176, 363, 366, 367, 373, 374, 378, 384, 385,
             391, 392, 394, 399, 400, 403, 405, 412, 420, 422, 428,
             434, 437, 439, 448, 450, 451, 458, 463, 464, 466, 468,
             470, 472, 476, 477, 481, 482, 484, 486, 491, 493, 495,
             501, 503, 505, 509, 511, 519, 524, 528, 529, 532, 535,
             536, 540, 541, 545, 547, 555, 557, 559, 561, 565, 567,
             569, 570, 574, 576, 579, 580, 583, 585, 590, 593, 596,
             601, 602, 603, 605, 611, 615, 618, 623, 624, 625, 628,
             629, 630, 632, 635, 636, 643, 656, 660, 662, 664, 668,
             670, 678, 680, 686, 687, 689, 690, 703, 710, 717, 718,
             719]]
  elif mode == 'CA':
    seqs = [[1, 8, 10, 12, 18, 25, 26, 30, 32, 33, 34, 51, 52, 53,
             67, 70, 72, 79, 93, 101, 109, 120, 135, 139, 141, 153,
             156, 161, 162, 163, 166, 167, 170, 171, 177, 179, 180,
             369, 372, 382, 387, 409, 410, 413, 419, 424, 425, 431,
             432, 433, 435, 441, 445, 446, 453, 455, 457, 460, 461,
             462, 469, 474, 475, 478, 480, 488, 492, 494, 508, 516,
             522, 523, 526, 542, 546, 548, 549, 554, 558, 566, 571,
             572, 578, 581, 584, 589, 595, 600, 604, 608, 610, 612,
             614, 616, 619, 621, 626, 633, 634, 637, 638, 640, 641,
             649, 652, 661, 663, 666, 667, 677, 682, 683, 692, 693,
             695, 697, 705, 709]]
  elif mode == 'EU':
    seqs = [[45, 90, 175, 370, 375, 398, 401, 404, 429, 438, 443, 444,
             452, 471, 507, 514, 530, 538, 573, 639, 651, 657, 671,
             711, 716]]
  elif mode == 'GE':
    seqs = [[89, 129, 137, 143, 174, 383, 395, 467, 489, 504, 512, 517,
             527, 597, 659, 684, 691]]
  elif mode == 'LA':
    seqs = [[21, 48, 49, 66, 73, 107, 125, 155, 169, 362, 364, 365,
             368, 371, 376, 380, 390, 479, 496, 498, 506, 510, 553,
             588, 606, 654, 665, 669, 702]]
  elif mode == 'LE':
    seqs = [[5, 9, 19, 69, 71, 87, 100, 102, 111, 115, 146, 386, 389,
             402, 423, 427, 440, 442, 449, 456, 483, 490, 556, 607,
             620, 622, 627, 642, 644, 645, 648, 653, 672, 673, 675,
             688, 694]]
  elif mode == 'ME':
    seqs = [[16, 20, 36, 41, 46, 60, 63, 64, 65, 74, 77, 81, 83, 84,
             96, 97, 98, 104, 121, 123, 138, 173, 361, 396, 397, 411,
             417, 421, 426, 436, 465, 473, 500, 515, 521, 531, 534,
             537, 539, 543, 544, 562, 563, 564, 568, 587, 591, 599,
             647, 650, 655, 674, 701, 713, 714, 715]]
  elif mode == 'NE':
    seqs = [[11, 29, 78, 99, 108, 124, 150, 160, 377, 407, 408, 415,
             418, 447, 499, 520, 525, 560, 592, 598, 609, 617, 631,
             679, 704, 706]]
  elif mode == 'SA':
    seqs = [[2, 7, 23, 28, 43, 54, 55, 56, 57, 59, 76, 82, 85, 92,
             117, 118, 144, 145, 157, 164, 379, 388, 393, 406, 416,
             430, 459, 487, 502, 513, 550, 552, 577, 582, 586, 658,
             696, 700, 707, 712]]
  elif mode == 'O':
    seqs = [[17, 132, 646, 676, 698]]

  #seqs = [[i for i in range(seq_len)]]
  #print(seqs)
  #seqs = [[45, 90, 175, 370, 375, 398, 401, 404, 429, 438, 443, 444, 452, 471, 507, 514, 530, 538, 573, 639, 651, 657, 671, 711, 716]]
  #seqs = seqs + [[i for i in range(2,seq_len,3)]]
  #print(seqs)

  metrics = {}
  for _, seq in enumerate(seqs):
      pascal_evaluator = jr.object_detection_evaluation.PascalDetectionEvaluator(categories, task)

      # Reads the ground truth data.
      gt_boxes, gt_g_labels, gt_act_labels,  _, gt_difficult = read_text_file(seq, groundtruth, True, 0)

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
      pred_boxes, pred_g_labels, pred_act_labels, pred_scores, _ = read_text_file(seq, detections, False, 0)

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
                if gt_boxes[image_key] == []:
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

      if len(seq)>1:
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
