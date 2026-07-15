import cv2
import math
import os
import dspy
import json
import argparse
from PIL import Image

from ablation_prompts import IdentifyGroups_Direction, IdentifyGroups_Transitive, IdentifyGroups_DirectionTransitive, vlm_IdentifyGroups_DirectionImage, vlm_IdentifyGroups_TransitiveImage, vlm_IdentifyGroups_DirectionTransitiveImage, IdentifyGroups_withbboxes, vlm_IdentifyGroupsImage_withbboxes, vlm_IdentifyGroups2DImage, IdentifyGroups2D, vlm_IdentifyGroups2DText, vlm_IdentifyGroups_DirectionText, vlm_IdentifyGroups_TransitiveText, vlm_IdentifyGroups_DirectionTransitiveText

from prompts import IdentifyGroups

from prompts import vlm_GroupsQAonlyImage, vlm_IdentifyGroupsImage, vlm_IdentifyGroupsImage_idsonly, baseline2

from prompts import vlm_IdentifyGroupsText

from prompts import IdentifyGroups_AllFrames, vlm_IdentifyGroups_AllFramesText, vlm_IdentifyGroups_AllFramesImage, vlm_GroupsQAonlyFullImage, full_baseline2, vlm_IdentifyGroups_AllFramesImage_withbboxes, vlm_IdentifyGroups_AllFramesImage_idsonly

CV2_COLORS = [ 
    (0, 0, 255),      # Red
    (0, 255, 0),      # Green
    (255, 0, 0),      # Blue
    (0, 255, 255),    # Yellow
    (255, 255, 0),    # Cyan
    (255, 0, 255),    # Magenta
    (0, 0, 0),        # Black
    (255, 255, 255),  # White

    (128, 0, 0),      # Maroon
    (0, 128, 0),      # Dark Green
    (0, 0, 128),      # Navy
    (128, 128, 0),    # Olive
    (128, 0, 128),    # Purple
    (0, 128, 128),    # Teal
    (192, 192, 192),  # Silver
    (128, 128, 128),  # Gray

    (0, 165, 255),    # Orange (OpenCV-style BGR)
    (203, 192, 255),  # Pink (light magenta)
    (42, 42, 165),    # Brown
    (60, 179, 113),   # Medium Sea Green
    (147, 20, 255),   # Deep Pink
    (50, 205, 50),    # Lime Green
    (255, 105, 180),  # Hot Pink
    (19, 69, 139),    # Dark Slate Blue
]

def _frame_xyz_list(frame):
    """Return the flat list of {'person_id','x','y',...} dicts for a frame, handling both the
    plain list[dict] shape (p1/p2/.../p5) and the two-part [[bbox dicts],[xyz dicts]] shape
    (p1_bbox/p1_visual/p1_visual_only, see get_frame_bboxes/get_allframes_bboxes)."""
    if len(frame) == 2 and isinstance(frame[0], list) and isinstance(frame[1], list):
        return frame[1]
    return frame


def get_movement_direction(input_text, target_frame, stationary_threshold=0.3):
    """For each person in the target frame, find their earliest position among the frames
    before it (not just frame 0 -- handles people who enter partway through the window) and
    compute an 8-way directional label (or 'stationary' below stationary_threshold) for their
    net displacement to the target frame. Labels are relative to the data's own x/y axes, not
    true geographic compass directions."""
    target_detections = _frame_xyz_list(input_text[target_frame - 1])
    pre_target_frames = [_frame_xyz_list(f) for f in input_text[:target_frame - 1]]

    earliest_position = {}
    for frame in pre_target_frames:
        for det in frame:
            pid = det['person_id']
            if pid not in earliest_position:
                earliest_position[pid] = (det['x'], det['y'])

    COMPASS = ['E', 'NE', 'N', 'NW', 'W', 'SW', 'S', 'SE']
    directions = {}
    for det in target_detections:
        pid = det['person_id']
        if pid not in earliest_position:
            directions[pid] = 'stationary'
            continue
        x0, y0 = earliest_position[pid]
        dx, dy = det['x'] - x0, det['y'] - y0
        if (dx ** 2 + dy ** 2) ** 0.5 < stationary_threshold:
            directions[pid] = 'stationary'
        else:
            angle = math.degrees(math.atan2(dy, dx)) % 360
            directions[pid] = COMPASS[int((angle + 22.5) % 360 // 45)]
    return directions


def full_inference(dspy_module, input_text, target_frame, mode='llm', frame_path='', prompt_method='p1'):
    is_error = False
    try:
        is_bbox_shape = prompt_method in ('p1_bbox', 'p1_visual', 'p1_visual_only')
        raw_target = input_text[target_frame - 1]
        target_bboxes, target_xyz = raw_target if is_bbox_shape else (None, raw_target)

        movement = get_movement_direction(input_text, target_frame)
        if prompt_method == 'p1_visual_only':
            # No numeric channel for this method -- movement_direction is skipped, the image is
            # the only signal (see the p1_visual_only decision this was built with).
            enriched_xyz = target_xyz
        else:
            enriched_xyz = [
                {**det, 'movement_direction': movement.get(det['person_id'], 'stationary')}
                for det in target_xyz
            ]

        if mode == 'llm' or mode =='vlm_text':
            predictions = dspy_module(all_frames=[enriched_xyz], target_frame=target_frame)
        if mode == 'vlm_image':
            img_folder = frame_path[:-10]
            step = 3 if target_frame in (22, 42) else 1
            frame_indices = list(range(1, target_frame + 1, step))
            if frame_indices[-1] != target_frame:
                frame_indices.append(target_frame)
            video = [dspy.Image.from_file(f'{img_folder}{str(i).zfill(5)}.jpeg') for i in frame_indices]

            if prompt_method == 'p1_bbox':
                image = dspy.Image.from_file(frame_path)
                predictions = dspy_module(image=image, video=video, boundingboxes=target_bboxes, detections=enriched_xyz, target_frame=target_frame)
            elif prompt_method == 'p1_visual':
                annotated_image = draw_bboxes_with_ids(frame_path, target_bboxes)
                predictions = dspy_module(image=annotated_image, video=video, detections=enriched_xyz, target_frame=target_frame)
            elif prompt_method == 'p1_visual_only':
                annotated_image = draw_bboxes_with_ids(frame_path, target_bboxes)
                person_ids = [d['person_id'] for d in target_bboxes]
                predictions = dspy_module(image=annotated_image, video=video, person_ids=person_ids, target_frame=target_frame)
            else:
                image = dspy.Image.from_file(frame_path)
                predictions = dspy_module(image=image, video=video, all_frames=[enriched_xyz], target_frame=target_frame)

        output = predictions.groups
    except Exception as e:
        output = str(e)
        is_error = True
    return output, is_error

def full_inference_wrapper(lm, dspy_module, input_text, target_frame, mode='llm', frame_path='', prompt_method='p1'):

    if mode == 'llm' or mode == 'vlm_text':
        output, is_error = full_inference(dspy_module, input_text, target_frame, mode, '', prompt_method)
    elif mode == 'vlm_image':
        output, is_error = full_inference(dspy_module, input_text, target_frame, mode, frame_path, prompt_method)

    res = {}
    if not is_error:
        res['hist'] = str(lm.history[-1])
        res['groups'] = output
        res['error'] = None
    else:
        res['hist'] = str(lm.history[-1]) if len(lm.history) > 0 else ''
        res['groups'] = None
        res['error'] = output
    return res 

def draw_bboxes_with_ids(image_path, boundingboxes):
    """Overlay each person's box (boundingboxes' t/l/b/r pixel corners, see get_frame_bboxes'
    p1_bbox/p1_visual/p1_visual_only branch) and track id on the frame image, returning a
    dspy.Image built from the annotated array. Used by inference() for prompt_method='p1_visual'
    (alongside the numeric 'detections' field) and 'p1_visual_only' (image only, no numeric
    coordinates at all) so the VLM sees the box/id correspondence visually instead of, or in
    addition to, the text input."""
    img = cv2.imread(image_path)
    color = (255, 0, 0)  # blue (BGR), same for every person
    for box in boundingboxes:
        t, l, b, r = box['t'], box['l'], box['b'], box['r']
        cv2.rectangle(img, (int(t), int(l)), (int(b), int(r)), color, 2)
        cv2.putText(img, str(box['person_id']), (int(t), max(int(l) - 5, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return dspy.Image.from_PIL(Image.fromarray(img_rgb))


def inference(dspy_module, input_text, mode='llm', image_path='optional', prompt='p1'):
    is_error = False
    try:
        if prompt == 'p1_bbox':
            if mode == 'llm' or mode == 'vlm_text':
                predictions = dspy_module(boundingboxes=input_text[0], detections=input_text[1])
            elif mode == 'vlm_image':
                predictions = dspy_module(image=dspy.Image.from_file(image_path), boundingboxes=input_text[0], detections=input_text[1])
        elif prompt == 'p1_visual':
            if mode == 'llm' or mode == 'vlm_text':
                predictions = dspy_module(boundingboxes=input_text[0], detections=input_text[1])
            elif mode == 'vlm_image':
                annotated_image = draw_bboxes_with_ids(image_path, input_text[0])
                predictions = dspy_module(image=annotated_image, detections=input_text[1])
        elif prompt == 'p1_visual_only':
            if mode == 'vlm_image':
                annotated_image = draw_bboxes_with_ids(image_path, input_text[0])
                person_ids = [d['person_id'] for d in input_text[0]]
                predictions = dspy_module(image=annotated_image, person_ids=person_ids)
        else:
            if mode == 'llm' or mode == 'vlm_text':
                predictions = dspy_module(detections=input_text)
            elif mode == 'vlm_image':
                predictions = dspy_module(image=dspy.Image.from_file(image_path), detections=input_text)
 
        output = predictions.groups
    except Exception as e:
        output = str(e)
        is_error = True
    return output, is_error


def inference_wrapper(lm, dspy_module, input_text, mode='llm', image_path='optional', prompt='optional'):
    if mode == 'llm' or mode == 'vlm_text':
        output, is_error = inference(dspy_module, input_text, mode, prompt=prompt)
    elif mode == 'vlm_image':
        output, is_error = inference(dspy_module, input_text, mode, image_path=image_path, prompt=prompt)

    #print(output)
    #print(is_error)
    res = {}
    #print(lm.history)
    if not is_error:
        res['hist'] = str(lm.history[-1]) 
        res['groups'] = output
        res['error'] = None
    else:
        res['hist'] = str(lm.history[-1]) if len(lm.history) > 0 else ''
        res['groups'] = None
        res['error'] = output
    return res 

def get_dspy_cot(mode, prompt_method):
    
    use_direction = False
    if mode == 'llm':
        if prompt_method == 'p1':
            dspy_cot = dspy.ChainOfThought(IdentifyGroups)
        elif prompt_method == 'p2':
            dspy_cot = dspy.ChainOfThought(IdentifyGroups_Direction)
            use_direction = True
        elif prompt_method == 'p3':
            dspy_cot = dspy.ChainOfThought(IdentifyGroups_Transitive)
        elif prompt_method == 'p4':
            dspy_cot = dspy.ChainOfThought(IdentifyGroups_DirectionTransitive)
            use_direction = True
        elif prompt_method == 'p5':
            dspy_cot = dspy.ChainOfThought(IdentifyGroups2D)
        elif prompt_method == 'p1_bbox':
            dspy_cot = dspy.ChainOfThought(IdentifyGroups_withbboxes)
    elif mode == 'vlm_image':
        if prompt_method == 'baseline1':
            dspy_cot = dspy.ChainOfThought(vlm_GroupsQAonlyImage)
        elif prompt_method == 'baseline2':
            dspy_cot = baseline2()
        elif prompt_method == 'p1':
            dspy_cot = dspy.ChainOfThought(vlm_IdentifyGroupsImage)
        elif prompt_method == 'p1_visual':
            dspy_cot = dspy.ChainOfThought(vlm_IdentifyGroupsImage)
        elif prompt_method == 'p1_visual_only':
            dspy_cot = dspy.ChainOfThought(vlm_IdentifyGroupsImage_idsonly)
        elif prompt_method == 'p2':
            dspy_cot = dspy.ChainOfThought(vlm_IdentifyGroups_DirectionImage)
            #use_direction = True
        elif prompt_method == 'p3':
            dspy_cot = dspy.ChainOfThought(vlm_IdentifyGroups_TransitiveImage)
        elif prompt_method == 'p4':
            dspy_cot = dspy.ChainOfThought(vlm_IdentifyGroups_DirectionTransitiveImage)
            #use_direction = True
        elif prompt_method == 'p5':
            dspy_cot = dspy.ChainOfThought(vlm_IdentifyGroups2DImage)
        elif prompt_method == 'p1_bbox':
            dspy_cot = dspy.ChainOfThought(vlm_IdentifyGroupsImage_withbboxes)

    #elif mode == 'vlm_text':
    #    if prompt_method == 'p1':
    #        dspy_cot = dspy.ChainOfThought(vlm_IdentifyGroupsText)
    #    elif prompt_method == 'p2':
    #        dspy_cot = dspy.ChainOfThought(vlm_IdentifyGroups_DirectionText)
    #        use_direction = True
    #    elif prompt_method == 'p3':
    #        dspy_cot = dspy.ChainOfThought(vlm_IdentifyGroups_TransitiveText)
    #    elif prompt_method == 'p4':
    #        dspy_cot = dspy.ChainOfThought(vlm_IdentifyGroups_DirectionTransitiveText)
    #        use_direction = True
    #    elif prompt_method == 'p5':
    #        dspy_cot = dspy.ChainOfThought(vlm_IdentifyGroups2DText)

    return (dspy_cot, use_direction)

def get_full_dspy_cot(mode, prompt_method):
    
    use_direction = False
    if mode == 'llm':
        if prompt_method == 'p1':
            dspy_cot = dspy.ChainOfThought(IdentifyGroups_AllFrames)
    elif mode == 'vlm_text':
        if prompt_method == 'p1':
            dspy_cot = dspy.ChainOfThought(vlm_IdentifyGroups_AllFramesText)
    elif mode == 'vlm_image':
        if prompt_method == 'baseline1':
            dspy_cot = dspy.ChainOfThought(vlm_GroupsQAonlyFullImage)
        if prompt_method == 'baseline2':
            dspy_cot = full_baseline2()
        elif prompt_method == 'p1':
            dspy_cot = dspy.ChainOfThought(vlm_IdentifyGroups_AllFramesImage)
        elif prompt_method == 'p1_bbox':
            dspy_cot = dspy.ChainOfThought(vlm_IdentifyGroups_AllFramesImage_withbboxes)
        elif prompt_method == 'p1_visual':
            dspy_cot = dspy.ChainOfThought(vlm_IdentifyGroups_AllFramesImage)
        elif prompt_method == 'p1_visual_only':
            dspy_cot = dspy.ChainOfThought(vlm_IdentifyGroups_AllFramesImage_idsonly)

    return (dspy_cot, use_direction)

def get_allframes_bboxes(data, use_direction, depth_method, prompt_method):

    if data['dataset'] == 'JRDB_gold':
        depth_method = '3D'

    is_bbox_shape = prompt_method in ('p1_bbox', 'p1_visual', 'p1_visual_only')

    all_frames, bboxes = [], []
    for frame in data['frames']:
        frame_input_data = [[], []] if is_bbox_shape else []
        personid2bbox = {}
        for i,det in enumerate(frame['detections']):

            personid2bbox[det['track_id']] = det['bbox']
            if prompt_method == 'p5':
                frame_input_data.append({'person_id': det['track_id'], 'x': (det['bbox'][0]+det['bbox'][2])//2, 'y': (det['bbox'][1]+det['bbox'][3])//2})
            elif is_bbox_shape:
                frame_input_data[1].append({'person_id': det['track_id'], 'x': round(det[depth_method][0],4), 'y': round(det[depth_method][1],4), 'z': round(det[depth_method][2],4)})
                frame_input_data[0].append({'person_id': det['track_id'], 't': round(det['bbox'][0],2), 'l': round(det['bbox'][1],2), 'b': round(det['bbox'][2],2), 'r': round(det['bbox'][3],2)})
            else:
                if use_direction:
                    frame_input_data.append({'person_id': det['track_id'], 'x': round(det[depth_method][0],4), 'y': round(det[depth_method][1],4), 'z': round(det[depth_method][2],4), 'direction':det['direction']})
                else:
                    frame_input_data.append({'person_id': det['track_id'], 'x': round(det[depth_method][0],4), 'y': round(det[depth_method][1],4), 'z': round(det[depth_method][2],4)})
        all_frames.append(frame_input_data)
        bboxes.append(personid2bbox)

    return (all_frames, bboxes)




def get_frame_bboxes(data, use_direction, depth_method, frame_id, prompt_method):

    #print(frame_id)

    if data['dataset'] == 'JRDB_gold':
        depth_method = '3D'

    frame_found = False
    for frame in data['frames']:
        if frame['frame_id'] == frame_id:
            frame_found = True
            break
    
    if not frame_found:
        print(f"Frame ID {frame_id} not found in the data.")
        return ([],{})

    if prompt_method == 'p1_bbox' or prompt_method == 'p1_visual' or prompt_method == 'p1_visual_only':
        frame_input_data = [[],[]]
    else:
        frame_input_data = []

    personid2bbox = {}
    for i,det in enumerate(frame['detections']):
        personid2bbox[det['track_id']] = det['bbox']
        if prompt_method == 'p5':
            frame_input_data.append({'person_id': det['track_id'], 'x': (det['bbox'][0]+det['bbox'][2])//2, 'y': (det['bbox'][1]+det['bbox'][3])//2})
        elif prompt_method == 'p1_bbox' or prompt_method == 'p1_visual' or prompt_method == 'p1_visual_only':
            frame_input_data[1].append({'person_id': det['track_id'], 'x': round(det[depth_method][0],4), 'y': round(det[depth_method][1],4), 'z': round(det[depth_method][2],4) })
            frame_input_data[0].append({'person_id': det['track_id'], 't': round(det['bbox'][0],2), 'l': round(det['bbox'][1],2), 'b': round(det['bbox'][2],2), 'r': round(det['bbox'][3],2)})
        else:
            if use_direction:
                frame_input_data.append({'person_id': det['track_id'], 'x': round(det[depth_method][0],4), 'y': round(det[depth_method][1],4), 'z': round(det[depth_method][2],4), 'direction':det['direction']})
            else:
                frame_input_data.append({'person_id': det['track_id'], 'x': round(det[depth_method][0],4), 'y': round(det[depth_method][1],4), 'z': round(det[depth_method][2],4)})
        
    return (frame_input_data, personid2bbox)


def save_frame(output, personid2bbox, res_path, save_filename, frame_path, save_image, model, mode, depth_method, prompt_method, frame_id):

    if save_image == True:
        img = cv2.imread(frame_path)

        if prompt_method == 'baseline1' or prompt_method == 'baseline2' :
            for ind, p in enumerate(output['groups']):
                for bbox in p:
                    #print(len(bbox))
                    if len(bbox) == 4:
                        if 'Qwen3' in model or 'Cosmos' in model:
                            height, width, _ = img.shape 
                            img = cv2.rectangle(img, (int(bbox[0]/ 1000 *width), int(bbox[1]/1000*height)), (int(bbox[2]/1000*width), int(bbox[3]/1000*height)), CV2_COLORS[ind%len(CV2_COLORS)], 2)
                        else:
                            img = cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), CV2_COLORS[ind%len(CV2_COLORS)], 2)
        else:
            for i, group in enumerate(output['groups']):
                for person_id in group:
                    if str(person_id) in personid2bbox:
                        xl, yl, x2, y2 = personid2bbox[str(person_id)]
                        cv2.rectangle(img, (int(xl),int(yl)), (int(x2),int(y2)), CV2_COLORS[i%len(CV2_COLORS)], 2)
                    if person_id in personid2bbox:
                        xl, yl, x2, y2 = personid2bbox[person_id]
                        cv2.rectangle(img, (int(xl),int(yl)), (int(x2),int(y2)), CV2_COLORS[i%len(CV2_COLORS)], 2)



        res_path = os.path.join(res_path,model.split("/")[1],mode,depth_method,prompt_method,str(frame_id),save_filename,)
        os.makedirs(res_path, exist_ok=True)
        save_path = os.path.join(res_path, 'result.png')
        cv2.imwrite(save_path, img)
            
        with open(f'{res_path}/{save_filename}.json', "w") as f:
            json.dump(output, f, indent=4)
        
    else:

        res_path = os.path.join(res_path,model.split("/")[1],mode,depth_method,prompt_method,str(frame_id),save_filename,)
        os.makedirs(res_path, exist_ok=True)
        with open(f'{res_path}/{save_filename}.json', "w") as f:
            json.dump(output, f, indent=4)

def save_full_frame(output, bboxes, res_path, save_filename, frame_path, save_image, model, mode, depth_method, prompt_method, frame_id):

    if save_image == True:
        img = cv2.imread(frame_path)

        if prompt_method == 'baseline1' or prompt_method == 'baseline2':
            for ind, p in enumerate(output['groups']):
                for bbox in p:
                    if len(bbox) == 4:
                        if 'Qwen3' in model or 'Cosmos' in model:
                            height, width, _ = img.shape 
                            img = cv2.rectangle(img, (int(bbox[0]/ 1000 *width), int(bbox[1]/1000*height)), (int(bbox[2]/1000*width), int(bbox[3]/1000*height)), CV2_COLORS[ind%len(CV2_COLORS)], 2)
                        else:
                            img = cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), CV2_COLORS[ind%len(CV2_COLORS)], 2)
        else:
            for i, group in enumerate(output['groups']):
                for person_id in group:
                    if str(person_id) in bboxes[frame_id-1]:
                        xl, yl, x2, y2 = bboxes[frame_id-1][str(person_id)]
                        cv2.rectangle(img, (int(xl),int(yl)), (int(x2),int(y2)), CV2_COLORS[i%len(CV2_COLORS)], 2)
                    if person_id in bboxes[frame_id-1]:
                        xl, yl, x2, y2 = bboxes[frame_id-1][person_id]
                        cv2.rectangle(img, (int(xl),int(yl)), (int(x2),int(y2)), CV2_COLORS[i%len(CV2_COLORS)], 2)

        # configure this path                
        res_path = os.path.join(res_path,model.split("/")[1],mode,depth_method,prompt_method,str(frame_id),save_filename,)
        os.makedirs(res_path, exist_ok=True)
        save_path = os.path.join(res_path, 'result.png')
        cv2.imwrite(save_path, img)
    
        #save output json
        with open(f'{res_path}/{save_filename}.json', "w") as f:
            json.dump(output, f, indent=4)
    
    else:

        res_path = os.path.join(res_path,model.split("/")[1],mode,depth_method,prompt_method,str(frame_id),save_filename,)
        os.makedirs(res_path, exist_ok=True)
        with open(f'{res_path}/{save_filename}.json', "w") as f:
            json.dump(output, f, indent=4)


def save_json(data, res_path, save_filename, model, mode, depth_method, prompt_method):
    res_path = os.path.join(res_path, model.split('/')[1]+'/'+ mode + '/'+ depth_method+'/'+prompt_method, save_filename,)
    os.makedirs(res_path, exist_ok=True)
    with open(f'{res_path}/{save_filename}.json', "w") as f:
        json.dump(data, f, indent=4)


def parse_args():
    parser = argparse.ArgumentParser(description="Metadata JSON file")
    parser.add_argument('filename', type=str)
    parser.add_argument('setting', type=str, choices=['single', 'full'])
    parser.add_argument('mode', type=str, choices=['llm', 'vlm_text', 'vlm_image'])
    parser.add_argument('model', type=str)
    parser.add_argument('frame_id', type=int)
    parser.add_argument('--depth_method', type=str, choices=['naive_3D_60FOV', 'naive_3D_110FOV', 'naive_3D_160FOV', 'unidepth_3D', 'detany_3D','wilddet_3D'], default='detany_3D')
    parser.add_argument('--prompt_method', type=str, choices=['baseline1','baseline2','p1', 'p2', 'p3', 'p4', 'p5','p1_bbox','p1_visual','p1_visual_only'], default='p1')
    parser.add_argument('--api_base', type=str, default="http://localhost:8000/v1")
    parser.add_argument('--api_key', type=str, default="testkey")
    parser.add_argument('--temperature', type=float, default=0.6)
    parser.add_argument('--max_tokens', type=int, default=24000)
    parser.add_argument('--frame_path', type=str, default='VBIG_dataset/videos_frames', help='Path to frame for visualization')
    parser.add_argument("--save_image", action="store_true", help="Save Image")
    parser.add_argument('--start', type=str, default = 0)
    return parser.parse_args()


def parse_numbers(s):
    numbers = []
    
    if ':' in s:
        parts = s.split(':')
        if len(parts) == 2:
            start, end = map(int, parts)
            numbers.extend(range(start, end + 1))
        elif len(parts) == 3:
            start, end, step = map(int, parts)
            numbers.extend(range(start, end + 1, step))
    else:
        parts = s.split(',')
        for part in parts:
            if '-' in part:
                low, high = map(int, part.split('-'))
                numbers.extend(range(low, high + 1))
            else:
                numbers.append(int(part))
    return numbers


def parse_args_allframes():
    parser = argparse.ArgumentParser(description="Metadata JSON file")
    parser.add_argument('filename', type=str)
    parser.add_argument('setting', type=str, choices=['single', 'full'])
    parser.add_argument('mode', type=str, choices=['llm', 'vlm_text', 'vlm_image'])
    parser.add_argument('model', type=str)
    # parser.add_argument('frame_id', type=int)
    parser.add_argument(
        "--frame_ids",
        type=parse_numbers,
        required=True,
        help="Comma-separated numbers or ranges, e.g. 1,2,5-7,10 or 0:51 or 0:51:15"
    )
    parser.add_argument('--depth_method', type=str, choices=['naive_3D_60FOV', 'naive_3D_110FOV', 'naive_3D_160FOV', 'unidepth_3D', 'detany_3D', 'wilddet_3D'], default='naive_3D_60FOV')
    parser.add_argument('--prompt_method', type=str, choices=['baseline1','baseline2','p1', 'p2', 'p3', 'p4','p1_bbox','p1_visual','p1_visual_only'], default='p1')
    parser.add_argument('--api_base', type=str, default="http://localhost:8000/v1")
    parser.add_argument('--api_key', type=str, default="testkey")
    parser.add_argument('--temperature', type=float, default=0.6)
    parser.add_argument('--max_tokens', type=int, default=24000)
    parser.add_argument('--frame_path', type=str, default='VBIG_dataset/videos_frames', help='Path to frame for visualization')
    # parser.add_argument("--save_image", action="store_true", help="Save Image")
    return parser.parse_args()


def parse_args_inference_tasks():
    parser = argparse.ArgumentParser(description="Inference")
    parser.add_argument('filename', type=str, help='Path to JSON file with groups')
    # parser.add_argument('setting', type=str, choices=['single', 'all'])
    parser.add_argument('model', type=str)
    parser.add_argument(
        "--frame_ids",
        type=parse_numbers,
        required=False,
        help="Comma-separated numbers or ranges, e.g. 1,2,5-7,10 or 0:51 or 0:51:15"
    )
    parser.add_argument('--api_base', type=str, default="http://localhost:8000/v1")
    parser.add_argument('--api_key', type=str, default="testkey")
    parser.add_argument('--temperature', type=float, default=0.6)
    parser.add_argument('--max_tokens', type=int, default=24000)
    parser.add_argument('--frame_path', type=str, default='VBIG_dataset/videos_frames', help='Path to frame for visualization')
    return parser.parse_args()


def compute_group_bbox(groups, id_to_bbox, return_counts=False):
    gboxes = []
    counts = []
    for _, group in enumerate(groups):
        g_xmin, g_ymin, g_max, g_ymax = -1, -1, -1, -1
        for person in group:
            if str(person) not in id_to_bbox:
                continue
            xmin, ymin, xmax, ymax = id_to_bbox[str(person)]
            if g_xmin == -1 or xmin < g_xmin:
                g_xmin = xmin
            if g_ymin == -1 or ymin < g_ymin:
                g_ymin = ymin
            if g_max == -1 or xmax > g_max:
                g_max = xmax
            if g_ymax == -1 or ymax > g_ymax:
                g_ymax = ymax
        gboxes.append([g_xmin, g_ymin, g_max, g_ymax])
        counts.append(len(group))
    if return_counts:
        return gboxes, counts
    else:
        return gboxes

