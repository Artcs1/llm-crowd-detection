import cv2 
import os
import dspy
import json
import argparse
import pandas as pd
from tqdm.auto import tqdm

from prompts import IdentifyGroups, IdentifyGroups_Direction, IdentifyGroups_Transitive, IdentifyGroups_DirectionTransitive
from prompts import vlm_GroupsQAonlyImage, vlm_IdentifyGroupsImage, vlm_IdentifyGroups_DirectionImage, vlm_IdentifyGroups_TransitiveImage, vlm_IdentifyGroups_DirectionTransitiveImage, baseline2
from prompts import vlm_IdentifyGroupsText, vlm_IdentifyGroups_DirectionText, vlm_IdentifyGroups_TransitiveText, vlm_IdentifyGroups_DirectionTransitiveText

from prompts import IdentifyGroups_AllFrames, vlm_IdentifyGroups_AllFramesText, vlm_IdentifyGroups_AllFramesImage, vlm_GroupsQAonlyFullImage, full_baseline2

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

def full_inference(dspy_module, input_text, target_frame, mode='llm', frame_path=''):
    is_error = False
    try:
        if mode == 'llm' or mode =='vlm_text':
            predictions = dspy_module(all_frames=input_text, target_frame=target_frame)
        if mode == 'vlm_image':
                
            image = dspy.Image.from_file(frame_path)
            video = []
            for i in range(0,50,5):
                img_folder = frame_path[:-10]
                image_file = f'{img_folder}{str(i+1).zfill(5)}.jpeg'
                video.append(dspy.Image.from_file(image_file))
            predictions = dspy_module(image=image, video=video, all_frames=input_text, target_frame = target_frame)

        output = predictions.groups
    except Exception as e:
        output = str(e)
        is_error = True
    return output, is_error

def full_inference_wrapper(lm, dspy_module, input_text, target_frame, mode='llm', frame_path= ''):
        
    if mode == 'llm' or mode == 'vlm_text': 
        output, is_error = full_inference(dspy_module, input_text, target_frame, mode, '') 
    elif mode == 'vlm_image': 
        output, is_error = full_inference(dspy_module, input_text, target_frame, mode, frame_path)

    res = {}
    if not is_error:
        res['hist'] = str(lm.history[-1])
        res['groups'] = output
        res['error'] = None
    else:
        res['hist'] = str(lm.history[-1])
        res['groups'] = None
        res['error'] = output
    return res 

def inference(dspy_module, input_text, mode='llm', image_path='optional'):
    is_error = False
    try:
        if mode == 'llm' or mode == 'vlm_text':
            predictions = dspy_module(detections=input_text)
        elif mode == 'vlm_image':
            predictions = dspy_module(image=dspy.Image.from_file(image_path), detections=input_text)
        output = predictions.groups
    except Exception as e:
        output = str(e)
        is_error = True
    return output, is_error


def inference_wrapper(lm, dspy_module, input_text, mode='llm', image_path='optional'):
    if mode == 'llm' or mode == 'vlm_text':
        output, is_error = inference(dspy_module, input_text, mode)
    elif mode == 'vlm_image':
        output, is_error = inference(dspy_module, input_text, mode, image_path)
    res = {}
    if not is_error:
        res['hist'] = str(lm.history[-1])
        res['groups'] = output
        res['error'] = None
    else:
        res['hist'] = str(lm.history[-1])
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
    elif mode == 'vlm_image':
        if prompt_method == 'baseline1':
            dspy_cot = dspy.ChainOfThought(vlm_GroupsQAonlyImage)
        elif prompt_method == 'baseline2':
            dspy_cot = baseline2()
        elif prompt_method == 'p1':
            dspy_cot = dspy.ChainOfThought(vlm_IdentifyGroupsImage)
        elif prompt_method == 'p2':
            dspy_cot = dspy.ChainOfThought(vlm_IdentifyGroups_DirectionImage)
            use_direction = True
        elif prompt_method == 'p3':
            dspy_cot = dspy.ChainOfThought(vlm_IdentifyGroups_TransitiveImage)
        elif prompt_method == 'p4':
            dspy_cot = dspy.ChainOfThought(vlm_IdentifyGroups_DirectionTransitiveImage)
            use_direction = True
    elif mode == 'vlm_text':
        if prompt_method == 'p1':
            dspy_cot = dspy.ChainOfThought(vlm_IdentifyGroupsText)
        elif prompt_method == 'p2':
            dspy_cot = dspy.ChainOfThought(vlm_IdentifyGroups_DirectionText)
            use_direction = True
        elif prompt_method == 'p3':
            dspy_cot = dspy.ChainOfThought(vlm_IdentifyGroups_TransitiveText)
        elif prompt_method == 'p4':
            dspy_cot = dspy.ChainOfThought(vlm_IdentifyGroups_DirectionTransitiveText)
            use_direction = True

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
 

    return (dspy_cot, use_direction)

def get_allframes_bboxes(data, use_direction, depth_method):
        
    all_frames, bboxes = [], []
    for frame in data['frames']:
        frame_input_data = []
        personid2bbox = {}
        for i,det in enumerate(frame['detections']):
            personid2bbox[det['track_id']] = det['bbox']
            if use_direction:
                frame_input_data.append({'person_id': det['track_id'], 'x': det[depth_method][0], 'y': det[depth_method][1], 'z': det[depth_method][2], 'direction':det['direction']})
            else:
                frame_input_data.append({'person_id': det['track_id'], 'x': det[depth_method][0], 'y': det[depth_method][1], 'z': det[depth_method][2]})
        all_frames.append(frame_input_data)
        bboxes.append(personid2bbox)

    return (all_frames, bboxes)


def get_frame_bboxes(data, use_direction, depth_method, frame_id):

    frame_found = False
    for frame in data['frames']:
        if frame['frame_id'] == frame_id:
            frame_found = True
            break
    
    if not frame_found:
        print(f"Frame ID {frame_id} not found in the data.")
        return ([],{})

    frame_input_data = []
    personid2bbox = {}
    for i,det in enumerate(frame['detections']):
        personid2bbox[det['track_id']] = det['bbox']
        if use_direction:
            frame_input_data.append({'person_id': det['track_id'], 'x': det[depth_method][0], 'y': det[depth_method][1], 'z': det[depth_method][2], 'direction':det['direction']})
        else:
            frame_input_data.append({'person_id': det['track_id'], 'x': det[depth_method][0], 'y': det[depth_method][1], 'z': det[depth_method][2]})
    
    return (frame_input_data, personid2bbox)


def save_frame(output, personid2bbox, res_path, save_filename, frame_path, save_image, model, mode, depth_method, prompt_method):
    if save_image == True:
        img = cv2.imread(frame_path)

        if prompt_method == 'baseline1' or prompt_method == 'baseline2' :
            for ind, p in enumerate(output['groups']):
                for bbox in p:
                    img = cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), CV2_COLORS[ind], 2)
        else:
            for i, group in enumerate(output['groups']):
                for person_id in group:
                    xl, yl, x2, y2 = personid2bbox[person_id]
                    cv2.rectangle(img, (xl,yl), (x2,y2), CV2_COLORS[i], 2)

        res_path = os.path.join(res_path, model.split('/')[1]+'/'+ mode + '/' + depth_method+'/'+prompt_method, save_filename,)
        os.makedirs(res_path, exist_ok=True)
        save_path = os.path.join(res_path, 'result.png')
        cv2.imwrite(save_path, img)
            
        with open(f'{res_path}/{save_filename}.json', "w") as f:
            json.dump(output, f, indent=4)
        
    else:

        res_path = os.path.join(res_path, model.split('/')[1]+'/'+ mode + '/'+ depth_method+'/'+prompt_method, save_filename,)
        os.makedirs(res_path, exist_ok=True)
        with open(f'{res_path}/{save_filename}.json', "w") as f:
            json.dump(output, f, indent=4)

def save_full_frame(output, bboxes, res_path, save_filename, frame_path, save_image, model, mode, depth_method, prompt_method, frame_id):

    if save_image == True:
        img = cv2.imread(frame_path)

        if prompt_method == 'baseline1' or prompt_method == 'baseline2' :
            for ind, p in enumerate(output['groups']):
                for bbox in p:
                    img = cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), CV2_COLORS[ind], 2)
        else:
            for i, group in enumerate(output['groups']):
                for person_id in group:
                    xl, yl, x2, y2 = bboxes[frame_id][person_id]
                    cv2.rectangle(img, (xl,yl), (x2,y2), CV2_COLORS[i], 2)

        # configure this path                
        res_path = os.path.join(res_path, model.split('/')[1]+'/'+ mode + '/' + depth_method+'/'+prompt_method, save_filename,)
        os.makedirs(res_path, exist_ok=True)
        save_path = os.path.join(res_path, 'result.png')
        cv2.imwrite(save_path, img)
    
        #save output json
        with open(f'{res_path}/{save_filename}.json', "w") as f:
            json.dump(output, f, indent=4)
    
    else:

        res_path = os.path.join(res_path, model.split('/')[1]+'/'+ mode + '/'+ depth_method+'/'+prompt_method, save_filename,)
        os.makedirs(res_path, exist_ok=True)
        with open(f'{res_path}/{save_filename}.json', "w") as f:
            json.dump(output, f, indent=4)

def parse_args():
    parser = argparse.ArgumentParser(description="Metadata JSON file")
    parser.add_argument('filename', type=str)
    parser.add_argument('setting', type=str, choices=['single', 'full'])
    parser.add_argument('mode', type=str, choices=['llm', 'vlm_text', 'vlm_image'])
    parser.add_argument('model', type=str)
    parser.add_argument('frame_id', type=int)
    parser.add_argument('--depth_method', type=str, choices=['naive_3D_60FOV', 'naive_3D_110FOV', 'naive_3D_160FOV', 'unidepth_3D', 'detany_3D'], default='naive_3D_60FOV')
    parser.add_argument('--prompt_method', type=str, choices=['baseline1','baseline2','p1', 'p2', 'p3', 'p4'], default='p1')
    parser.add_argument('--api_base', type=str, default="http://localhost:8000/v1")
    parser.add_argument('--api_key', type=str, default="testkey")
    parser.add_argument('--temperature', type=float, default=0.6)
    parser.add_argument('--max_tokens', type=int, default=32768)
    parser.add_argument('--frame_path', type=str, default='VBIG_dataset/videos_frames', help='Path to frame for visualization')
    parser.add_argument("--save_image", action="store_true", help="Save Image")
    return parser.parse_args()


