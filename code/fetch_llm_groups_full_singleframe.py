import cv2
import os
import dspy
import json
import argparse
import pandas as pd
from constants import CV2_COLORS
from tqdm.auto import tqdm
from prompts import IdentifyGroups_AllFrames, vlm_IdentifyGroups_AllFramesText, vlm_IdentifyGroups_AllFramesImage, vlm_GroupsQAonlyFullImage, full_baseline2


def parse_args():
    parser = argparse.ArgumentParser(description="Metadata JSON file")
    parser.add_argument('filename', type=str)
    parser.add_argument('mode', type=str)
    parser.add_argument('model', type=str)
    parser.add_argument('frame_id', type=int)
    parser.add_argument('--depth_method', type=str, choices=['naive_3D_60FOV', 'naive_3D_110FOV', 'naive_3D_160FOV', 'unidepth_3D', 'detany_3D'], default='naive_3D_60FOV')
    parser.add_argument('--prompt_method', type=str, choices=['baseline1', 'baseline2', 'p1', 'p2', 'p3', 'p4'], default='p1')
    parser.add_argument('--api_base', type=str, default="http://localhost:8000/v1")
    parser.add_argument('--api_key', type=str, default="testkey")
    parser.add_argument('--temperature', type=float, default=0.6)
    parser.add_argument('--max_tokens', type=int, default=32768)
    parser.add_argument('--frame_path', type=str, default='VBIG_dataset/videos_frames', help='Path to frame for visualization')
    parser.add_argument("--save_image", action="store_true", help="Activar modo debug")
    return parser.parse_args()


def inference(dspy_module, input_text, target_frame, mode='llm', frame_path=''):
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


def inference_wrapper(lm, dspy_module, input_text, target_frame, mode='llm', frame_path= ''):
    
    if mode == 'llm' or mode == 'vlm_text': 
        output, is_error = inference(dspy_module, input_text, target_frame, mode, '')
    elif mode == 'vlm_image': 
        output, is_error = inference(dspy_module, input_text, target_frame, mode, frame_path)

    print(is_error)

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


def main():
    args = parse_args()
    #print(args)
    with open(args.filename, 'r') as f:
        data = json.load(f)

    lm = dspy.LM('openai/'+args.model, api_key=args.api_key, api_base=args.api_base, temperature=args.temperature, max_tokens=args.max_tokens)
    dspy.configure(lm=lm)

    use_direction = False
    if args.mode == 'llm':
        if args.prompt_method == 'p1':
            dspy_cot = dspy.ChainOfThought(IdentifyGroups_AllFrames)
    elif args.mode == 'vlm_text':
        if args.prompt_method == 'p1':
            dspy_cot = dspy.ChainOfThought(vlm_IdentifyGroups_AllFramesText)
    elif args.mode == 'vlm_image':
        if args.prompt_method == 'baseline1':
            dspy_cot = dspy.ChainOfThought(vlm_GroupsQAonlyFullImage)
        if args.prompt_method == 'baseline2':
            dspy_cot = full_baseline2()
        elif args.prompt_method == 'p1':
            dspy_cot = dspy.ChainOfThought(vlm_IdentifyGroups_AllFramesImage)



    all_frames, bboxes = [], []
    for frame in data['frames']:
        frame_input_data = []
        personid2bbox = {}
        for i,det in enumerate(frame['detections']):
            personid2bbox[det['track_id']] = det['bbox']
            if use_direction:
                frame_input_data.append({'person_id': det['track_id'], 'x': det[args.depth_method][0], 'y': det[args.depth_method][1], 'z': det[args.depth_method][2], 'direction':det['direction']})
            else:
                frame_input_data.append({'person_id': det['track_id'], 'x': det[args.depth_method][0], 'y': det[args.depth_method][1], 'z': det[args.depth_method][2]})
        all_frames.append(frame_input_data)
        bboxes.append(personid2bbox)

    save_filename = args.filename.split('/')[-1][:-5]
    frame_path = f'{args.frame_path}/{save_filename}/{str(args.frame_id).zfill(5)}.jpeg'

    if args.mode == 'llm' or args.mode == 'vlm_text':
        output = inference_wrapper(lm, dspy_cot, all_frames, args.frame_id, args.mode)
    elif args.mode == 'vlm_image':
        output = inference_wrapper(lm, dspy_cot, all_frames, args.frame_id, args.mode, frame_path)

    if output['error'] is not None:
        print(f"Error during inference: {output['error']}")
        return

    output['frame_id'] = args.frame_id
    output['id_tobbox'] = bboxes[args.frame_id]
    print(output)

    if args.save_image == True:
        img = cv2.imread(frame_path)

        if args.prompt_method == 'baseline1' or args.prompt_method == 'baseline2' :
            for ind, p in enumerate(output['groups']):
                for bbox in p:
                    img = cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), CV2_COLORS[ind], 2)
        else:
            for i, group in enumerate(output['groups']):
                for person_id in group:
                    xl, yl, x2, y2 = bboxes[args.frame_id][person_id]
                    cv2.rectangle(img, (xl,yl), (x2,y2), CV2_COLORS[i], 2)

        # configure this path                
        res_path = 'results_full'
        res_path = os.path.join(res_path, args.model.split('/')[1]+'/'+ args.mode + '/' + args.depth_method+'/'+args.prompt_method, save_filename,)
        os.makedirs(res_path, exist_ok=True)
        save_path = os.path.join(res_path, 'result.png')
        cv2.imwrite(save_path, img)
    
        #save output json
        with open(f'{res_path}/{save_filename}.json', "w") as f:
            json.dump(output, f, indent=4)
    
    else:

        res_path = 'results_full'
        res_path = os.path.join(res_path, args.model.split('/')[1]+'/'+ args.mode + '/'+ args.depth_method+'/'+args.prompt_method, save_filename,)
        os.makedirs(res_path, exist_ok=True)
        with open(f'{res_path}/{save_filename}.json', "w") as f:
            json.dump(output, f, indent=4)


if __name__ == "__main__":
    main()
