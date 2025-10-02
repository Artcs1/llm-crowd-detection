import cv2
import os
import dspy
import json
import argparse
import pandas as pd
from constants import CV2_COLORS
from tqdm.auto import tqdm
from prompts import IdentifyGroups_AllFrames



def parse_args():
    parser = argparse.ArgumentParser(description="Metadata JSON file")
    parser.add_argument('filename', type=str)
    parser.add_argument('model', type=str)
    parser.add_argument('frame_id', type=int)
    parser.add_argument('--depth_method', type=str, choices=['naive_3D_60FOV', 'naive_3D_110FOV', 'naive_3D_160FOV', 'unidepth_3D', 'detany_3D'], default='naive_3D_60FOV')
    parser.add_argument('--prompt_method', type=str, choices=['p1', 'p2', 'p3', 'p4'], default='p1')
    parser.add_argument('--api_base', type=str, default="http://localhost:8000/v1")
    parser.add_argument('--api_key', type=str, default="testkey")
    parser.add_argument('--temperature', type=float, default=0.6)
    parser.add_argument('--max_tokens', type=int, default=32768)
    parser.add_argument('--frame_path', type=str, default=None, help='Path to frame for visualization')
    return parser.parse_args("/lustre/nvwulf/scratch/pchitale/workspace/llm-crowd-detection/data/json/json_0928/Cusco_Peru_0003002_clip_001.json Qwen/Qwen3-30B-A3B-Instruct-2507 15 --depth_method detany_3D --prompt_method p1 --frame_path /lustre/nvwulf/scratch/pchitale/workspace/llm-crowd-detection/data/videos/videos_0928/Cusco_Peru_0003002_clip_001/00015.jpeg".split())


def inference(dspy_module, input_text, target_frame):
    is_error = False
    try:
        predictions = dspy_module(all_frames=input_text, target_frame=target_frame)
        output = predictions.groups
    except Exception as e:
        output = str(e)
        is_error = True
    return output, is_error


def inference_wrapper(lm, dspy_module, input_text, target_frame):
    output, is_error = inference(dspy_module, input_text, target_frame)
    res = {}
    if not is_error:
        res['groups'] = output
        res['hist'] = lm.history[-1]
        res['error'] = None
    else:
        res['groups'] = None
        res['hist'] = lm.history[-1]
        res['error'] = output
    return res


def main():
    args = parse_args()
    print(args)
    with open(args.filename, 'r') as f:
        data = json.load(f)

    lm = dspy.LM('openai/'+args.model, api_key=args.api_key, api_base=args.api_base, temperature=args.temperature, max_tokens=args.max_tokens)
    dspy.configure(lm=lm)

    use_direction = False
    if args.prompt_method == 'p1':
        dspy_cot = dspy.ChainOfThought(IdentifyGroups_AllFrames)
    # elif args.prompt_method == 'p2':
    #     dspy_cot = dspy.ChainOfThought(IdentifyGroups_Direction)
    #     use_direction = True
    # elif args.prompt_method == 'p3':
    #     dspy_cot = dspy.ChainOfThought(IdentifyGroups_Transitive)
    # elif args.prompt_method == 'p4':
    #     dspy_cot = dspy.ChainOfThought(IdentifyGroups_DirectionTransitive)
    #     use_direction = True


    # frame_found = False
    # for frame in data['frames']:
    #     if frame['frame_id'] == args.frame_id:
    #         frame_found = True
    #         break
    
    # if not frame_found:
    #     print(f"Frame ID {args.frame_id} not found in the data.")
    #     return


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
    
    output = inference_wrapper(lm, dspy_cot, all_frames, args.frame_id)
    if output['error'] is not None:
        print(f"Error during inference: {output['error']}")
        return

    output['frame_id'] = args.frame_id

    if args.frame_path is not None:
        img = cv2.imread(args.frame_path)

        for i, group in enumerate(output['groups']):
            for person_id in group:
                xl, yl, x2, y2 = bboxes[args.frame_id][person_id]
                cv2.rectangle(img, (xl,yl), (x2,y2), CV2_COLORS[i], 2)

        # configure this path                
        res_path = '/lustre/nvwulf/scratch/pchitale/workspace/llm-crowd-detection/results/SIE_AllFrames'
        res_path = os.path.join(res_path, args.model.split('/')[1]+'/'+args.depth_method+'/'+args.prompt_method, args.frame_path.split('/')[-2],)
        os.makedirs(res_path, exist_ok=True)
        save_path = os.path.join(res_path, os.path.basename(args.frame_path).split('.')[0] + '.png')
        print(save_path)
        cv2.imwrite(save_path, img)
        
        #save output json
        with open(save_path.replace('.png', '.txt'), 'w') as f:
            f.write(str(output))
    
    else:
        print(json.dumps(output, indent=4))

        # get last folder path from args.frame_path
        
    

if __name__ == "__main__":
    main()
