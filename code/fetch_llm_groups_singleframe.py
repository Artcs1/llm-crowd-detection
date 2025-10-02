import cv2
import os
import dspy
import json
import argparse
import pandas as pd
from constants import CV2_COLORS
from tqdm.auto import tqdm
from prompts import IdentifyGroups, IdentifyGroups_Direction, IdentifyGroups_Transitive, IdentifyGroups_DirectionTransitive
from prompts import IdentifyGroupsImage, IdentifyGroups_DirectionImage, IdentifyGroups_TransitiveImage, IdentifyGroups_DirectionTransitiveImage

from utils import *



def parse_args():
    parser = argparse.ArgumentParser(description="Metadata JSON file")
    parser.add_argument('filename', type=str)
    parser.add_argument('mode', type=str)
    parser.add_argument('model', type=str)
    parser.add_argument('frame_id', type=int)
    parser.add_argument('--depth_method', type=str, choices=['naive_3D_60FOV', 'naive_3D_110FOV', 'naive_3D_160FOV', 'unidepth_3D', 'detany_3D'], default='naive_3D_60FOV')
    parser.add_argument('--prompt_method', type=str, choices=['p1', 'p2', 'p3', 'p4'], default='p1')
    parser.add_argument('--api_base', type=str, default="http://localhost:8000/v1")
    parser.add_argument('--api_key', type=str, default="testkey")
    parser.add_argument('--temperature', type=float, default=0.6)
    parser.add_argument('--max_tokens', type=int, default=32768)
    parser.add_argument('--frame_path', type=str, default='VBIG_dataset/videos_frames', help='Path to frame for visualization')
    parser.add_argument("--save", action="store_true", help="Activar modo debug")
    return parser.parse_args()

def main():


    args = parse_args()
    with open(args.filename, 'r') as f:
        data = json.load(f)

    lm = dspy.LM('openai/'+args.model, api_key=args.api_key, api_base=args.api_base, temperature=args.temperature, max_tokens=args.max_tokens)
    dspy.configure(lm=lm)
    os.makedirs(args.model, exist_ok=True)
    
    dspy_cot, use_direction = get_dspy_cot(args.mode, args.prompt_method)
   
    frame_found = False
    for frame in data['frames']:
        if frame['frame_id'] == args.frame_id:
            frame_found = True
            break
    
    if not frame_found:
        print(f"Frame ID {args.frame_id} not found in the data.")
        return


    frame_input_data = []
    personid2bbox = {}
    for i,det in enumerate(frame['detections']):
        personid2bbox[det['track_id']] = det['bbox']
        if use_direction:
            frame_input_data.append({'person_id': det['track_id'], 'x': det[args.depth_method][0], 'y': det[args.depth_method][1], 'z': det[args.depth_method][2], 'direction':det['direction']})
        else:
            frame_input_data.append({'person_id': det['track_id'], 'x': det[args.depth_method][0], 'y': det[args.depth_method][1], 'z': det[args.depth_method][2]})
    
    save_filename = args.filename.split('/')[-1][:-5]
    frame_path = f'{args.frame_path}/{save_filename}/{str(args.frame_id).zfill(5)}.jpeg'

    if args.mode == 'llm':
        output = inference_wrapper(lm, dspy_cot, frame_input_data, args.mode)
    elif args.mode =='vlm':
        output = inference_wrapper(lm, dspy_cot, frame_input_data, args.mode, frame_path)

    if output['error'] is not None:
        print(f"Error during inference: {output['error']}")
        return

    output['frame_id'] = args.frame_id
    output['id_tobbox'] = personid2bbox

    if args.save == True:
        img = cv2.imread(frame_path)

        for i, group in enumerate(output['groups']):
            for person_id in group:
                xl, yl, x2, y2 = personid2bbox[person_id]
                cv2.rectangle(img, (xl,yl), (x2,y2), CV2_COLORS[i], 2)

        # configure this path                
        res_path = 'results'
        res_path = os.path.join(res_path, args.model.split('/')[1]+'/'+args.depth_method+'/'+args.prompt_method, save_filename,)
        print(res_path)
        os.makedirs(res_path, exist_ok=True)
        save_path = os.path.join(res_path, 'result.png')
        cv2.imwrite(save_path, img)
        
        #save output json
        with open(f'{args.model}/{save_filename}.json', "w") as f:
            json.dump(output, f, indent=4)
        #with open(save_path.replace('.png', '.txt'), 'w') as f:
        #    f.write(str(output))
    
    else:
        
        with open(f'{args.model}/{save_filename}.json', "w") as f:
            json.dump(output, f, indent=4)



        # get last folder path from args.frame_path
        
    

if __name__ == "__main__":
    main()
