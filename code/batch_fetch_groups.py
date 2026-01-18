import cv2
import os
import dspy
import json
import argparse
import pandas as pd
from tqdm.auto import tqdm
#from prompts import IdentifyGroups, IdentifyGroups_Direction, IdentifyGroups_Transitive, IdentifyGroups_DirectionTransitive
#from prompts import IdentifyGroupsImage, IdentifyGroups_DirectionImage, IdentifyGroups_TransitiveImage, IdentifyGroups_DirectionTransitiveImage

from utils import *

def main():


    args = parse_args()

    collected_files = [os.path.join(args.filename,f) for f in os.listdir(args.filename) if os.path.isfile(os.path.join(args.filename, f))]
    collected_files.sort()
    #collected_files = collected_files[::-1]

    for current_file in tqdm(collected_files):
        #try:
        print(current_file)
        if True:
            with open(current_file, 'r') as f:
                data = json.load(f)
        
            lm = dspy.LM('openai/'+args.model, api_key=args.api_key, api_base=args.api_base, temperature=args.temperature, max_tokens=args.max_tokens)
            dspy.configure(lm=lm)
            
            if args.setting == 'single':
                dspy_cot, use_direction = get_dspy_cot(args.mode, args.prompt_method)
                frame_input_data, personid2bbox = get_frame_bboxes(data, use_direction, args.depth_method, args.frame_id, args.prompt_method)
            elif args.setting == 'full':
                dspy_cot, use_direction = get_full_dspy_cot(args.mode, args.prompt_method)
                all_frames, bboxes = get_allframes_bboxes2(data, use_direction, args.depth_method, args.prompt_method)
        
            save_filename = current_file.split('/')[-1][:-5]
            frame_path = f'{args.frame_path}/{save_filename}/{str(args.frame_id).zfill(5)}.jpeg'
        
            if args.setting == 'single':
                if args.mode == 'llm' or args.mode =='vlm_text':
                    output = inference_wrapper(lm, dspy_cot, frame_input_data, args.mode)
                elif args.mode == 'vlm_image':
                    output = inference_wrapper(lm, dspy_cot, frame_input_data, args.mode, frame_path)
            elif args.setting == 'full':
                if args.mode == 'llm' or args.mode == 'vlm_text':
                    output = full_inference_wrapper(lm, dspy_cot, all_frames, args.frame_id, args.mode)
                elif args.mode == 'vlm_image':
                    output = full_inference_wrapper(lm, dspy_cot, all_frames, args.frame_id, args.mode, frame_path)
        
            if output['error'] is not None:
                print(f"Error during inference: {output['error']}")
                return
            
        
            output['frame_id'] = args.frame_id
        
            if args.setting == 'single':
                output['id_tobbox'] = personid2bbox
                res_path = 'predictions/'+ args.frame_path.split('/')[-2] + '/results'
                save_frame(output, personid2bbox, res_path, save_filename, frame_path, args.save_image, args.model, args.mode, args.depth_method, args.prompt_method, args.frame_id)
            elif args.setting == 'full':
                output['id_tobbox'] = bboxes[args.frame_id-1]
                res_path = 'predictions/'+ args.frame_path.split('/')[-2] + '/results_full'
                save_full_frame(output, bboxes, res_path, save_filename, frame_path, args.save_image, args.model, args.mode, args.depth_method, args.prompt_method, args.frame_id)
        
        #except Exception as e:
        #    print(f'Fail in: {current_file}, seting: {args.setting},  mode: {args.mode}, model: {args.model}, prompt: {args.prompt_method}')
    
if __name__ == "__main__":
    main()
