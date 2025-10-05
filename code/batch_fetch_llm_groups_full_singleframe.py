import cv2
import os
import dspy
import json
import argparse
import pandas as pd
from tqdm.auto import tqdm
from prompts import IdentifyGroups_AllFrames, vlm_IdentifyGroups_AllFramesText, vlm_IdentifyGroups_AllFramesImage, vlm_GroupsQAonlyFullImage, full_baseline2

from utils import *


def main():
    args = parse_args()
    collected_files = [os.path.join(args.filename,f) for f in os.listdir(args.filename) if os.path.isfile(os.path.join(args.filename, f))]

    for current_file in tqdm(collected_files):

        try: 

            lm = dspy.LM('openai/'+args.model, api_key=args.api_key, api_base=args.api_base, temperature=args.temperature, max_tokens=args.max_tokens)
            dspy.configure(lm=lm)

            dspy_cot, use_direction = get_full_dspy_cot(args.mode, args.prompt_method)
            with open(current_file, 'r') as f:
                data = json.load(f)
            all_frames, bboxes = get_allframes_bboxes(data, use_direction, args.depth_method)
        
            save_filename = current_file.split('/')[-1][:-5]
            frame_path = f'{args.frame_path}/{save_filename}/{str(args.frame_id).zfill(5)}.jpeg'
        
            if args.mode == 'llm' or args.mode == 'vlm_text':
                output = full_inference_wrapper(lm, dspy_cot, all_frames, args.frame_id, args.mode)
            elif args.mode == 'vlm_image':
                output = full_inference_wrapper(lm, dspy_cot, all_frames, args.frame_id, args.mode, frame_path)
        
            if output['error'] is not None:
                print(f"Error during inference: {output['error']}")
                return
        
            output['frame_id'] = args.frame_id
            output['id_tobbox'] = bboxes[args.frame_id]
            res_path = 'results_full'
            save_full_frame(output, bboxes, res_path, save_filename, frame_path, args.save_image, args.model, args.mode, args.depth_method, args.prompt_method, args.frame_id)

        except Exception as e:
            print('Fail in:', current_file)
    
        
    
if __name__ == "__main__":
    main()
