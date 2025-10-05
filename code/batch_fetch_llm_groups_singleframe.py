import cv2
import os
import dspy
import json
import argparse
import pandas as pd
from tqdm.auto import tqdm
from prompts import IdentifyGroups, IdentifyGroups_Direction, IdentifyGroups_Transitive, IdentifyGroups_DirectionTransitive

from tqdm import tqdm
from utils import *

def main():

    args = parse_args()

    collected_files = [os.path.join(args.filename,f) for f in os.listdir(args.filename) if os.path.isfile(os.path.join(args.filename, f))]

    lm = dspy.LM('openai/'+args.model, api_key=args.api_key, api_base=args.api_base, temperature=args.temperature, max_tokens=args.max_tokens)
    dspy.configure(lm=lm)

    dspy_cot, use_direction = get_dspy_cot(args.mode, args.prompt_method)

    for current_file in tqdm(collected_files):

        try:

            #print(current_file)
            with open(current_file, 'r') as f:
                data = json.load(f)

            frame_input_data, personid2bbox = get_frame_bboxes(data, use_direction, args.depth_method, args.frame_id)

            save_filename = current_file.split('/')[-1][:-5]
            frame_path = f'{args.frame_path}/{save_filename}/{str(args.frame_id).zfill(5)}.jpeg'
            
            if args.mode == 'llm' or args.mode == 'vlm_text':
                output = inference_wrapper(lm, dspy_cot, frame_input_data, args.mode)
            elif args.mode =='vlm_image':
                output = inference_wrapper(lm, dspy_cot, frame_input_data, args.mode, frame_path)
    
            if output['error'] is not None:
                print(f"Error during inference: {output['error']}")
                continue
    
            output['frame_id'] = args.frame_id
            output['id_tobbox'] = personid2bbox
    
            res_path = 'results'
            save_frame(output, personid2bbox, res_path, save_filename, frame_path, args.save_image, args.model, args.mode, args.depth_method, args.prompt_method)
   
        except Exception as e:
            print('Fail in:', current_file)

if __name__ == "__main__":
    main()
