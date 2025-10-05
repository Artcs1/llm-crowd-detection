import cv2
import os
import dspy
import json
import argparse
import pandas as pd
from constants import CV2_COLORS
from tqdm.auto import tqdm
from prompts import IdentifyGroups_AllFrames, vlm_IdentifyGroups_AllFramesText, vlm_IdentifyGroups_AllFramesImage, vlm_GroupsQAonlyFullImage, full_baseline2

from utils import *


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

def main():
    args = parse_args()
    #print(args)
    with open(args.filename, 'r') as f:
        data = json.load(f)

    lm = dspy.LM('openai/'+args.model, api_key=args.api_key, api_base=args.api_base, temperature=args.temperature, max_tokens=args.max_tokens)
    dspy.configure(lm=lm)

    dspy_cot, use_direction = get_full_dspy_cot(args.mode, args.prompt_method)
    all_frames, bboxes = get_allframes_bboxes(data, use_direction, args.depth_method)

    save_filename = args.filename.split('/')[-1][:-5]
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


if __name__ == "__main__":
    main()
