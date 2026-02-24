import glob
import json
import os
import argparse
import pickle

from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Select mode, prompt method, model, and VLM mode")
    parser.add_argument("--dataset", type=str, choices=["JRDB_fixed_gold","JRDB_fixed","BLENDER","SEKAI_OURS","SEKAI_OURS_200"], required=True, help="Dataset options")
    parser.add_argument("--mode", type=str, choices=["single","full"], required=True, help="Mode: single or full")
    parser.add_argument("--depth_method", type=str, choices=["naive_3D_60FOV","detany_3D","unidepth_3D"], default="naive_3D_60FOV", help="Depth method")
    parser.add_argument("--prompt_method", type=str, choices=["baseline1","baseline2","p1","p2","p3","p4","p5"], required=True, help="Prompt method")
    parser.add_argument("--model", type=str, required=True, help="Specify the model name or path")
    parser.add_argument("--vlm_mode", type=str, choices=["llm","vlm_image","vlm_text"], required=True, help="VLM mode: image or text")
    parser.add_argument('--frame_id', type=int)
    args = parser.parse_args()
    

    base_dir = os.getcwd()
    #print(base_dir)
    #base_dir = "/home/artcs1/Desktop/llm-crowd-detection/code"
    
    if args.dataset == 'JRDB_fixed':
        H, W = 480, 3760
    elif args.dataset == 'JRDB_fixed_gold':
        H, W = 480, 3760
    elif args.dataset == 'BLENDER':
        H, W = 3240, 3240
    elif args.dataset == 'SEKAI_OURS':
        H, W = 1080, 1920
    elif args.dataset == 'SEKAI_OURS_200':
        H, W = 1080, 1920
    
    results_folder = f"groupings/{args.dataset}/results" if args.mode == "single" else f"predictions/{args.dataset}/results_{args.mode}"
    
    path = os.path.join(
        base_dir,
        results_folder,            # results or results_full
        args.model,                # e.g., Qwen2.5-VL-3B-Instruct
        args.vlm_mode,             # vlm_image or vlm_text
        args.depth_method,         # fixed subfolder
        args.prompt_method,        # baseline1, p1, etc.
        str(args.frame_id),        # frame_id
        "*"                        # wildcard for files
    )

    if not os.path.exists(group_path):
        os.makedirs(group_path)
    
    files = glob.glob(path)
    files.sort()
    
    print(path)
    print(args.model)
    scenarios = set()

    det_file = f"detection_files/{args.dataset}/{results_folder.split('/')[-1]}_{args.model}_{args.vlm_mode}_{args.depth_method}_{args.prompt_method}.pickle"
    det_directory = "/".join(det_file.split('/')[:-1])
    
    if not os.path.exists(det_directory):
        os.makedirs(det_directory)
    
    if os.path.exists(det_file):
        os.remove(det_file)
    
    
    for ind, file in enumerate(tqdm(files)):
        last = file.split('/')
        json_file = f'{file}/{last[-1]}.json'
    
        with open(json_file, 'r') as f:
            data = json.load(f)
    
        scenario       = last[-1]

        if args.dataset == 'SEKAI_OURS':
            scenarios.add(scenario)
            number   = 0 # int(split_scenario[-1])
        else:
            split_scenario = scenario.split('_')
            orig_scenario  = "".join(split_scenario[:-1]) 
            number   = int(split_scenario[-1])
            scenarios.add(orig_scenario)
    

        print(scenario)
        print(split_scenario)
        print(orig_scenario)
        print(number)


        dc, idx, int_img, lvl, group_id = 1, len(scenarios)-1, (number+1)*15, 1, 1

        print(data.items())
