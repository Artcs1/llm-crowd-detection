import glob
import json
import os
import argparse
import pickle

from tqdm import tqdm

def add_prediction(det_file, idx, int_img, x1, y1, x2, y2, group_id, dc, lvl):
    
    PRED_list = [idx, int_img, x1, y1, x2, y2, group_id, dc, lvl]
    str_to_be_added = [str(k) for k in PRED_list]
    str_to_be_added = (" ".join(str_to_be_added))
    f = open(det_file, "a+")
    f.write(str_to_be_added + "\r\n")
    f.close()
 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Select mode, prompt method, model, and VLM mode")
    parser.add_argument("--dataset", type=str, choices=["JRDB_gold","JRDB","BLENDER","SEKAI_OURS","SEKAI_OURS_200"], required=True, help="Dataset options")
    parser.add_argument("--mode", type=str, choices=["single","full"], required=True, help="Mode: single or full")
    parser.add_argument("--depth_method", type=str, choices=["naive_3D_60FOV","detany_3D","unidepth_3D"], default="naive_3D_60FOV", help="Depth method")
    parser.add_argument("--prompt_method", type=str, choices=["baseline1","baseline2","p1","p2","p3","p4","p5"], required=True, help="Prompt method")
    parser.add_argument("--model", type=str, required=True, help="Specify the model name or path")
    parser.add_argument("--vlm_mode", type=str, choices=["llm","vlm_image","vlm_text"], required=True, help="VLM mode: image or text")
    parser.add_argument('--frame_id', type=int)
    parser.add_argument("--output_mode", type=str, choices=["finegrained","coarse"], default="finegrained", help="Formatting the output")
    args = parser.parse_args()
    

    base_dir = os.getcwd()
    #print(base_dir)
    #base_dir = "/home/artcs1/Desktop/llm-crowd-detection/code"
    
    if args.dataset == 'JRDB':
        H, W = 480, 3760
    if args.dataset == 'JRDB_gold':
        H, W = 480, 3760
    elif args.dataset == 'BLENDER':
        H, W = 3240, 3240
    elif args.dataset == 'SEKAI_OURS':
        H, W = 1080, 1920
    elif args.dataset == 'SEKAI_OURS_200':
        H, W = 1080, 1920
    
    results_folder = f"predictions/{args.dataset}/results" if args.mode == "single" else f"predictions/{args.dataset}/results_{args.mode}"
    groups_folder = f"groups/{args.dataset}/results" if args.mode == "single" else f"groups/{args.dataset}/results_{args.mode}"
    
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

    group_path = os.path.join(
        base_dir,
        groups_folder,            # results or results_full
        args.model,                # e.g., Qwen2.5-VL-3B-Instruct
        args.vlm_mode,             # vlm_image or vlm_text
        args.depth_method,         # fixed subfolder
        args.prompt_method         # baseline1, p1, etc.
    )

    if not os.path.exists(group_path):
        os.makedirs(group_path)
    
    files = glob.glob(path)
    files.sort()
    
    print(path)

    scenarios = set()
    det_file = f"detection_files/{args.dataset}/{results_folder.split('/')[-1]}_{args.model}_{args.vlm_mode}_{args.depth_method}_{args.prompt_method}.txt"
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
    
        dc, idx, int_img, lvl, group_id = 1, len(scenarios)-1, (number+1)*15, 1, 1
        
        predicted_groups = []
    
        if 'baseline' in args.prompt_method:
            groups = data['groups']


            if args.output_mode == 'coarse':
                for group in groups:
                    flag = False
                    xmin, ymin,xmax, ymax = W+H,W+H,0,0
                    for person in group:
                        if len(person) == 4:
                            x1, y1, x2, y2 = min(person[0], person[2]), min(person[1], person[3]),max(person[0], person[2]),max(person[1], person[3])
                            if x1 == x2 or y1 == y2:
                                continue
                            if x1>=0 and x2<=W and y1>=0 and y2<=H:
                                flag = True
                                xmin, ymin, xmax, ymax = min(x1,xmin), min(y1,ymin), max(x2,xmax), max(y2,ymax)
                    if flag:
                        add_prediction(det_file, idx, int_img, xmin, ymin, xmax, ymax, group_id, dc, lvl)
                        group_id+=1

            elif args.output_mode == 'finegrained':
                for group in groups:
                    predicted_group = []
                    flag = False
                    for person in group:
                        if len(person) == 4:
                            x1, y1, x2, y2 = person
                            if x1 == x2 or y1 == y2:
                                continue
                            if x1>=0 and x2<=W and y1>=0 and y2<=H:
                                flag = True
                                add_prediction(det_file, idx, int_img, min(x1,x2), min(y1,y2), max(x2,x1), max(y1,y2), group_id, dc, lvl)
                                predicted_group.append([min(x1,x2), min(y1,y2), max(x2,x1), max(y1,y2)])
                    if flag: 
                        group_id+=1
                        predicted_groups.append(predicte_group)
    
        else:    
            groups          = data['groups']
            personid2bbox   = data['id_tobbox']
            #print(personid2bbox)

            if args.output_mode == 'coarse':
                all_persons = set()
                for group in groups:
                    flag = False
                    xmin, ymin,xmax, ymax = W+H,W+H,0,0
                    for person in group:
                        if str(person) in personid2bbox and str(person) not in all_persons:
                            flag = True
                            all_persons.add(str(person))
                            x1, y1, x2, y2 = personid2bbox[str(person)]
                            xmin, ymin, xmax, ymax = min(x1,xmin), min(y1,ymin), max(x2,xmax), max(y2,ymax)
        
                    if flag:
                        add_prediction(det_file, idx, int_img, xmin, ymin, xmax, ymax, group_id, dc, lvl)
                        group_id+=1
           
        
            elif args.output_mode == 'finegrained':

                all_persons = set()
                for group in groups:
                    predicted_group = []
                    flag = False
                    for person in group:
                        if str(person) in personid2bbox and str(person) not in all_persons:
                            flag = True
                            all_persons.add(str(person))
                            x1, y1, x2, y2 = personid2bbox[str(person)]
                            add_prediction(det_file, idx, int_img, x1, y1, x2, y2, group_id, dc, lvl)
                            predicted_group.append([x1,y1,x2,y2])
                    if flag:
                        group_id+=1
                        predicted_groups.append(predicted_group)

                predicted_group = []
                for key, value in personid2bbox.items():
                    if key not in all_persons:
                        predicted_group = []
                        x1, y1, x2, y2 = value
                        add_prediction(det_file, idx, int_img, x1, y1, x2, y2, group_id, dc, lvl)
                        predicted_group.append([x1,y1,x2,y2])
                        group_id+=1
                        predicted_groups.append(predicted_group)

        #print(file)
    
        save_path = f"{group_path}/{str(args.frame_id)}_{file.split('/')[-1]}.pkl"
        with open(save_path, "wb") as f:
            pickle.dump(predicted_groups, f)
