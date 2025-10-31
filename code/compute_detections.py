import glob
import json
import os
import argparse

parser = argparse.ArgumentParser(description="Select mode, prompt method, model, and VLM mode")
parser.add_argument("--mode", type=str, choices=["single","full"], required=True, help="Mode: single or full")
parser.add_argument("--prompt_method", type=str, choices=["baseline1","baseline2","p1","p2","p3","p4"], required=True, help="Prompt method")
parser.add_argument("--model", type=str, required=True, help="Specify the model name or path")
parser.add_argument("--vlm_mode", type=str, choices=["llm","vlm_image","vlm_text"], required=True, help="VLM mode: image or text")
args = parser.parse_args()

base_dir = "/home/artcs1/Desktop/llm-crowd-detection/code"

H = 480
W = 3760

results_folder = "results" if args.mode == "single" else f"results_{args.mode}"

path = os.path.join(
    base_dir,
    results_folder,            # results or results_full
    args.model,                # e.g., Qwen2.5-VL-3B-Instruct
    args.vlm_mode,             # vlm_image or vlm_text
    "naive_3D_60FOV",         # fixed subfolder
    args.prompt_method,        # baseline1, p1, etc.
    "*"                        # wildcard for files
)

files = glob.glob(path)
print(path)
files.sort()
print(len(files))

scenarios = set()
det_file = f"detection_files/{results_folder}_{args.model}_{args.vlm_mode}_3D_{args.prompt_method}.txt"

if not os.path.exists(det_file.split('/')[0]):
    os.makedirs(det_file.split('/')[0])

if os.path.exists(det_file):
    os.remove(det_file)

for ind, file in enumerate(files):
    last = file.split('/')
    json_file = f'{file}/{last[-1]}.json'

    with open(json_file, 'r') as f:
        data = json.load(f)

    scenario      = last[-1]

    split_scenario = scenario.split('_')
    orig_scenario = "".join(split_scenario[:-1]) 
    scenarios.add(orig_scenario)
    number   = int(split_scenario[-1])

    dc = 1
    idx = len(scenarios)-1
    int_img = (number+1)*15
    lvl= 1
    group_id = 1


    if 'baseline' in args.prompt_method:
        groups = data['groups']
        for group in groups:
            for person in group:
                if len(person) == 4:
                    x1, y1, x2, y2 = person
                    if x1 == x2 or y1 == y2:
                        continue
                    if x1>=0 and x2<=W and y1>=0 and y2<=H:
                        PRED_list = [idx, int_img, min(x1,x2), min(y1,y2), max(x1,x2), max(y1,y2), group_id, dc, lvl]
                        str_to_be_added = [str(k) for k in PRED_list]
                        str_to_be_added = (" ".join(str_to_be_added))
                        f = open(det_file, "a+")
                        f.write(str_to_be_added + "\r\n")
                        f.close()
            group_id+=1
        
    else:    

        groups          = data['groups']
        personid2bbox   = data['id_tobbox']

        all_persons = set()
        for group in groups:
            flag = False
            for person in group:
                if str(person) in personid2bbox and str(person) not in all_persons:
                    flag = True
                    all_persons.add(str(person))
                    x1, y1, x2, y2 = personid2bbox[str(person)]
                    PRED_list = [idx, int_img, x1, y1, x2, y2, group_id, dc, lvl]
                    str_to_be_added = [str(k) for k in PRED_list]
                    str_to_be_added = (" ".join(str_to_be_added))
                    f = open(det_file, "a+")
                    f.write(str_to_be_added + "\r\n")
                    f.close()
            if flag:
                group_id+=1

        for key, value in personid2bbox.items():
            if key not in all_persons:
                x1, y1, x2, y2 = value
                PRED_list = [idx, int_img, x1, y1, x2, y2, group_id, dc, lvl]
                str_to_be_added = [str(k) for k in PRED_list]
                str_to_be_added = (" ".join(str_to_be_added))
                f = open(det_file, "a+")
                f.write(str_to_be_added + "\r\n")
                f.close()
                group_id+=1

