import glob
import json

path = '/home/artcs/Desktop/llm-crowd-detection/code/results/Qwen2.5-VL-3B-Instruct/vlm_text/naive_3D_60FOV/p1/*'
files = glob.glob(path)
files.sort()

scenarios = set()

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

    groups          = data['groups']
    personid2bbox   = data['id_tobbox']

    lvl= 1
    group_id = 1
    all_persons = set()
    for group in groups:
        for person in group:
            if str(person) in personid2bbox:
                all_persons.add(str(person))
                x1, y1, x2, y2 = personid2bbox[str(person)]
                PRED_list = [idx, int_img, x1, y1, x2, y2, group_id, dc, lvl]
                str_to_be_added = [str(k) for k in PRED_list]
                str_to_be_added = (" ".join(str_to_be_added))
                f = open('det.txt', "a+")
                f.write(str_to_be_added + "\r\n")
                f.close()
        group_id+=1

    for key, value in personid2bbox.items():
        if key not in all_persons:
            x1, y1, x2, y2 = value
            PRED_list = [idx, int_img, x1, y1, x2, y2, group_id, dc, lvl]
            str_to_be_added = [str(k) for k in PRED_list]
            str_to_be_added = (" ".join(str_to_be_added))
            f = open('det.txt', "a+")
            f.write(str_to_be_added + "\r\n")
            f.close()
            group_id+=1
