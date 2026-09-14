import os
import json
import argparse
import numpy as np
import pandas as pd

from tqdm.auto import tqdm
from cultural_utils import *



# add two arguments: data_dir and model_name using argparse
parser = argparse.ArgumentParser()
parser.add_argument("data_dir")
parser.add_argument("model_name")
args = parser.parse_args()

data_dir = args.data_dir
model_name = args.model_name

vid_dir = os.path.join(data_dir, 'videos')
clipdata_dir = os.path.join(data_dir, 'jsons_step1')

res_file = os.path.join(data_dir, 'results_cultural', model_name, 'annotations.json')

# Load the cultural annotations
with open(res_file, 'r') as f:
    cultural_data = json.load(f)


rows = []

for i in tqdm(range(len(cultural_data['annotations']))):
    annotation = cultural_data['annotations'][i]
    videoName = annotation['videoFolder'].split('/')[-2]
    frame = annotation['videoInfo']['annotationFrame']+1
    clipInfo_path = os.path.join(clipdata_dir, videoName + ".json")

    l_ga, l_gc, l_ghh, l_gh = [], [], [], []
    for g in annotation['groups']:
        co = g['cultural_output']
        ga, gc, ghh, gh = co['group_activity'], co['group_clothing'], co['group_handholding'], co['group_hugging']
        if ga is not None:
            l_ga.extend([a.lower() for a in ga])
        if gc is not None:
            l_gc.extend([c.lower() for c in gc])
        l_ghh.append(1 if ghh else 0)
        l_gh.append(1 if gh else 0)

    with open(clipInfo_path, 'r') as f:
        clipInfo = json.load(f)

    dset = clipInfo['dataset']
    density = clipInfo['density']
    city = clipInfo['city']
    country = clipInfo['country']
    source = clipInfo['source'].split('/')[-1]

    vid, clip = source.rsplit('_', 1)

    df_row = {
        "clip": videoName,
        "clip_index": int(videoName.split("_")[-1]),
        "frame": frame,
        "dataset": dset,
        "density": density,
        "city": city,
        "country": country,
        "num_groups": len(annotation['groups']),
        "activity": l_ga,
        "clothing": l_gc,
        "handholding": sum(l_ghh),
        "handholding_binary": 1 if sum(l_ghh) > 0 else 0,
        "hugging": sum(l_gh),
        "hugging_binary": 1 if sum(l_gh) > 0 else 0,
        "file": source,
        "video": vid,
        }
    
    rows.append(df_row)


df = pd.DataFrame(rows)

df['country'] = df['country'].replace('UAE', 'United Arab Emirates')
df['country'] = df['country'].replace('USA', 'United States')
df['country'] = df['country'].replace('Korea', 'South Korea')
df['country'] = df['country'].replace('Czechia', 'Czech Republic')

df['globe_region'] = df['country'].map(country_to_region_globe)

df.to_pickle(os.path.join(data_dir, 'results_cultural', model_name, 'pd_annotations.pkl'))
