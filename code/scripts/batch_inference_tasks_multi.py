import os
import sys
import subprocess
import argparse
from tqdm.auto import tqdm

# <json path> <frame dir> <vl model> 
#/home/pcascanteb/workspace/llm-crowd-detection/results/Qwen2.5-72B-Instruct/llm/detany_3D/p1 /home/pcascanteb/data/VBIG_dataset/videos_frames Qwen/Qwen2.5-VL-72B-Instruct
#/home/pcascanteb/workspace/llm-crowd-detection/results/Qwen2.5-32B-Instruct/llm/detany_3D/p1/ /home/pcascanteb/data/VBIG_dataset/videos_frames baidu/ERNIE-4.5-VL-28B-A3B-PT

# get these using argparse
# now add optional start and end index
parser = argparse.ArgumentParser()
parser.add_argument('json_dir', type=str)
parser.add_argument('video_dir', type=str)
parser.add_argument('model', type=str)
parser.add_argument('--start', type=int, default=None)
parser.add_argument('--end', type=int, default=None)
args = parser.parse_args()

json_dir = args.json_dir
video_dir = args.video_dir
model = args.model

files = [os.path.join(json_dir, f, f+'.json') for f in os.listdir(json_dir) if os.path.isdir(os.path.join(json_dir, f))]

if args.start is not None:
    files = files[args.start:]
if args.end is not None:
    files = files[:args.end]

for file in tqdm(files):
    print(f'Processing {file}')
    cmd = [
        'python', 'code/inference_tasks_multi.py',
        file,
        model,
        '--frame_ids', '1',
        '--frame_path', video_dir,
        '--max_tokens', '24000',
    ]
    subprocess.run(cmd)
