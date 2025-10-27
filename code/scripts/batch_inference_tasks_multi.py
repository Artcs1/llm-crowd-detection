import os
import sys
import subprocess
from tqdm.auto import tqdm

# <json path> <frame dir> <vl model> 
#/home/pcascanteb/workspace/llm-crowd-detection/results/Qwen2.5-72B-Instruct/llm/detany_3D/p1 /home/pcascanteb/data/VBIG_dataset/videos_frames Qwen/Qwen2.5-VL-72B-Instruct
json_dir = sys.argv[1]
video_dir = sys.argv[2]
model = sys.argv[3]

files = [os.path.join(json_dir, f, f+'.json') for f in os.listdir(json_dir) if os.path.isdir(os.path.join(json_dir, f))]

for file in tqdm(files):
    cmd = [
        'python', 'code/inference_tasks_multi.py',
        file,
        model,
        '--frame_ids', '1,25,50',
        '--frame_path', video_dir,
        '--max_tokens', '24000',
    ]
    subprocess.run(cmd)